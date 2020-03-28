'''
This is an example code for (Multivariate) deep robust regression (under covariate shift). 
The density ratio is estimated independently and "plugged in" the framework.
So we just set all density ratio to be 1 in this example and apply the method
to samples in the same dataset for training and testing, a.k.a mimic an iid (no shift) case.

'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torch.utils.data as data
import os
import copy
torch.set_default_tensor_type('torch.DoubleTensor')


'''
d is the dimension of the last layer of network

'''

d = 10

'''
d_out is the dimension of the output variable

'''

d_out = 3


'''
mean0 and var0 is set according to prior knowledge and will not
affect the result a lot in the no shift case.
However, if there is shift and target data is far from source,
base distribution will dominate the result.

'''

mean0 = np.zeros([1, d_out])
var0 = np.eye(d_out)

def predict_regression(weight, Myy, Myx, output, mean0, var0, device):
    '''
      Given output from network, make prediction of mean and variance

      mean0: (batch, d_out)
      var0: (batch, d_out, d_out)
      output: batch * d
      Myy: d_out*d_out
      Myx: d+1, d_out
      weight: batch * 1

      output:
      meanY : (batch, d_out, 1)
      varY : (batch, d_out, d_out)

    '''
    bs = np.shape(output)[0]
    x_1 = torch.cat((output, torch.ones((bs, 1)).to(device)), 1)
    Myy = torch.tensor(Myy)
    Myx = torch.tensor(Myx)

    #varY : (batch, d_out, d_out)

    weight = weight.unsqueeze_(-1)
    weight = weight.unsqueeze_(-1)

    weight = weight.expand(bs,1,1).to(device)
    Myy = Myy.unsqueeze_(0)
    Myy = Myy.expand(bs, d_out, d_out).to(device)
    # weight (bs, 1, 1)
    # Myy (bs, d_out, d_out)
    varY = torch.inverse((2.0 *weight*Myy + torch.inverse(torch.tensor(var0).to(device))))

    #x_1 (bs, 1, d+1)
    x_1 = x_1.unsqueeze_(1)
    x_1 = x_1.expand(bs, 1, d+1).to(device)
    # Myx = (bs, d+1, d_out)
    Myx = Myx.unsqueeze_(0)
    Myx = Myx.expand(bs, d+1, d_out).to(device)
    
    
    temp = torch.tensor(np.matmul(mean0,np.linalg.inv(var0)))
    temp = temp.unsqueeze_(0)
    temp = temp.expand(bs, 1, d_out).to(device)
    meanY = torch.bmm((-2.0*(torch.bmm(weight, torch.bmm(x_1, Myx)))+ temp), varY)
    # (bs, 1, d_out)
    meanY = meanY.squeeze(1)

    return meanY, varY

     
def M_gradient(x, meanY, varY, y, Myy, Myx):
    '''
     Calculate gradient for additional parameters, Myy and Myx
     for a batch of data
    
    '''

    bs = np.shape(x)[0]
    d = np.shape(x)[1]
    d_out = np.shape(y)[1]
    x = np.reshape(x.detach().numpy(), (bs, d))
    y = np.reshape(y.detach().numpy(), (bs, d_out))

    meanY = np.reshape(meanY.detach().numpy(), (bs, d_out))
    varY = np.reshape(varY.detach().numpy(), (bs, d_out, d_out))

    y_vec = np.concatenate((np.reshape(y, (bs, d_out)), x, np.ones((bs, 1))), axis = 1)
    emp = np.einsum('bi,bj->bij', y_vec, y)
    
    y_mean_vec = np.concatenate((np.reshape(meanY, (bs, d_out)), x, np.ones((bs, 1))), 1)
    var_vec = np.concatenate((varY, np.zeros((bs, d+1, d_out))), 1)
    exp = np.einsum('bi,bj->bij', y_mean_vec, meanY) + var_vec
    
    grad =  np.mean(exp, 0) - np.mean(emp, 0)
    return np.reshape(grad, (d+d_out+1, d_out))


class regression_gradient(torch.autograd.Function):
    '''
      back-prop gradients to network
      the gradient for "phi(x)" -- representing network -- is (y-mean_y)* Myx[0:-1]
    
    '''

    @staticmethod
    def forward(ctx, x, M, y, meanY):
        ctx.save_for_backward(x, M, y, meanY)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        
        x, M, y, meanY= ctx.saved_tensors
        # (bs, d_out)
        grad_x = grad_output.clone()
        # (bs, d) (bs, d_out) * ( d_out, d)
        
        grad_x = grad_x * torch.matmul((y - meanY), M.reshape(-1, d))/grad_x.shape[0]
        
        return grad_x, None, None, None


class Net(nn.Module):
    '''
       defining the network structure

    '''
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.D_in, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H, self.D_out),
            )

    def forward(self, x):
    
        x = x.view(-1, self.D_in)
        x = self.model(x)
        return x

def train_regression(args, model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0):
    '''
     training the model,  Myy and Myx is trained using batch gradient descent with learning rate decay
     (different learning rate can be used)
    
     Myy: d_out, d_out
     Myx: d+1 , d_out
    
    '''
    model.train()
    
    # constraints on Myy to make final variance above 0
    lowerB = -np.linalg.inv(2*var0)
    
    # recording gradients
    grad_yy = np.empty([0, d_out, d_out])
    grad_yx = np.empty([0, d+1, d_out])

    # learning rate for Myx
    lr2 = args.lr
    
    # decay rate
    lr2 = lr2 * (10 / (10 + np.sqrt(epoch)))
    
    # learning rate for Myy
    lr1 = args.lr
    lr1 = lr1 * (10 / (10 + np.sqrt(epoch)))

    for batch_idx, (data, target, weight) in enumerate(train_loader):
        
        data, target, weight = data.to(device), target.to(device), weight.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # predict current meanY and varY
        meanY, varY = predict_regression(weight, Myy, Myx, output, mean0, var0, device)
        
        # calculate gradients for Myy and Myx
        grad = M_gradient(output, meanY, varY, target, Myy, Myx)
        
        grad_yy = np.concatenate((grad_yy, np.expand_dims(grad[0:d_out, :], 0)), 0)
        grad_yx = np.concatenate((grad_yx, np.expand_dims(grad[d_out:, :],0)), 0)

        diff = lr1*(grad[0:d_out, :]) 
        Myy = Myy + diff
        
        #check whether Myy is valide
        # while True:
        #     try: 
        #         test = np.linalg.cholesky(Myy - lowerB)
        #         print(test)
        #         break
        #     except:
        #         raise
        #         Myy = Myy + np.abs(diff)/2

        Myx = Myx + lr2*grad[d_out:, :]

        # batch size?
        bs = np.shape(output)[0]
        
        # get gradient for "phi(x)" and back-prop
        output_last = regression_gradient.apply(output, torch.tensor(Myx[0:-1]), torch.reshape(target, (bs, d_out)), torch.reshape(meanY, (bs, d_out)))
        output_last.backward(torch.ones(output_last.shape),retain_graph=True)
        
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))
            
    print('gradient yy:', np.linalg.norm(np.mean(grad_yy, 0)))
    print('gradient yx:', np.linalg.norm(np.mean(grad_yx, 0)))

    return Myy, Myx


def test_regression(args, model, Myy, Myx, device, test_loader, mean0, var0):
    '''
      test model

      testloss is the L-2 loss

    '''
    model.eval()
    test_loss = 0
    y_prediction = np.empty([0, d_out])
    y_var = np.empty([0, d_out, d_out])
    with torch.no_grad():
        for data, target, weight in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            d = np.shape(data)[0]
            target = torch.reshape(target, (-1, d_out))
            
            meanY, varY = predict_regression(weight, Myy, Myx, output, mean0, var0, device)
            # loss =  -np.log(1/(np.sqrt(varY)*np.sqrt(2*3.14)))+(target-meanY).pow(2)/(2*varY)
            loss = torch.sum((target-meanY).pow(2), 1)
            
            test_loss += torch.sum(loss)
            y_prediction = np.concatenate((y_prediction, meanY), axis=0)

            y_var = np.concatenate((y_var, varY), axis = 0)

    test_loss = torch.sqrt(test_loss / len(test_loader.dataset))
    print('Average loss: {:.4f}\n'.format(test_loss))

    return y_prediction, y_var, test_loss

def train_validate_test(args, epoch, device, use_cuda, train_model, train_loader, test_loader, validate_loader, lbd):
    '''
    train, validate, and test the model

    '''
    d_out = len(train_loader.dataset[0][1])
    
    Myy = np.ones((d_out, d_out))
    Myx = np.ones((d+1, d_out))

    '''
    mean0 and var0 is set according to prior knowledge and will not
    affect the result a lot in the no shift case.
    However, if there is shift and target data is far from source,
    base distribution will dominate the result.

    '''

    mean0 = np.zeros([1, d_out])
    var0 = np.eye(d_out)
    optimizer = optim.SGD(train_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=lbd)
    for epoch in range(1, epoch + 1):
        Myy, Myx = train_regression(args, train_model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0) 
        
        # in the no shift case, we can do validation, but in the shift case, it may not make sense
        meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, validate_loader,mean0, var0 )
        
    print('\nTesting on test set')

    meanY, var, loss = test_regression(args, train_model, Myy, Myx, device, test_loader, mean0, var0)
    return train_model, Myy, Myx, meanY, var, loss


class Mydata(data.Dataset):
    '''
        Read data

    '''
    def __init__(self, filename):
        raw_data = np.loadtxt(filename)
        self.raw_data = raw_data

    def __getitem__(self, index):
        features, target = self.raw_data[index][0:-2], self.raw_data[index][-2:]
        return features, target

    def __len__(self):
        return len(self.raw_data)

    def get_raw_data(self):
        return self.raw_data

class SplitData(data.Dataset):
    '''
        Read data
    '''
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return len(self.features)


class WeightDataSet(data.Dataset):
    '''
       dataset class with instance weight 

    '''
    def __init__(self, original_data, weights):
        '''
           weights are same dimensional with original data

        '''
        self.data = original_data
        self.weights = weights

    def __getitem__(self, index):

        img, target = self.data[index] 

        weight = self.weights[index]

        return img, target, weight

    def __len__(self):
        return len(self.data)

def main():
    '''
       Training settings

    '''
    parser = argparse.ArgumentParser(description='Covariate Shift')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs-training', type=int, default=100, metavar='N',
                        help='number of epochs in training (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    rawdata = Mydata("housing.data")
    rawdata = np.random.permutation(rawdata)
    train_data = data.Subset(rawdata, range(0, int(0.8*len(rawdata))))
    
    # 20% test set
    test_data = data.Subset(rawdata, range(int(0.8*len(rawdata)), len(rawdata)))

    m_train = len(train_data)
    
    # for covariate shift, plug in densities for source data
    weight_st = np.ones(m_train)
    
    weighted_train = WeightDataSet(train_data, weight_st)
    m_test = len(test_data)
    
    # for covariate shift, plug in densities for target data
    weight_st = np.ones(m_test)
    weighted_test = WeightDataSet(test_data, weight_st)

    test_loader = data.DataLoader(weighted_test,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # 10% of training as validation
    m_validate = int(0.1*m_train)

    validate_loader = data.DataLoader(data.Subset(weighted_train,range(0, m_validate)),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # 10% validation set, note that under shift cases, cannot valiate using training data
    train_loader = data.DataLoader(data.Subset(weighted_train, range(m_validate, m_train)),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    base_model = Net(12,256, d)
    base_model = base_model.to(device)

    model, Myy, Myx, y_pred, y_var, _  = train_validate_test(args, args.epochs_training,device, use_cuda, base_model, 
        train_loader, test_loader , validate_loader, 0.001)
            
if __name__ == '__main__':
    main()
