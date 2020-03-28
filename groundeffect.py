import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_tensor_type('torch.DoubleTensor')

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

def read_weight(filename):
    model_weight = torch.load(filename)
    model = Network()
    model.load_state_dict(model_weight)
    return model

fa_model = read_weight('Fa_net_12_3_full_Lip16.pth')

# state is a 12-d vector
# 0: height, z
# 1,2,3: velocity, \dot{x}, \dot{y}, \dot{z}
# 4,5,6,7: quaternion, qx, qy, qz, qw
# 8,9,10,11: control, u1, u2, u3, u4

# input range and constraints
# 0 <= z <= 1.5
# -1 <= \dot{x,y,z} <= 1 
# qx^2 + qy^2 + qz^2 + qw^2 = 1
# 0 <= u1,u2,u3,u4 <= 1

# *******************************************
# Task 1: learn a 4d -> 3d function
# fix qx,qy,qz,qw = 0,0,0,1
# fix u1,u2,u3,u4 = 0.7,0.7,0.7,0.7
# learn the function (z,\dot{x},\dot{y},\dot{z}) -> fa
def fa_output(fa_model, height, velocity, controls=[0.7] * 4):
	state = np.zeros([1, 12])
	state[0, 0] = height
	state[0, 1:4] = velocity
	state[0, 7] = 1.0
	state[0, 8:12] = controls
	state_torch = torch.from_numpy(state)
	print(state_torch)
	Fa = fa_model(state_torch)
	return Fa

print(fa_output(fa_model=fa_model, height=0.2, velocity=np.array([0,0,0])))

# *******************************************
# Task 2: learn a 8d -> 3d function
# fix qx,qy,qz,qw = 0,0,0,1
# learn the function (z,\dot{x},\dot{y},\dot{z},u1,u2,u3,u4) -> fa