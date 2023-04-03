import torch 

'''
The policy network will accept state vectors as inputs, 
and it will produce a (discrete) probability distribution over the possible actions. 
In this case we directly select an action to perform in the environment. 
This class of algorithms is called **policy gradient methods. 
'''

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, fc1_size, fc2_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, fc1_size)
        self.fc2 = torch.nn.Linear(fc1_size, fc2_size)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=0)
        
    def forward(self, input):
        out = self.fc1(input)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out
