import torch
import matplotlib.pyplot as plt
import numpy as np

# ----CALCULATING FUTURE REWARDS ---- #

def discount_rewards(rewards, gamma):
    #series of time steps, i.e. action taken during one cart pole experiment
    len_rewards =len(rewards) 
    #compute exponentially decaying rewards, this is done because we want to reward more the first action since it is the least responsible for the pole
    #falling down and loosing the episode. So the last actions will be discounted more than the older actions.
    #(N.B.) This is an opposite reasoning than usually considering last rewards (i.e. the ones leading to the goal) more important than older one . 
    #Therefore this situation considers the cart pole task where when the pole falls we lose and the episode ends.
    disc_return = torch.pow(gamma,torch.arange(len_rewards).float()) * rewards
    #Normalizes the rewards to be within the [0,1] interval to improve numerical stability
    disc_return /= disc_return.max()
    return disc_return

# ---- LOSS FUNCTION -------#

#the loss function expects an array of action probabilities for the actions that were taken and the discounted rewards
def loss_function(preds, discounted_rewards):
    #it computes the log of the probabilities, multiplies by the discounted rewards, sums them all, and flips the sign.
    #make our loss function the negative log-probability of the action given the state scaled by the reward return 
    #this is because we want to update hyperparameters in such a way to maximixe the probability for the action we took.
    return -1* torch.sum(discounted_rewards * torch.log(preds))


def running_mean(x,N):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y

def plot(array, title, N):
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(15.5, 7.5)
    ax.set_title(title)
    y_axis_name = title.split(' ')
    y_a_name = ''
    for i in range(1,len(y_axis_name)):
        y_a_name+=y_axis_name[i]+' '
    ax.set_xlabel("Epochs")
    ax.set_ylabel(y_a_name)
    fig.set_size_inches(10,6)
    plt.plot(running_mean(array,N))
    plt.show()