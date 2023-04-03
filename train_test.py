import torch
import random
import numpy as np
from utils import *
from tqdm import tqdm

# ---- TRAIN ---- #

def train(env, max_duration, max_episodes, gamma, model, score, actions, optimizer, render_interval):
    for episode in tqdm(range(max_episodes)):
        #reset the environment at each episode
        current_state = env.reset()
        #check when episode ends
        done = False
        #list of state, action, rewards
        transitions = []

        #loop till the end of the episode max duration
        for t in range(max_duration):
            #compute probabilities over actions by using the policy network
            action_probs = model(torch.from_numpy(current_state).float())
            #select an action stochastically. Action with higher probabilities tend to be selected more than others.
            #this strategy ensures also some degree of exploration.
            action_taken = np.random.choice(actions, p=action_probs.data.numpy())
            #our current state becomes our prev state
            prev_state = current_state
            #take the action in the environment and get our new current state
            current_state, reward, done, _ = env.step(action_taken)
            #append the current transition in the array of transitions (as experience replay buffer)
            transition = (prev_state, action_taken, t+1)
            transitions.append(transition)
            #check if we lose the game 
            if done: break
            if episode % render_interval == 0:
                env.render()
        env.close()

        #determine the length of the episode given by the n.of actions taked without the pole falling
        ep_len = len(transitions)
        score.append(ep_len) 
        
        #extract batches as experience replay buffer
        #rewards batch = collects all the rewards in the episode in a single tensor
        rewards_batch = torch.Tensor(np.array([r for (s, a, r) in transitions])).flip(dims=(0,))
        #compute discounted rewards
        disc_rewards = discount_rewards(rewards_batch, gamma)
        #creating batches for both state and actions
        state_batch = torch.Tensor(np.array([s for (s, a, r) in transitions]))
        action_batch = torch.Tensor(np.array([a for (s, a, r) in transitions]))
        #recomputes the action probabilities for all the states in the episode
        pred_batch = model(state_batch)
        #gather all probabilites given the index of the action
        #it is basically the subsets the action-probabilities associated with the actions that were actually taken
        prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
        #compute loss
        loss = loss_function(prob_batch, disc_rewards)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    return model, np.array(score)


# ---- TEST ----- #

def test(n_games, env, trained_model, actions, max_duration, render_interval):
    scores_play = []
    for i in range(n_games):
        t=0
        done = False
        current_state  = env.reset()
        while not done:
            pred = trained_model(torch.from_numpy(current_state).float()) 
            action = np.random.choice(actions, p=pred.data.numpy()) 
            next_state, reward, done, _ = env.step(action) 
            current_state = next_state 
            t += 1
            if t > max_duration: break
            if i % render_interval == 0:
                env.render()
        env.close()
        scores_play.append(t)
    
    print('AVG score: ', np.mean(np.array(scores_play)))
    print('Best score: ', max(scores_play))
    