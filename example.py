import numpy as np
import random

def gym_example(usage_example, env, actions):
    if usage_example:
        #set the number of games (episodes) to play
        games = 2
        for game in range(games):
            #reset the game at the beginning
            state = env.reset()
            #check if we end the game
            done = False
            #keep track of the total score
            tot_reward = 0
            #create array of action index
            actions_idx = np.arange(0, actions, dtype=int)
            #print(actions_idx)
            #loop till the game is not finished
            while not done:
                #render the environment
                env.render()
                #select random action
                action_taken = env.action_space.sample()
                #env.step() perform the action in the environment, it returns:
                # observation (n_state) =  this will be an element of the environment’s observation_space. This may, for instance, be a numpy array containing the positions and velocities of certain objects.
                # reward =  The amount of reward returned as a result of taking the action
                # Terminated (done) =  whether a terminal state (as defined under the MDP of the task) is reached. In this case further step() calls could return undefined results.
                # info =  info contains auxiliary diagnostic information.  This might, for instance, contain: metrics that describe the agent’s performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward. 
                n_state, reward, done, _ = env.step(action_taken)
                print(f'Observation shape: \n {n_state.shape} \n Reward: {reward} \n Done: {done} \n')
                #reward +=1 since the pole does not falling down
                tot_reward += reward
        print('tot_reward: ', tot_reward)

        env.close()
    