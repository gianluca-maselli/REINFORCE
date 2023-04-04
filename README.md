# REINFORCE

The REINFORCE algorithm is a **policy-gradient method** which does not rely on an epsilon-greedy strategy as DQN, but rather, directly sample actions from the policy. The idea is to use a **policy network** to output a probability distribution over the possibile action. Therefore at the beginning our agent has an uniform probability to extract one action with respect to the others. However, as the training goes on, the actions with higher probabilities are picked more often than the ones with lower probability. In other words we say that a particular action is "reinforced" with respect to the others. 

In order to update the policy network parameters the objective function defined as $$- \gamma_t * G_t \log \pi_s (a|\theta)$$ is minimized, where $\gamma_t$ is the discount factor with depends on the time step $t$ value and $G_t$ is the total return (or future return). 
In general actions temporally more distant from the received reward should be weighted less than action closer. However, in this case, when using Cart Pole env the reasoning is different. In fact, we need to discount more the last action since it is responsible of the pole falls. In order to deal with it, the discount is exponentially decayed from $1$ to $y_t = y_0^{T-t}$, where $T$ is the number of time steps. 

The Cart Pole experiment is considered solved if the agent can play an episode beyond 200 time steps. With the current configuration of the network, at test time, this result is achieved at least one time. 
