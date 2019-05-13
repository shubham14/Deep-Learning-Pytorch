import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb
from torch.distributions import Categorical 


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Create layers for the network described in the ###
        ### homework writeup                               ###
        ######################################################
        self.lin1 = nn.Linear(4, 24)
        self.lin2 = nn.Linear(24, 36)
        self.lin3 = nn.Linear(36, 1)
        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################


    def forward(self, x):
        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Forward through the network                    ###
        ######################################################
        x = torch.Tensor(x)
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.sigmoid(self.lin3(out))
        return out
        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def simulate(env, policy_net, steps, state_pool, action_pool, reward_pool,
             episode_durations):
    state = env.reset()
    state = torch.from_numpy(state).float()
    state = Variable(state)
    env.render(mode='rgb_array')

    for t in count():

        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Use policy_net to sample actions given the       #
        ### current state                                    #
        ######################################################
        out = policy_net(state)
        m = Bernoulli(out)
        action = m.sample()
        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################

        action = action.data.numpy().astype(int)[0]
        next_state, reward, done, _ = env.step(action)
        env.render(mode='rgb_array')

        # To mark boundarys between episodes
        if done:
            reward = 0

        state_pool.append(state)
        action_pool.append(float(action))
        reward_pool.append(reward)

        state = next_state
        state = torch.from_numpy(state).float()
        state = Variable(state)

        steps += 1

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

    return state_pool, action_pool, reward_pool, episode_durations, steps


def main():

    episode_durations = []

    # Parameters
    num_episode = 500
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.99

    env = gym.make('CartPole-v0')
    policy_net = PolicyNet()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    for e in range(num_episode):
        state_pool, action_pool, reward_pool, episode_durations, steps = simulate(
            env, policy_net, steps, state_pool, action_pool, reward_pool,
            episode_durations)
        # Update policy
        if e > 0 and e % batch_size == 0:

            # Discounted reward
            running_add = 0
            for i in reversed(range(steps)):
                ######################################################
                ###              START OF YOUR CODE                ###
                ######################################################
                ### Compute the discounted future reward for every   #
                ### step in the sampled trajectory and store them    #
                ### in the reward_pool list                          #
                ###################################################### 
                if reward_pool[i] == 0:
                    running_add = 0  
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add
                ######################################################
                ###               END OF YOUR CODE                 ###
                ######################################################

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std


            # Gradient Desent
            optimizer.zero_grad()
            policy_loss = []
            for i in range(steps):

                ######################################################
                ###              START OF YOUR CODE                ###
                ######################################################
                ### Implement the policy gradients objective using   #
                ### the state/action pairs acquired from the         #
                ### function  simulate(...) and the computed         #
                ### discounted future rewards stored in the          #
                ### reward_pool list and perform backward() on the   #
                ### computed objective for the optimier.step call    #
                ######################################################
                s, a, r = state_pool[i], Variable(torch.Tensor([action_pool[i]])), reward_pool[i]
                out = policy_net(s)
                m = Bernoulli(out)
                loss = -m.log_prob(a) * r 
                policy_loss.append(loss)
            
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            
                ######################################################
                ###               END OF YOUR CODE                 ###
                ######################################################

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0


if __name__ == '__main__':
    main()

