import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ##### TODO ######
        ### Complete definition 
        self.lin1 = nn.Linear(4, 128)
        self.lin2 = nn.Linear(128, 2)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, x):
        ##### TODO ######
        ### Complete definition
        out1 = F.relu(self.lin1(x))
        out1 = F.softmax(self.lin2(out1), -1)

        #critic
        out2 = F.relu(self.lin1(x))
        out2 = self.lin3(out2)

        return out1, out2


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action

def sample_episode():

    state, ep_reward = env.reset(), 0
    episode = []

    for t in range(1, 10000):  # Run for a max of 10k steps

        action = select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())

        episode.append((state, action, reward))
        state = next_state

        ep_reward += reward

        if args.render:
            env.render()

        if done:
            break

    return episode, ep_reward

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
    

def compute_losses(episode):

    ####### TODO #######
    #### Compute the actor and critic losses
    actor_loss, critic_loss = None, None
    A = [] # list to store advantage values for every time step
    pol_list = []
    for i, eps in enumerate(episode):
        A_sum = 0
        state, action, reward = eps
        state = torch.Tensor(state)
        act, val = model(state)

        pol_list.append(act.max())
        for t_prime in range(i + 1, len(episode) + 1):
            A_sum += (args.gamma ** (t_prime - i + 1)) * episode[i][2] 
        A_sum -= val # model(state) is the value function for each 
        A.append(A_sum)
    A = torch.stack(A)
    actor_loss = -torch.sum(torch.log(torch.stack(pol_list)) * A)
    critic_loss = torch.sum(A ** 2)
    return actor_loss, critic_loss

def main():
    running_reward = 10
    loss_history = []
    for i_episode in count(1):

        episode, episode_reward = sample_episode()

        optimizer.zero_grad()

        actor_loss, critic_loss = compute_losses(episode)

        loss = actor_loss + critic_loss

        loss.backward()

        optimizer.step()

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        loss_history.append(running_reward)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(episode)))
            # episodes = [i + 1 for i in range(len(loss_history))]
            # plt.plot(episodes, loss_history)
            # plt.show()
            plot_durations(loss_history)
            plt.savefig('reward_history.png')
            break


if __name__ == '__main__':
    main()
