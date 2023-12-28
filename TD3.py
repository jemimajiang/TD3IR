import cv2
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random

class DilatedConvBlock(nn.Module):
    def __init__(self,dilate):
        super(DilatedConvBlock,self).__init__()
        self.diconv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=dilate,dilation=dilate, bias=True)
    def forward(self,x):
        h=F.relu(self.diconv(x))
        return h

class Actor(nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()
        self.max_action = int(max_action)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64,3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            DilatedConvBlock(2),
            DilatedConvBlock(3),
            DilatedConvBlock(4),
            DilatedConvBlock(3),
            DilatedConvBlock(2),

        )
        self.conv_Wz = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv_Uz = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.conv_Wr = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv_Ur = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.conv_W = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv_U = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=True)
        self.MLP = nn.Conv2d(1, self.max_action, 3, stride=1, padding=1, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, s, ht_1):
        # s为当前状态，ht_1为历史隐藏层
        # 经过扩张卷积 1，2，3，4，3，2
        x_t = self.layers(s)
        # 计算GRU，得到新的隐藏层输出 h_t
        z_t = F.sigmoid(self.conv_Wz(x_t)+self.conv_Uz(ht_1))
        r_t = F.sigmoid(self.conv_Wr(x_t)+self.conv_Ur(ht_1))
        h_tilde_t = F.tanh(self.conv_W(x_t) + self.conv_U(r_t * ht_1))
        h_t = (1 - z_t) * ht_1 + z_t * h_tilde_t

        # 经过两个conv1和softmax，确定策略梯度得到下一个状态
        s = F.relu(self.conv1(h_t))
        s = F.softmax(self.MLP(s))
        # 修改隐藏层到 tensor{1,1,h,w}
        h_t = torch.mean(h_t, dim=1)
        return s, h_t


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self):
        super(Critic, self).__init__()
        # GRU
        self.conv_Wz = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv_Uz = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.conv_Wr = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv_Ur = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.conv_W = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv_U = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=True)
        self.diconv2 = DilatedConvBlock(2)
        # Q1
        self.layer1 = nn.Sequential(
            nn.Conv2d(2,64,3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            DilatedConvBlock(2),
            DilatedConvBlock(3),
            DilatedConvBlock(4),
            DilatedConvBlock(3),
            # DilatedConvBlock(2),
            # nn.Conv2d(64,1,3,stride=1,padding=1,bias=True)
        )
        # Q2
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            DilatedConvBlock(2),
            DilatedConvBlock(3),
            DilatedConvBlock(4),
            DilatedConvBlock(3),
            # DilatedConvBlock(2),
            # nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=True)
        )

    def forward(self, s, a, ht_1):
        s_a = torch.cat([s, a], 1)
        q1 = self.layer1(s_a)
        # 计算GRU，得到新的隐藏层输出 h_t
        z_t = F.sigmoid(self.conv_Wz(q1) + self.conv_Uz(ht_1))
        r_t = F.sigmoid(self.conv_Wr(q1) + self.conv_Ur(ht_1))
        h_tilde_t = F.tanh(self.conv_W(q1) + self.conv_U(r_t * ht_1))
        h_t_q1 = (1 - z_t) * ht_1 + z_t * h_tilde_t
        q1 = F.relu(self.conv1(self.diconv2(h_t_q1)))
        h_t_q1 = torch.mean(h_t_q1, dim=1)

        q2 = self.layer2(s_a)
        # 计算GRU，得到新的隐藏层输出 h_t
        z_t = F.sigmoid(self.conv_Wz(q2) + self.conv_Uz(ht_1))
        r_t = F.sigmoid(self.conv_Wr(q2) + self.conv_Ur(ht_1))
        h_tilde_t = F.tanh(self.conv_W(q2) + self.conv_U(r_t * ht_1))
        h_t_q2 = (1 - z_t) * ht_1 + z_t * h_tilde_t
        q2 = F.relu(self.conv1(self.diconv2(h_t_q2)))
        h_t_q2 = torch.mean(h_t_q2, dim=1)

        return q1, q2, h_t_q1, h_t_q2

    def Q1(self, s, a,ht_1):
        s_a = torch.cat([s, a], 1)
        q1 = self.layer1(s_a)
        # 计算GRU，得到新的隐藏层输出 h_t
        z_t = F.sigmoid(self.conv_Wz(q1) + self.conv_Uz(ht_1))
        r_t = F.sigmoid(self.conv_Wr(q1) + self.conv_Ur(ht_1))
        h_tilde_t = F.tanh(self.conv_W(q1) + self.conv_U(r_t * ht_1))
        h_t_q1 = (1 - z_t) * ht_1 + z_t * h_tilde_t
        q1 = F.relu(self.conv1(self.diconv2(h_t_q1)))
        h_t_q1 = torch.mean(h_t_q1, dim=1)

        return q1, h_t_q1


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e5)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size,) + state_dim)
        self.a = np.zeros((self.max_size,) +action_dim)
        self.r = np.zeros((self.max_size,) + state_dim)
        self.s_ = np.zeros((self.max_size,) +state_dim)
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.batch_size = 6  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 7e-4  # learning rate
        self.policy_noise = 0.2 * max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * max_action  # Clip the noise
        self.policy_freq = 1000  # The frequency of policy updates
        self.actor_pointer = 0

        self.actor = Actor(max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)



    def choose_action(self, s, ht_1):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        ht_1 = torch.unsqueeze(torch.tensor(ht_1, dtype=torch.float), 0)
        a_softmax, h_t = self.actor(s, ht_1)
        a = torch.argmax(a_softmax, dim=1)
        return a, h_t

    def learn(self, relay_buffer):
        self.actor.train()
        self.actor_target.train()
        self.actor_pointer += 1
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            ht_1 = torch.zeros(1, 1, batch_s.shape[2], batch_s.shape[3])
            a_softmax, _ = self.actor_target(batch_s_, ht_1)
            next_action = (torch.argmax(a_softmax, dim=1).unsqueeze(1) + noise)

            # Trick 2:clipped double Q-learning
            target_Q1, target_Q2,_,_ = self.critic_target(batch_s_, next_action,ht_1)
            # 扩展 dw 的形状为 (16, 1, 70, 70)
            expanded_dw = batch_dw.unsqueeze(2).unsqueeze(3).expand(-1, -1, 63, 63)
            target_Q = batch_r + self.GAMMA * (1 - expanded_dw) * torch.min(target_Q1, target_Q2)

        # Get the current Q
        current_Q1, current_Q2,_,_ = self.critic(batch_s, batch_a,ht_1)
        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_freq == 0:
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = True
            # Compute actor loss
            Q_value,_ = self.critic.Q1(batch_s, batch_a, ht_1)
            actor_loss = -Q_value.mean()  # Only use Q1
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # # Unfreeze critic networks
            # for params in self.critic.parameters():
            #     params.requires_grad = True

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(env, agent, total_steps):
    times = 1  # Perform three evaluations and calculate the average
    len = 5
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        ht_1 = np.zeros(s.shape)
        episode_reward = 0
        for epi_step in range(len):
            a,h_t = agent.choose_action(s,ht_1)  # We do not add noise when evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
            ht_1 = h_t

            # 保存结果s
            save_img = np.clip(s, 0, 1)
            save_img = (save_img[0] * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(f"runs/res/train_{total_steps:04d}_len_{epi_step:01d}.png", save_img)
        evaluate_reward += episode_reward

    return np.floor(evaluate_reward / times)

if __name__ == '__main__':
    from ImageEnv import ImageDenoisingEnv
    env = ImageDenoisingEnv()
    env_evaluate = ImageDenoisingEnv()
    number = 1

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    max_action = float(8)
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = TD3(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/TD3/TD3_envImage_number_{}'.format(number))

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 1e6  # Maximum number of training steps
    random_steps = 20  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training


    while total_steps < max_train_steps:
        s = env.reset()
        total_steps += 1
        episode_steps = 0
        ht_1 = np.zeros(s.shape)
        print("total_steps: {}".format(total_steps))
        while episode_steps < max_episode_steps:
            episode_steps += 1
            print("\tepisode_steps: {}".format(episode_steps))
            if total_steps < random_steps:  # Take random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                # Add Gaussian noise to action for exploration
                a,h_t = agent.choose_action(s,ht_1)
                ht_1 = h_t
                a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
            s_, r, done, _ = env.step(a)

            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            replay_buffer.store(s, a, r, s_, dw)  # Store the transition
            s = s_

            # Update one step
            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = np.mean(evaluate_policy(env_evaluate, agent,total_steps))
                evaluate_rewards.append(evaluate_reward)
                print("\tevaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_image', evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/TD3_envImage_number_{}.npy'.format(evaluate_num), evaluate_rewards)
