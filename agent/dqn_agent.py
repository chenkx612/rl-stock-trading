# agent/dqn_agent.py
from agent import Agent
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm  
import os

# Agent子类：对于EasyTradingEnv，采用DQN算法
class DQNAgent(Agent):
    def __init__(self, state_dim, action_dim, epsilon=0.1, alpha=0.001, gamma=0.99, batch_size=64, memory_size=10000):
        super(DQNAgent, self).__init__(state_dim, action_dim)
        # 初始化网络、优化器和经验回放池
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)  # 经验回放池
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 或 CPU
        
        self.q_network = self.build_model().to(self.device)  # 主网络
        self.target_network = self.build_model().to(self.device)  # 目标网络
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.update_target_network()  # 初始化目标网络参数

    def build_model(self):
        """构建 Q 网络模型"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

    def update_target_network(self):
        """更新目标网络参数"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, is_training=True):
        """
        选择动作：在训练时使用 epsilon-greedy 策略，在回测时只选择最大 Q 值的动作
        :param state: 当前状态
        :param is_training: 是否处于训练模式
        :return: 选择的动作
        """
        if is_training and np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机选择动作（探索）
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # 选择 Q 值最大的动作（利用）

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """从经验回放池中采样"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).to(self.device)
        )

    def update(self):
        """单次更新 Q 网络"""
        if len(self.memory) < self.batch_size:
            return  # 如果经验不足一个批次，跳过更新

        states, actions, rewards, next_states, dones = self.sample_batch()

        # 计算当前 Q 值
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            target_q_value = rewards + self.gamma * target_q_values.max(1)[0] * (1 - dones)

        # 计算损失
        loss = nn.MSELoss()(q_value, target_q_value)

        # 更新主网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes=1000):
        """训练智能体，并记录每个 episode 的累计 return"""
        returns = []  # 用于记录每个 episode 的累计 return
        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            state = env.reset()
            done = False
            total_reward = 0  # 当前 episode 的累计 return

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                self.update()
                state = next_state
                total_reward += reward

            # 记录当前 episode 的累计 return
            returns.append(total_reward)

            # 每 10 个回合更新目标网络
            if episode % 10 == 0:
                self.update_target_network()

        return returns

    def save_model(self, filename):
        """保存模型参数到本地"""
        # 定义模型保存路径
        model_path = os.path.join(self.model_dir, filename) 

        # 保存模型
        torch.save(self.q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, filename):
        """加载本地模型参数"""
        model_path = os.path.join(self.model_dir, filename) 
        self.q_network.load_state_dict(torch.load(model_path, weights_only=True))
        self.q_network.to(self.device)
        print(f"Model loaded from {model_path}")
