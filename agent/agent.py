# agent/agent.py
from tqdm import tqdm
import os

class Agent:
    def __init__(self, state_dim, action_dim):
        """
        初始化智能体

        :param action_space: 动作空间
        :param state_space: 状态空间
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = "models"  # 模型参数存储路径
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def choose_action(self, state):
        """
        根据当前状态选择一个动作。具体的选择策略会在子类中实现。

        :param state: 当前状态
        :return: 选择的动作
        """
        raise NotImplementedError("choose_action方法必须在子类中实现")

    def update(self, state, action, reward, next_state, done):
        """
        更新策略的核心方法。具体的更新逻辑会在子类中实现。

        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否回合结束
        """
        raise NotImplementedError("update方法必须在子类中实现")

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
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # 记录当前 episode 的累计 return
            returns.append(total_reward)

        return returns
    
    def save_model(self, filename):
        """保存模型参数到本地"""
        raise NotImplementedError("save_model方法必须在子类中实现")

    def load_model(self, filename):
        """加载本地模型参数"""
        raise NotImplementedError("load_model方法必须在子类中实现")
