import numpy as np
import random

class Agent:
    def __init__(self, action_space, state_space):
        """
        初始化智能体

        :param action_space: 动作空间
        :param state_space: 状态空间
        """
        self.action_space = action_space  # 动作空间
        self.state_space = state_space  # 状态空间

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
        """
        训练智能体与环境进行交互

        :param env: 环境
        :param num_episodes: 训练的回合数
        """
        for episode in range(num_episodes):
            state = env.reset()  # 重置环境，获取初始状态
            done = False

            while not done:
                action = self.choose_action(state)  # 根据当前状态选择一个动作
                next_state, reward, done, info = env.step(action)  # 执行动作，获取反馈
                self.update(state, action, reward, next_state, done)  # 更新智能体的策略
                state = next_state  # 更新状态

            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes} completed.")


# Agent子类：对于EasyTradingEnv，执行简单的策略，买入并一直持有
class BaselineAgent(Agent):
    def __init__(self, action_space, state_space):
        super(BaselineAgent, self).__init__(action_space, state_space)

    def choose_action(self, state):
        """
        根据基线策略选择动作：
        - 用全部资金买入股票。
        - 之后一直保持持有状态。

        :param state: 当前状态 [balance, stock_owned, stock_price]
        :return: 动作(0: 持有, 1: 买入, 2: 卖出)
        """
        return 1  # 剩余资金不够买入一手时，EasyTradingEnv会选择持有

    def update(self, state, action, reward, next_state, done):
        """
        基线策略无需更新，不做任何操作。

        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        pass
