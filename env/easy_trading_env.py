# env/easy_traing_env.py
from env.trading_env import TradingEnv
import numpy as np
from gym import spaces
from collections import deque

# TradingEnv的子类，假设股市中只包含一种股票，每次买入或卖出单位为100股
class EasyTradingEnv(TradingEnv):
    def __init__(self, stock_data, initial_balance=10000, window_size=5):
        super(EasyTradingEnv, self).__init__(stock_data, initial_balance)

        self.stock_owned = 0  # 当前持有股票数量
        self.stock_price = 0  # 当前股价
        self.window_size = window_size # 记录股价趋势的时间窗口大小
        self.past_returns = deque([0] * self.window_size, maxlen=window_size)
        
        # 动作空间：持有（0）、买入（1）、卖出（2）
        self.action_space = spaces.Discrete(3)

        # 观察空间：包括 (balance, stock_owned, stock_price) 和过去 window_size 天的涨跌百分比
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [-1] * window_size),  
            high=np.array([np.inf, np.inf, np.inf] + [1] * window_size),  
            dtype=np.float32
        )

    def get_state(self):
        # 获取当前状态，包含 balance、stock_owned 和 stock_price，后跟过去 window_size 天的涨跌幅
        return np.array([self.balance, self.stock_owned, self.stock_price] + list(self.past_returns))

    def reset(self):
        super(EasyTradingEnv, self).reset()

        self.stock_owned = 0
        self.stock_price = self.stock_data.iloc[self.current_step]['close']
        self.past_returns = deque([0] * self.window_size, maxlen=self.window_size)

        return self.get_state()
    
    def step(self, action):
        # 计算昨日资产总价值，以便在之后计算reward
        pre_value = self.balance + self.stock_owned * self.stock_price 

        # 更新past_returns
        stock_data_row = self.stock_data.iloc[self.current_step]
        current_price = stock_data_row['close']  # 当前股票价格
        if self.current_step > 0:
            pct_change = (current_price - self.stock_price) / self.stock_price
            self.past_returns.append(pct_change)

        # 更新stock_price
        self.stock_price = current_price

        # 执行动作，更新balance, stock_owned
        if action == 1:
            # 买入100股
            cost = 100 * self.stock_price
            if self.balance >= cost:  # 如果资金足够
                self.balance -= cost  # 扣除资金
                self.stock_owned += 100  # 增加持有的股票
        elif action == 2:
            # 卖出100股
            if self.stock_owned >= 100:  # 如果有足够的股票
                self.balance += 100 * self.stock_price  # 增加资金
                self.stock_owned -= 100  # 减少持有的股票
        elif action != 0: # 如果动作非法
            raise ValueError("Invalid action")

        # 计算reward
        reward = self.balance + self.stock_owned * self.stock_price - pre_value 

        # 更新当前步骤
        self.current_step += 1
        done = self.current_step > len(self.stock_data) - 1  # 如果到达数据末尾，结束

        # 获取当前状态
        state = self.get_state()

        # 返回新的状态、奖励、是否结束标志
        return state, reward, done, {}

    def render(self):
        super(EasyTradingEnv, self).render()

        print(f"Stocks Owned: {self.stock_owned}")
        print(f"Stock Price: {self.stock_price}")
        print(f"Total Value: {self.balance + self.stock_owned * self.stock_price}")
