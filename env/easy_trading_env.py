# env/easy_traing_env.py
from env.trading_env import TradingEnv
import numpy as np
from gym import spaces
from collections import deque

# TradingEnv的子类，假设股市中只包含一种股票，每次买入只能全仓，卖出只能平仓
class EasyTradingEnv(TradingEnv):
    def __init__(self, stock_data, initial_balance=10000, trade_cycle=1, window_size=5):
        super(EasyTradingEnv, self).__init__(stock_data, initial_balance, trade_cycle)

        self.stock_owned = 0  # 当前持有股票数量
        self.stock_price = stock_data.iloc[0]['close']  # 当前股价
        self.window_size = window_size * trade_cycle # 记录股价趋势的时间窗口大小
        self.past_returns = deque([0] * self.window_size, maxlen=self.window_size)
        
        # 动作空间：持有（0）、全仓（1）、平仓（2）
        self.action_space = spaces.Discrete(3)

        # 观察空间：包括 (balance, stock_owned, stock_price) 和过去 window_size 天的涨跌百分比
        self.observation_space = spaces.Box(
            low=np.array([-100.0] * self.window_size),  
            high=np.array([100.0] * self.window_size),  
            dtype=np.float64
        )
    
    def update_total_value(self):
        self.total_value = self.balance + self.stock_owned * self.stock_price
        return self.total_value

    def get_state(self):
        # 获取当前状态，包含 balance、stock_owned 和 stock_price，后跟过去 window_size 天的涨跌幅
        return list(self.past_returns)

    def reset(self):
        super(EasyTradingEnv, self).reset()

        self.stock_owned = 0
        self.stock_price = self.stock_data.iloc[self.current_step]['close']
        self.past_returns = deque([0] * self.window_size, maxlen=self.window_size)

        return self.get_state()
    
    def step(self, action):        
        # 执行动作，更新balance, stock_owned
        if action == 1:
            # 全仓
            buy_num = self.balance // self.stock_price
        elif action == 2:
            # 平仓
            buy_num = - self.stock_owned
        else:
            buy_num = 0
        self.balance -= buy_num * self.stock_price
        self.stock_owned += buy_num

        # 更新股价，历史涨跌数据，资产总价值
        for _ in range(self.trade_cycle):

            # 检查是否到达数据末尾
            self.current_step += 1
            done = self.current_step >= len(self.stock_data)  # 如果到达数据末尾，结束
            if done:
                state = self.get_state()
                pre_value = self.total_value
                self.total_value = self.update_total_value()
                reward = self.total_value - pre_value
                return state, reward, done, {}

            # 更新stock_price
            stock_data_row = self.stock_data.iloc[self.current_step]
            pre_price = self.stock_price
            self.stock_price = stock_data_row['close']

            # 更新past_returns
            if self.current_step > 0:
                pct_change = (self.stock_price - pre_price) / pre_price * 100
                self.past_returns.append(pct_change)

        # 计算reward
        pre_value = self.total_value
        self.total_value = self.update_total_value()
        reward = self.total_value - pre_value

        # 获取当前状态
        state = self.get_state()

        # 返回新的状态、奖励、是否结束标志
        return state, reward, done, {}

    def render(self):
        super(EasyTradingEnv, self).render()

        print(f"Stocks Owned: {self.stock_owned}")
        print(f"Stock Price: {self.stock_price}")
        print(f"Total Value: {self.balance + self.stock_owned * self.stock_price}")
