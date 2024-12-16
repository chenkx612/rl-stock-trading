# env/trading_env.py
import gym

# 交易环境基类
class TradingEnv(gym.Env):
    def __init__(self, stock_data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        # 设置初始状态
        self.stock_data = stock_data  # 股票数据
        self.initial_balance = initial_balance  # 初始账户余额
        self.current_step = 0  # 当前时间步
        self.balance = self.initial_balance  # 当前余额
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        
    def step(self, action):
        raise NotImplementedError("step方法必须在子类中实现")
    
    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
 