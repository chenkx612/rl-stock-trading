# env/trading_env.py
import gym

# 交易环境基类
class TradingEnv(gym.Env):
    def __init__(self, stock_data, initial_balance=10000, trade_cycle=1):
        super(TradingEnv, self).__init__()

        # 设置初始状态
        self.stock_data = stock_data  # 股票数据
        self.initial_balance = initial_balance  # 初始账户余额
        self.trade_cycle = trade_cycle  # 调仓周期
        self.current_step = 0  # 当前时间步
        self.balance = self.initial_balance  # 当前余额
        self.total_value = self.balance # 资产总价值
        self.start_date = self.stock_data.iloc[0]['date']
        self.end_date = self.stock_data.iloc[-1]['date']

    def update_total_value(self):
        '''更新total_value, 也即当前所有资产总价值'''
        raise NotImplementedError("该方法必须在子类中实现")
    
    def get_state(self):
        raise NotImplementedError("get_state方法必须在子类中实现")
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.total_value = self.balance
        
    def step(self, action):
        '''返回 next_state, reward, down, info'''
        raise NotImplementedError("step方法必须在子类中实现")
    
    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
 
    def get_return_rate(self):
        '''获取当前收益率'''
        self.update_total_value()
        return (self.total_value - self.initial_balance) / self.initial_balance

    def get_annual_return_rate(self):
        '''获取当前年均收益率'''
        current_date = self.stock_data.iloc[self.current_step-1]['date']
        years = (current_date - self.start_date).days / 365.25
        return (self.total_value / self.initial_balance) ** (1/years) - 1
