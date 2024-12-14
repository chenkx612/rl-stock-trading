import data
from visualization import Visualizer
from strategy import BaselineAgent
from env import EasyTradingEnv
import pandas as pd
from matplotlib import pyplot as plt

def test_agent():
    # 生成股票数据
    data_gen = data.SingleStockDataGenerator()
    data_gen.generate()
    stock_data = data_gen.get('2022-12-03', '2024-12-13')

    # 创建交易环境
    env = EasyTradingEnv(stock_data=stock_data, initial_balance=10000)

    # 创建基线智能体
    agent = BaselineAgent(action_space=env.action_space, state_space=env.observation_space)

    # 与环境交互
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        print('current action: ', action)
        state, reward, done, info = env.step(action)
        env.render()

def test_single_stock():
    data_gen = data.SingleStockDataGenerator()
    data_gen.generate()
    visual = Visualizer()
    visual.plot_price(data_gen.data)

def test_multi_stock():
    stock_num = 100
    data_gen = data.SingleStockDataGenerator()
    last_prices = []
    for _ in range(100):
        data_gen.reset()
        data_gen.generate()
        last_prices.append(data_gen.data.iloc[-1]['close'])
    
    print('min price: ', min(last_prices))
    print('max price: ', max(last_prices))
    
    # 创建直方图
    plt.hist(last_prices, bins=10, edgecolor='black')  # bins控制分组的数量

    # 添加标题和标签
    plt.title('Stock Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')

    # 显示图形
    plt.show()

if __name__ == '__main__':
    # test_gen()    
    # test_agent()
    test_single_stock()
