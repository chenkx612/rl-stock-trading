from utils import Visualizer, RandomSeedManager, SingleStockDataGenerator
from agent import DQNAgent
from env import EasyTradingEnv
import pandas as pd
from matplotlib import pyplot as plt
import torch

def test_gen():
    # set random seed
    seed_manager = RandomSeedManager()

    stock_num = 1000
    data_gen = SingleStockDataGenerator(start_date='2010-01-01', end_date='2020-01-01')
    rates = []
    for _ in range(stock_num):
        data_gen.reset()
        data_gen.generate()
        rates.append(data_gen.get_average_return_rate())

    print('mean return rate: ', sum(rates) / len(rates))
    
    print(data_gen.get_average_return_rate())
    visual = Visualizer()
    visual.plot_price(data_gen.data)

    # 创建直方图
    plt.hist(rates, bins=10, edgecolor='black')  # bins控制分组的数量

    # 添加标题和标签
    plt.title('Annual Return rate Distribution')
    plt.xlabel('Return Rate')
    plt.ylabel('Frequency')

    # 显示图形
    plt.show()

def test_dqn():
    # set random seed
    seed_manager = RandomSeedManager()

    # generate data
    data_gen = SingleStockDataGenerator(start_date='2018-01-01', end_date='2020-01-01')
    data_gen.generate()

    # create env
    env = EasyTradingEnv(data_gen.data)  

    # create dqn agent
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.n  # 动作空间维度
    agent = DQNAgent(state_dim, action_dim)

    # 训练智能体
    returns = agent.train(env, num_episodes=10)
    # 保存模型
    agent.save_model("dqn_trading_model.pth")
    # 加载模型
    agent.load_model("dqn_trading_model.pth")
    # 可视化训练结果
    visual = Visualizer()
    visual.plot_average_return_rates(returns, 'DQN', 'Easy Trading Enviroment')

def main():
    # set random seed
    seed_manager = RandomSeedManager()

    # initiate visualizer
    viual = Visualizer()

    # generate data
    data_gen = SingleStockDataGenerator(start_date='2000-01-01', end_date='2024-01-01')
    data_gen.generate()
    viual.plot_price(data_gen.data)
    print('mean return rate of the stock market: ', data_gen.get_average_return_rate())

    # create env
    train_data = data_gen.get(start_date='2000-01-01', end_date='2019-12-31')
    val_data = data_gen.get(start_date='2020-01-01', end_date='2021-01-01')
    train_env = EasyTradingEnv(train_data)
    val_env = EasyTradingEnv(val_data)

    # check gpu
    if torch.cuda.is_available():
        print('using gpu')
    else:
        print('using cpu')

    # create dqn agent
    state_dim = train_env.observation_space.shape[0]  # 状态空间维度
    action_dim = train_env.action_space.n  # 动作空间维度
    agent = DQNAgent(state_dim, action_dim)

    # 训练智能体
    returns = agent.train(train_env, num_episodes=5)

    # 可视化训练结果
    visual = Visualizer()
    visual.plot_average_return_rates(returns, 'DQN', 'Easy Trading Enviroment')

    # 保存模型
    agent.save_model("dqn_model.pth")

    # validation
    return_rates = []  # 用于记录收益率随时间的变化
    done = False
    state = val_env.get_state()
    while not done:
        action = agent.choose_action(state, is_training=False)
        state, reward, done, _ = val_env.step(action)
        return_rates.append(val_env.get_return_rate())
    visual.plot_return_rates(return_rates, val_data['date'])

if __name__ == '__main__':
    # test_gen()    
    # test_agent()
    # test_dqn()
    main()
