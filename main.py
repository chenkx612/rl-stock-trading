from utils import Visualizer, RandomSeedManager, SingleStockDataGenerator
from agent import DQNAgent
from env import EasyTradingEnv
import torch

def main():
    # set random seed
    seed_manager = RandomSeedManager(seed=41)

    # initiate visualizer
    viual = Visualizer()

    # generate data
    data_gen = SingleStockDataGenerator(start_date='2000-01-01', end_date='2024-01-01')
    data_gen.generate()
    viual.plot_price(data_gen.data)
    print('annual return rate of this stock: ', data_gen.get_average_return_rate())

    # create env
    train_start_date = '2000-01-01'
    train_end_date = '2019-12-31'
    val_start_date = '2020-01-01'
    val_end_date = '2024-01-01'
    train_data = data_gen.get(start_date=train_start_date, end_date=train_end_date)
    val_data = data_gen.get(start_date=val_start_date, end_date=val_end_date)
    train_env = EasyTradingEnv(train_data, trade_cycle=5)
    val_env = EasyTradingEnv(val_data, trade_cycle=5)

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
    visual.plot_return_rates(return_rates, val_start_date, val_end_date)
    print('annual return rate of validation: ', val_env.get_annual_return_rate())

if __name__ == '__main__':
    main()
