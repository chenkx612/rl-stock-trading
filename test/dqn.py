from utils import SingleStockDataGenerator, Visualizer
from env import EasyTradingEnv
from agent import DQNAgent

# generate data
data_gen = SingleStockDataGenerator(start_date='2018-01-01', end_date='2020-01-01')
data_gen.generate()

# create env
env = EasyTradingEnv(data_gen.data)  

# create agent
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = env.action_space.n  # 动作空间维度
agent = DQNAgent(state_dim, action_dim)

# train agent
returns = agent.train(env, num_episodes=10)
agent.save_model("dqn_trading_model.pth")
agent.load_model("dqn_trading_model.pth")
visual = Visualizer()
visual.plot_average_return_rates(returns, 'DQN', 'Easy Trading Enviroment')
