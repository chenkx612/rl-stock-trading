# 生成不同股票的历史数据
from utils import RandomSeedManager, Visualizer
from utils import MultiStockDataGenerator
import numpy as np

# 设定随机数种子
rsm = RandomSeedManager(seed=42)

# 可视化类实例化
visual = Visualizer()

# 生成不同股票的历史数据
symbols = ['SH', 'SZ', 'BJ']  # 股票代码
init_price = [np.random.rand() + 0.5 for _ in range(len(symbols))]  # 期初股价
multi_generator = MultiStockDataGenerator(
    stock_symbols=symbols, initial_price=init_price, start_date='2014-01-01', end_date='2024-01-01',
    trend_days=1, amplitude=0.01, burst_rate=0.5
)  # 多支股票数据生成器实例化
multi_generator.generate()  # 生成数据
prices = multi_generator.get(start_date='2019-01-01', end_date='2022-01-01')  # 截取数据
visual.plot_multi_price(prices)  # 可视化
