import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import torch

class Visualizer:
    def __init__(self):
        """
        初始化 Visualizer 类。
        """

    def plot_price(self, data, title="Stock Price", save_path=None):
        """
        绘制股票价格数据。
        :param data: 股票价格数据, pandas.DataFrame 格式，需包含 'date' 和 'close' 列。
        :param title: 图表标题。
        :param save_path: 如果提供保存路径，则将图表保存为文件。
        """
        # 检查data是否符合格式要求
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if not {'date', 'close'}.issubset(data.columns):
            raise ValueError("Data must contain 'date' and 'close' columns.")

        # 确保日期列为 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])

        # 绘制价格曲线
        plt.figure(figsize=(12, 6))
        plt.plot(data['date'], data['close'], label="Price", color="blue")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()

        # 显示或保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_returns(self, return_list, agent_name, env_name):
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('{} on {}'.format(agent_name, env_name))
        plt.show()


class RandomSeedManager:
    """
    管理随机数种子的类，用于确保实验的可重现性。
    """
    def __init__(self, seed=42):
        """
        初始化 RandomSeedManager, 并设定随机数种子，默认使用 seed=42。
        :param seed: 随机数种子，默认值为 42
        """
        self.seed = seed
        self.set_seed()

    def set_seed(self):
        """
        设置随机种子以确保结果可重现。
        该方法会设置 Python、NumPy 和 PyTorch 的随机种子。
        """
        # 设置 Python 随机数种子
        random.seed(self.seed)

        # 设置 NumPy 随机数种子
        np.random.seed(self.seed)

        # 设置 PyTorch 随机数种子
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # 设置所有GPU设备的种子

        # 对于深度学习模型，确保每次运行时相同的随机初始化
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_seed(self):
        """
        获取当前的种子。
        :return: 当前的种子
        """
        return self.seed
