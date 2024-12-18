# utils/visualizer.py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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

    def plot_average_return_rates(self, rates_list, agent_name, env_name):
        '''绘制agent训练过程中年均回报率随episodes的图像'''
        episodes_list = list(range(len(rates_list)))
        plt.plot(episodes_list, np.array(rates_list) * 100)
        plt.xlabel('Episodes')
        plt.ylabel('Average annual return rate(%)')
        plt.title('{} on {}'.format(agent_name, env_name))
        plt.show()

    def plot_return_rates(self, return_rates, dates):
        '''绘制收益率(%)随时间的图像'''
        plt.figure(figsize=(12, 6))
        plt.plot(dates, return_rates, color="blue")
        plt.xlabel("dates")
        plt.ylabel("revenue rate (%)")
        plt.grid(True)
        plt.show()
