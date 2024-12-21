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
        
    def plot_multi_price(
        self, data: pd.DataFrame, title: str = "Stock Price Comparison", 
        xlabel: str = "Date", ylabel: str = "Price"
    ):
        """
        绘制多只股票的价格趋势对比图

        参数:
        - data (pd.DataFrame): 包含日期和股票价格的DataFrame, 必须包含date列和至少两列股票价格数据
        - title (str): 图表的标题
        - xlabel (str): X轴的标签, 默认为"Date"
        - ylabel (str): Y轴的标签, 默认为"Price"
        """
        if 'date' not in data.columns:
            raise ValueError("DataFrame must contain a 'date' column")
        
        # 绘制每个股票的价格曲线
        for column in data.columns:
            if column != 'date':
                plt.plot(data['date'], data[column], label=column)
        
        # 设置标题和标签
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # 添加图例
        plt.legend()
        
        # 显示图像
        plt.xticks(rotation=45)  # 日期标签旋转，避免重叠
        plt.tight_layout()  # 调整布局
        plt.show()

    def plot_average_return_rates(self, rates_list, agent_name, env_name):
        '''绘制agent训练过程中年均回报率随episodes的图像'''
        episodes_list = list(range(len(rates_list)))
        plt.plot(episodes_list, np.array(rates_list) * 100)
        plt.xlabel('Episodes')
        plt.ylabel('Average annual return rate(%)')
        plt.title('{} on {}'.format(agent_name, env_name))
        plt.show()

    def plot_return_rates(self, return_rates, start_date, end_date):
        '''绘制收益率(%)随时间的图像'''
        dates = pd.date_range(start=start_date, end=end_date, periods=len(return_rates))
        plt.plot(dates, np.array(return_rates) * 100, color="blue")
        plt.xlabel("dates")
        plt.ylabel("revenue rate (%)")
        plt.grid(True)
        plt.show()
