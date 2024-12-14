import pandas as pd
import numpy as np

# 基类：负责数据的生成
class DataGenerator:
    def __init__(self, start_date, end_date, seed):
        """
        初始化 DataGenerator 类。
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'。
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'。
        :param seed: 随机数种子，用于结果的可重复性。
        """
        self.data = None # 用于存储生成的股票价格数据
        self.start_date = start_date # 用于记录数据的开始日期
        self.end_date = end_date # 用于记录数据的结束日期
        self.seed = seed 

    def generate(self):
        """
        根据提供的开始时间和结束时间生成每日股票价格数据。
        
        :return: None, 将生成的数据存储在 self.data 中。
        """
        raise NotImplementedError('generate方法必须在子类中实现')

    def save(self, filename):
        """
        将生成的数据保存到文件。
        :param filename: 保存文件的路径。
        """
        pass

    def load(self, filepath):
        """
        从文件中加载数据。
        :param filepath: 数据文件路径。
        """
        pass

    def get(self, start_date=None, end_date=None):
        """
        获取生成的数据。
        :param start_date: 起始日期（可选）。
        :param end_date: 结束日期（可选）。
        :return: 查询结果，格式为 pandas.DataFrame。
        """
        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date
        
        # 确保 'date' 列为 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # 使用条件筛选从 self.data 中抽取对应的时间段
        filtered = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date)]
        
        return filtered

    def reset(self):
        """
        重置当前数据，清空 self.data。
        """
        self.data = None


# DataGenerator子类：简单涨跌规则下生成的股票数据
class SingleStockDataGenerator(DataGenerator):
    def __init__(
        self, start_date='2022-01-01', end_date='2025-01-01', 
        initial_price=1, amplitude=0.02, trend_days=2, down_up_p=0.7, 
        up_down_p=0.6, plunge_count=5, plunge_p = 0.2, plunge_rate= 0.5, seed=None
    ):
        """
        初始化函数
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'。
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'。
        :param initial_price: 初始股价
        :param amplitude: 固定的涨跌幅度
        :param trend_days: 确定的趋势延续天数
        :param seed: 随机数种子，用于结果的可重复性
        :param down_up_p: 连跌trend_days后转涨的概率
        :param up_down_p: 连涨trend_days后转跌的概率
        :param plunge_count: 可能触发暴跌的连跌天数
        :param plunge_p: 连跌plunge_count后的暴跌可能
        :param plunge_rate: 暴跌比例
        """
        super().__init__(start_date, end_date, seed)
        self.initial_price = initial_price
        self.amplitude = amplitude
        self.trend_days = trend_days
        self.down_up_p = down_up_p
        self.up_down_p = up_down_p
        self.plunge_count = plunge_count
        self.plunge_rate = plunge_rate
        self.plunge_p = plunge_p

    def generate(self):
        """
        根据特定规则生成单支股票的每日价格数据。
        """
        # 初始化参数
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        prices = [self.initial_price]
        consecutive_days = 0  # 当前涨或跌的连续天数
        current_trend = 1 if np.random.rand() < 0.5 else -1  # 初始状态，随机决定涨或跌（1: 涨，-1: 跌）

        # 生成后续价格
        for _ in range(1, len(dates)):
            # 判断是否暴跌
            if consecutive_days >= self.plunge_count and current_trend == -1:
                if np.random.rand() < self.plunge_p:
                    new_price = prices[-1] * (1 - self.plunge_rate)
                    prices.append(new_price)
                    consecutive_days = 0
                    current_trend = 1 if np.random.rand() < 0.5 else -1 
                    continue
            
            if consecutive_days < self.trend_days:  # 趋势延续
                consecutive_days += 1
            else:  # 趋势可能反转
                if current_trend == 1:  # 连涨几天后
                    change_trend = np.random.rand() < self.up_down_p
                elif current_trend == -1:  # 连跌几天后
                    change_trend = np.random.rand() < self.down_up_p
                if not change_trend: # 趋势没变
                    consecutive_days += 1
                else: # 趋势改变，重新计数连续天数
                    current_trend = -current_trend
                    consecutive_days = 1  

            # 更新价格
            if current_trend == 1:  # 涨
                new_price = prices[-1] * (1 + self.amplitude)
            else:  # 跌
                new_price = prices[-1] * (1 - self.amplitude)
            prices.append(new_price)

        # 构建 DataFrame
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
