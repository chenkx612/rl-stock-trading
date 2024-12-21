# utils/data_generator.py
import pandas as pd
import numpy as np
import os

# 基类：负责数据的生成
class DataGenerator:
    def __init__(self, start_date, end_date):
        """
        初始化 DataGenerator 类。
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'。
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'。
        :param seed: 随机数种子，用于结果的可重复性。
        """
        self.data = None # 用于存储生成的股票价格数据
        self.start_date = start_date # 用于记录数据的开始日期
        self.end_date = end_date # 用于记录数据的结束日期
        self.data_dir = "data"  # 存储路径
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def generate(self):
        """
        根据提供的开始时间和结束时间生成每日股票价格数据。
        
        :return: None, 将生成的数据存储在 self.data 中。
        """
        raise NotImplementedError('generate方法必须在子类中实现')

    def save(self, filename, file_format="csv"):
        """
        保存股价数据到文件
        :param filename: 文件路径
        :param file_format: 文件格式("csv" 或 "pickle")
        """
        if self.data.empty:
            raise ValueError("数据还未生成，请先生成数据")

        # 定义模型保存路径
        data_path = os.path.join(self.data_dir, filename) 
        if file_format == "csv":
            # 保存为 CSV 文件
            self.data.to_csv(data_path, index=False)
            print(f"Data saved to {data_path} in CSV format.")
        elif file_format == "pickle":
            # 保存为 Pickle 文件
            self.data.to_pickle(data_path)
            print(f"Data saved to {data_path} in Pickle format.")
        else:
            print("Unsupported file format. Please use 'csv' or 'pickle'.")

    def load(self, filename, file_format="csv"):
        """
        从文件加载股价数据
        :param filename: 文件路径
        :param file_format: 文件格式("csv" 或 "pickle")
        """
        # 定义模型保存路径
        data_path = os.path.join(self.data_dir, filename) 
        if file_format == "csv":
            # 从 CSV 文件加载
            self.data = pd.read_csv(data_path)
            print(f"Data loaded from {data_path} in CSV format.")
            self._update_start_end_days()
        elif file_format == "pickle":
            # 从 Pickle 文件加载
            self.data = pd.read_pickle(data_path)
            print(f"Data loaded from {data_path} in Pickle format.")
            self._update_start_end_days()
        else:
            print("Unsupported file format. Please use 'csv' or 'pickle'.")

    def _update_start_end_days(self):
        if self.data.empty:
            raise ValueError("数据还未生成，请先生成数据")
        self.start_date = self.data.iloc[0]['date']
        self.end_date = self.data.iloc[-1]['date']

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


# DataGenerator子类：生成单支股票的历史收盘价数据
class SingleStockDataGenerator(DataGenerator):
    def __init__(
        self, start_date='2000-01-01', end_date='2024-01-01', 
        initial_price=1, amplitude=0.02, trend_days=2, down_up_p=0.78, 
        up_down_p=0.75, burst_count=5, burst_p=0.1, burst_rate=0.1
    ):
        """
        初始化函数
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'。
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'。
        :param initial_price: 初始股价
        :param amplitude: 涨跌幅期望
        :param trend_days: 趋势延续天数期望
        :param down_up_p: 连跌后转涨的概率
        :param up_down_p: 连涨后转跌的概率
        :param burst_count: 可能触发暴跌(涨)的连跌(涨)天数
        :param burst_p: 连跌(涨)burst_count后的暴跌(涨)可能
        :param burst_rate: 暴跌(涨)比例
        """
        super().__init__(start_date, end_date)
        self.initial_price = initial_price
        self.amplitude = amplitude
        self.trend_days = trend_days
        self.down_up_p = down_up_p
        self.up_down_p = up_down_p
        self.burst_count = burst_count
        self.burst_rate = burst_rate
        self.burst_p = burst_p

    def generate(self):
        # 初始化参数
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        prices = [self.initial_price]
        consecutive_days = 0  # 当前涨或跌的连续天数
        current_trend = 1 if np.random.rand() < 0.5 else -1  # 初始状态，随机决定涨或跌（1: 涨，-1: 跌）

        # 生成后续价格
        for _ in range(1, len(dates)):
            # 判断是否暴跌或暴涨
            if consecutive_days >= self.burst_count and np.random.rand() < self.burst_p:
                new_price = prices[-1] * (1 + current_trend * self.burst_rate)
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
    
    def get_average_return_rate(self):
        '''返回生成的单只股票的年均回报率'''
        years = (self.data.iloc[-1]['date'] - self.data.iloc[0]['date']).days / 365.25
        current_price = self.data.iloc[-1]['close']
        return (current_price / self.initial_price) ** (1/years) - 1


# DataGenerator子类：生成多支股票的历史收盘价数据
class MultiStockDataGenerator(DataGenerator):
    def __init__(
        self, stock_symbols, initial_price, start_date='2000-01-01', 
        end_date='2024-01-01', amplitude=0.02, trend_days=2, down_up_p=0.78, 
        up_down_p=0.75, burst_count=5, burst_p=0.1, burst_rate=0.1 
    ): 
        """
        初始化函数
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'。
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'。
        :param initial_price: 初始股价
        :param amplitude: 涨跌幅期望
        :param trend_days: 趋势延续天数期望
        :param down_up_p: 连跌后转涨的概率
        :param up_down_p: 连涨后转跌的概率
        :param burst_count: 可能触发暴跌(涨)的连跌(涨)天数
        :param burst_p: 连跌(涨)burst_count后的暴跌(涨)可能
        :param burst_rate: 暴跌(涨)比例
        """
        super().__init__(start_date, end_date)
        self.stock_symbols = stock_symbols
        self.initial_price = initial_price
        self.amplitude = amplitude
        self.trend_days = trend_days
        self.down_up_p = down_up_p
        self.up_down_p = up_down_p
        self.burst_count = burst_count
        self.burst_rate = burst_rate
        self.burst_p = burst_p

    def generate(self):
        dates = pd.date_range(start=self.start_date, end=self.end_date)  # 日期列
        price_lists = []  # 用于存储n支股票的历史收盘价数据

        # 逐一生成
        for i in range(len(self.stock_symbols)):
            prices = [self.initial_price[i]]
            consecutive_days = 0  # 当前涨或跌的连续天数
            current_trend = 1 if np.random.rand() < 0.5 else -1  # 初始状态，随机决定涨或跌（1: 涨，-1: 跌）

            # 生成后续价格
            for _ in range(1, len(dates)):

                # 判断是否暴跌或暴涨
                if consecutive_days >= self.burst_count and np.random.rand() < self.burst_p:
                    new_price = prices[-1] * (1 + current_trend * self.burst_rate)
                    prices.append(new_price)
                    consecutive_days = 0
                    current_trend = 1 if np.random.rand() < 0.5 else -1 
                    continue
                
                # 更新current_trend
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
            
            # 将生成的价格加入price_lists
            price_lists.append(prices)

        # 构建 DataFrame
        self.data = pd.DataFrame({
            'date': dates,
        })
        for i in range(len(self.stock_symbols)):
            self.data[self.stock_symbols[i]] = price_lists[i]
    
    def get_average_return_rate(self):
        '''以字典形式返回生成的多只股票的年均回报率'''
        years = (self.data.iloc[-1]['date'] - self.data.iloc[0]['date']).days / 365.25
        rates = {}
        for i in range(len(self.stock_symbols)):
            current_price = self.data.iloc[-1][self.stock_symbols[i]]
            rates[self.stock_symbols[i]] = ((current_price / self.initial_price[i]) ** (1/years) - 1)
        return rates
