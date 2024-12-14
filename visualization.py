import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg') # 切换到TkAgg后端

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
