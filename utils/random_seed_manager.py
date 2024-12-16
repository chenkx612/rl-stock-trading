# utils/random_seed_manageer.py
import random
import numpy as np
import torch

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
