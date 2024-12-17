# agent/baseline_agent.py
from agent import Agent

# Agent子类：对于EasyTradingEnv，执行简单的策略，买入并一直持有
class BaselineAgent(Agent):
    def choose_action(self, state):
        """
        根据基线策略选择动作：
        - 用全部资金买入股票。
        - 之后一直保持持有状态。

        :param state: 当前状态 [balance, stock_owned, stock_price]
        :return: 动作(0: 持有, 1: 买入, 2: 卖出)
        """
        return 1  # 剩余资金不够买入一手时，EasyTradingEnv会选择持有
