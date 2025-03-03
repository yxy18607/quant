import numpy as np
import pandas as pd
from BetaModules import DailyCTA

cfg = {'startdate': '20180101',
       'enddate': '20250131',
       'signal_id': 'timing2',
       'trade_price': 'open',
       'slippage': 0.4,
       'fee': 0.000023,
       'instruments': ['zz1000'],
       'mode': 0
       }

backtest = DailyCTA(cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve()