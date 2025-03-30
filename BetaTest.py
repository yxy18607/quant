import numpy as np
import pandas as pd
from BetaModules import DailyCTA

cfg = {'startdate': '20200101',
       'enddate': '20250228',
       'signal_id': 'timing1',
       'trade_price': 'open',
       'slippage': 0,
       'fee': 0.00001,
       'instruments': ['300779'],
       'mode': 1
       }

backtest = DailyCTA(cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve()