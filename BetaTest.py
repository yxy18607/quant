import numpy as np
import pandas as pd
from BetaModules import DailyCTA

cfg = {'startdate': '20230101',
       'enddate': '20250331',
       'signal_id': 'timing5',
       # 'zone': 'hk',
       'trade_price': 'open',
       'slippage': 1,
       'fee': 0.000023,
       'instruments': ['zz1000'],
       'mode': 0
       }

backtest = DailyCTA(cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve(os_startdate='20230101')
# backtest.plot_curve()