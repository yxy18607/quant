import numpy as np
import pandas as pd
from BetaModules import DailyCTA

cfg = {'startdate': '20230601',
       'enddate': '20250531',
       # 'signal_id': 'timing5',
       'signal_id': 'hktiming',
       'zone': 'hk',
       'trade_price': 'open',
       'slippage': 1,
       'fee': 0.000023,
       # 'instruments': ['zz1000'],
       'instruments': ['hsi'],
       'mode': 0
       }

backtest = DailyCTA(cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve(os_startdate='20230601')
# backtest.plot_curve()
# backtest.save_pnl()