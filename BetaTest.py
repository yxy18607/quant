import numpy as np
import pandas as pd
from BetaModules import DailyCTA

cfg = {'startdate': '20200801',
       'enddate': '20250531',
       'signal_id': 'timing5',
       'zone': 'hk',
       'trade_price': 'open',
       'slippage': 1,
       'fee': 0.0003,
       # 'instruments': ['zz1000'],
       'instruments': {'zz1000': 'hstech'},
       'mode': 0,
       # 'period': 'M'
       }

# pd.set_option('display.max_rows', None)
backtest = DailyCTA(cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
backtest.plot_curve(os_startdate='20230601')
# backtest.plot_curve()
# backtest.save_pnl()