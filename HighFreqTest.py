import numpy as np
import pandas as pd

signal_id = 'hfetf'
exec(f'from beta.{signal_id} import {signal_id}')

# instruments = pd.read_csv('./derivative_data/activeetf.csv')['代码'].tolist()
instruments = pd.read_csv('./derivative_data/etflist.csv')['f_info_windcode'].tolist()

cfg = {'startdate': '20180101',
       'enddate': '20230630',
       'signal_id': signal_id,
       'instruments': instruments,
       'fee': 0.00005,
       'load_unit': '60m',
       'dump_factor': True,
       'dump_pnl': True,
       # 'include_overnight': False
       }

backtest = eval(f'{signal_id}(cfg)')
backtest.backward()
backtest.get_pnl()
backtest.stats()
backtest.plot_curve()
# backtest.plot_curve(os_startdate='20230701')