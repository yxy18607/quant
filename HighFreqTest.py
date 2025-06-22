import numpy as np
import pandas as pd

signal_id = 'hfetf'
exec(f'from beta.{signal_id} import {signal_id}')

cfg = {'startdate': '20180101',
       'enddate': '20230530',
       'signal_id': signal_id,
       'fee': 0,
       'load_unit': '60m',
       'load_num': 20
       }

backtest = eval(f'{signal_id}(cfg)')
backtest.backward()
backtest.get_pnl()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve()
# backtest.plot_curve(os_startdate='20230601')
# backtest.save_pnl()