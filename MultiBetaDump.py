import numpy as np
import pandas as pd
import importlib
import os

# beta_list = [f[:-3] for f in os.listdir('./beta') if f.startswith('beta') and f.endswith('.py')]
beta_list = ['stockplus']
# beta_list = ['hftiming']
# beta_list = ['multitiming']
# beta_list = ['stocktiming1', 'stocktiming2','stocktiming3','stocktiming4','stocktiming5','stocktiming6']

# mode:0——重写，1——更新
mode = 0

cfg = {'startdate': '20180101',
       'enddate': '20250531',
    #    'period': 1,
       }

for beta_file in beta_list:
    print(f"-----------------dumping {beta_file}--------------------")
    module_path = f"beta.{beta_file}"  # 转换为可import的路径
    module = importlib.import_module(module_path)
    if hasattr(module, beta_file):
        beta_class = getattr(module, beta_file)
        beta = beta_class(cfg)
        if beta.generate_daily.__doc__ != 'UNIMPLEMENTED':
            beta.hf_to_daily()
        beta.generate_signal()
        if mode:
            if os.path.exists(f'./dump/{beta_file}.pkl'):
                old_beta = pd.read_pickle(f'./dump/{beta_file}_r.pkl')
                old_signal = pd.read_pickle(f'./dump/{beta_file}.pkl')
            else:
                old_beta = pd.DataFrame()
                old_signal = pd.DataFrame()
            new_beta = beta.beta.copy()
            new_signal = beta.signal_df.copy()
            new_index = new_beta.index.difference(old_beta.index)
            if len(new_index) > 0:
                new_beta = new_beta.loc[new_index]
                new_signal = new_signal.loc[new_index]
                beta.beta = pd.concat([old_beta, new_beta], axis=0, join='outer')
                beta.signal_df = pd.concat([old_signal, new_signal], axis=0, join='outer')
                beta.dump()
        else:
            beta.dump()