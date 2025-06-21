import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import Ridge
import joblib
from tqdm import tqdm
sys.path.append('..')
from MultiBetaModules import Beta

class stockridge(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.feature_list = open(f'./featurelist/features_{self.__class__.__name__}').read().rstrip().split('\n')
        self.num = len(self.feature_list)
        for i in range(self.num):
            exec(f'self.feature{i} = pd.read_pickle("./dump/{self.feature_list[i]}.pkl").loc[self.dateindex]')
        self.price = pd.read_pickle('./data/adjopen.pkl').loc[self.dateindex]
    
        self.flag_loaded = False
        self.flag_trained = False

        self.backward = 100 if cfg.get('backward') is None else cfg.get('backward')
        self.period = 1 if cfg.get('period') is None else cfg.get('period')

        self.set_label()
        # self.moving_alpha = []
        # self.temp_alpha = []
        # self.moving_label = []

    def get_offset_trd(self, date, offset):
        return self.calendar[np.maximum(self.calendar.searchsorted(date)+offset, 0)]

    def is_reset(self, didx):
        return (didx[4:6] in ['01', '04', '07', '10'])&(self.get_offset_trd(didx, -1)[4:6] in ['12', '03', '06', '09'])
    
    def first_day(self, didx):
        did = didx
        while not self.is_reset(did):
            did = self.get_offset_trd(did, -1)
        return self.get_offset_trd(did, -self.period-1)

    def set_label(self):
        label = self.price.pct_change(self.period, fill_method=None).shift(-self.period-1)
        self.label = label.rolling(5).rank(pct=True)

    def get_label(self, didx_start, didx_end):
        label = self.label.loc[didx_start:didx_end]
        return label[label.notna()].stack()

    def get_feature(self, didx_start, didx_end):
        features = []
        for i in range(self.num):
            ft = eval(f'self.feature{i}.loc[didx_start:didx_end]')
            ft.index.name = 'trade_date'
            exec(f'features.append(ft[ft.notna()].stack())')
        features = pd.concat(features, axis=1, join='inner')
        return features
        
    def generate_signal(self):
        df_pred = []
        pbar = tqdm(self.dateindex)
        for didx in pbar:
            if self.is_reset(didx):
                # if today is retrain day, restart training
                self.flag_trained = False
                didx_end = self.first_day(didx) # find the training end date of this period

                if self.dateindex.searchsorted(didx_end)-self.backward+1>=0 and not self.flag_trained:
                    didx_start = self.get_offset_trd(didx_end, -self.backward+1) # training start date
                    pbar.set_description(f'training from {didx_start} to {didx_end}')
                    train_data = self.get_feature(didx_start, didx_end)
                    train_label = self.get_label(didx_start, didx_end)
                    dataset = pd.concat([train_data, train_label], axis=1, join='inner')
                    dataset = dataset.to_numpy()
                    X = dataset[:, :-1]
                    y = dataset[:, -1]
                    model = Ridge(alpha=0.1)
                    model.fit(X, y)
                    self.flag_trained = True
                    self.flag_loaded = False
                    joblib.dump(model, f'./models/model_{self.__class__.__name__}.pkl')

            if not self.flag_loaded and self.flag_trained:
                model = joblib.load(f'./models/model_{self.__class__.__name__}.pkl')
                self.flag_loaded = True
            
            if self.flag_loaded:
                pbar.set_description(f'predicting {didx}')
                test_data = self.get_feature(didx, didx)
                X = test_data.to_numpy()
                y_pred = model.predict(X)
                df_pred.append(pd.Series(y_pred, index=test_data.index, name='pred'))

        df_pred = pd.concat(df_pred, axis=0).reset_index().pivot(index='trade_date', columns='code', values='pred') 
        df_pred = df_pred.reindex(columns=self.price.columns, index=self.dateindex)
        self.signal_df = df_pred.rolling(20).rank(pct=True)*2-1

        df_pred.to_pickle(f'./models/pred_{self.__class__.__name__}.pkl')

