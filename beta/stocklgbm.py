import numpy as np
import pandas as pd
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
sys.path.append('..')
from MultiBetaModules import Beta
import FastRolling as fr

class stocklgbm(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.feature_list = open(f'./featurelist/features_{self.__class__.__name__}').read().rstrip().split('\n')
        self.num = len(self.feature_list)
        for i in range(self.num):
            exec(f'self.feature{i} = pd.read_pickle("./dump/{self.feature_list[i]}_r.pkl").loc[self.dateindex]')
        self.price = pd.read_pickle('./data/adjopen.pkl').loc[self.dateindex]
    
        self.flag_loaded = False
        self.flag_trained = False
        self.random_seed = 3407

        self.backward = 200 if cfg.get('backward') is None else cfg.get('backward')
        self.period = 1 if cfg.get('period') is None else cfg.get('period')

        self.set_label()
        # self.moving_alpha = []
        # self.temp_alpha = []
        # self.moving_label = []

    def get_offset_trd(self, date, offset):
        return self.calendar[np.maximum(self.calendar.searchsorted(date)+offset, 0)]

    def is_reset(self, didx):
        return (didx[4:6] in ['01', '07'])&(self.get_offset_trd(didx, -1)[4:6] in ['12', '06'])
    
    def first_day(self, didx):
        did = didx
        while not self.is_reset(did):
            did = self.get_offset_trd(did, -1)
        return self.get_offset_trd(did, -self.period-1)

    def set_label(self):
        label = self.price.pct_change(self.period, fill_method=None).shift(-self.period-1)
        self.label = label

    def get_label(self, didx_start, didx_end):
        label = self.label.loc[didx_start:didx_end]
        label = label[label.notna()].stack()
        return label

    def get_feature(self, didx_start, didx_end):
        features = []
        for i in range(self.num):
            ft = eval(f'self.feature{i}.loc[didx_start:didx_end]')
            ft.index.name = 'trade_date'
            exec(f'features.append(ft[ft.notna()].stack())')
        features = pd.concat(features, axis=1, join='inner')
        # features = features.sub(features.mean(0), axis=1).div(features.std(0), axis=1)
        return features
        
    def generate_signal(self):
        df_pred = []
        print("--------------------based on raw feature--------------------")
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
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
                    train_dataset = lgb.Dataset(X_train, y_train)
                    val_dataset = lgb.Dataset(X_val, y_val)
                    params = {'objective': 'regression',
                              'metric': 'huber',
                              'max_depth': 5,
                              'num_leaves': 31,
                              'learning_rate': 0.01,
                            #   'min_data_in_leaf': 100,
                              'num_threads': 8,
                              'seed': self.random_seed,
                              'verbose': -1
                              }
                    callbacks = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=500)]
                    bst = lgb.train(params=params,
                                    train_set=train_dataset,
                                    num_boost_round=1000,
                                    valid_sets=[train_dataset, val_dataset],
                                    valid_names=['train', 'valid'],
                                    callbacks=callbacks
                                    )
                    self.flag_trained = True
                    self.flag_loaded = False
                    bst.save_model(f'./models/model_{self.__class__.__name__}.txt')

            if not self.flag_loaded and self.flag_trained:
                model = lgb.Booster(model_file=f'./models/model_{self.__class__.__name__}.txt')
                self.flag_loaded = True
            
            if self.flag_loaded:
                pbar.set_description(f'predicting {didx}')
                test_data = self.get_feature(didx, didx)
                X = test_data.to_numpy()
                y_pred = model.predict(X, num_iteration=model.best_iteration)
                df_pred.append(pd.Series(y_pred, index=test_data.index, name='pred'))

        df_pred = pd.concat(df_pred, axis=0).reset_index().pivot(index='trade_date', columns='code', values='pred') 
        df_pred = df_pred.reindex(columns=self.price.columns, index=self.dateindex)
        self.beta = df_pred
        self.signal_df = fr.signal_minmax_scaler(df_pred, 20)