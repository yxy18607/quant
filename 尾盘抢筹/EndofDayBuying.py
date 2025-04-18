# 第一部分，python标准库
import json
import time
import datetime
import logging

# 第二部分，第三方库
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB

# 第三部分，自定义类库
from quant.api import RealProcess, AccountConfig, Strategy, BackProcess, TradingDate
from gqdata.data import DataAPI

api_obj = DataAPI("token")

# 继承基础策略类
class EndofDayBuying(Strategy):
    # 第一部分：策略参数
    # id = 0  # 策略ID
    # start = '20250105'  # 回测起始时间
    # end = '20210115'  # 回测结束时间
    # freq = 'd'
    # refresh_rate = (1, ['14:45'])
    # accounts = {'stock_account': AccountConfig('security', capital_base=1e6)}  # 模拟盘账户信息

    # 第二部分：初始化策略，回测期间只运行一次，可以用于设置全局变量
    # ctx是回测期间的策略上下文，存储上述全局变量参数信息，并在整个策略执行期间更新并维护可用现金、证券的头寸、每日交易指令明细、历史行情数据等。
    def initialize(self, context):
        self.logger.info('DemoStrategy->initialize')

    def get_today(self, last_fc, df_dict, current_date='20241220'):
        # 计算当天14：45的因子值
        # data = api_obj.current_kline()
        data = api_obj.wd_ashareeodprices_ori(trade_date=current_date, fields=['s_info_windcode', 's_dq_close', 's_dq_amount', 's_dq_limit']).set_index('s_info_windcode')
        data.columns = ['new_price', 'amount', 'limit']
        # adjfactor = api_obj.gj_t_ashare_cal_adjfactor(fields=['s_info_windcode', 's_dq_adjfactor']).set_index('s_info_windcode')
        adjfactor = api_obj.wd_ashareeodprices_ori(trade_date=current_date, fields=['s_info_windcode', 's_dq_adjfactor']).set_index('s_info_windcode')
        last_fc = last_fc.set_index('code') # 上一天因子值
        df_c = pd.concat([last_fc, data], axis=1, join='inner')
        df_c['new_adjprice'] = df_c['new_price'] * adjfactor['s_dq_adjfactor']
        # s_dq_ama120_rate_keep_trend
        df_c['amt120_T-1'] = df_dict['amount'].iloc[1:].mean()
        df_c['amt120_T-2'] = df_dict['amount'].iloc[:-1].mean()
        df_c['amt119_T-1'] = df_dict['amount'].iloc[2:].sum()
        df_c['amt120_T'] = (df_c['amt119_T-1'] + df_c['amount'])/120
        df_c['ddamt120_sign'] = np.sign(df_c['amt120_T'] / df_c['amt120_T-1'] - df_c['amt120_T-1'] / df_c['amt120_T-2'])
        df_c['new_s_dq_ama120_rate_keep_trend'] = np.where(np.sign(df_c['ddamt120_sign']*df_c['s_dq_ama120_rate_keep_trend'])==1,
                                                           df_c['s_dq_ama120_rate_keep_trend'] + np.sign(df_c['s_dq_ama120_rate_keep_trend']),
                                                           df_c['ddamt120_sign'])
        # s_dq_ma10_rate_keep_trend
        df_c['ma10_T-1'] = df_dict['adjclose'].iloc[-10:].mean()
        df_c['ma10_T-2'] = df_dict['adjclose'].iloc[-11:-1].mean()
        df_c['ma9_T-1'] = df_dict['adjclose'].iloc[-9:].sum()
        df_c['ma10_T'] = (df_c['ma9_T-1'] + df_c['new_adjprice'])/10
        df_c['ddma10_sign'] = np.sign(df_c['ma10_T'] / df_c['ma10_T-1'] - df_c['ma10_T-1'] / df_c['ma10_T-2'])
        df_c['new_s_dq_ma10_rate_keep_trend'] = np.where(np.sign(df_c['ddma10_sign']*df_c['s_dq_ma10_rate_keep_trend'])==1,
                                                         df_c['s_dq_ma10_rate_keep_trend'] + np.sign(df_c['s_dq_ma10_rate_keep_trend']),
                                                         df_c['ddma10_sign'])
        # s_dq_returns_l5d_keep_trend
        df_c['ret5_T'] = df_c['new_adjprice'] / df_dict['adjclose'].iloc[-5]
        df_c['ret5_T-1'] = df_dict['adjclose'].iloc[-1] / df_dict['adjclose'].iloc[-6]
        df_c['ret5_sign'] = np.sign(df_c['ret5_T'] - df_c['ret5_T-1'])
        df_c['new_s_dq_returns_l5d_keep_trend'] = np.where(np.sign(df_c['ret5_sign']*df_c['s_dq_returns_l5d_keep_trend'])==1,
                                                           df_c['s_dq_returns_l5d_keep_trend'] + np.sign(df_c['s_dq_returns_l5d_keep_trend']),
                                                           df_c['ret5_sign'])
        # s_dq_close_ma10
        df_c['new_s_dq_close_ma10'] = df_c['new_adjprice'] / df_c['ma10_T'] - 1
        # s_dq_amount_ama120
        df_c['new_s_dq_amount_ama120'] = df_c['amount'] / df_c['amt120_T'] - 1
        # s_dq_limit_up_counts_l10d
        df_c['new_s_dq_limit_up_counts_l10d'] = (df_c['new_price'] == df_c['limit']) * 1 + \
            (df_dict['close'].iloc[-9:]==df_dict['limit'].iloc[-9:]).sum()
        return df_c[['new_s_dq_returns_l5d_keep_trend', 'new_s_dq_limit_up_counts_l10d', 'new_s_dq_close_ma10',
                     'new_s_dq_amount_ama120', 'new_s_dq_ma10_rate_keep_trend', 'new_s_dq_ama120_rate_keep_trend']]

    # 第三部分：策略轮训的下单逻辑，执行完成后，会输出每天的下单指令列表
    def handle_data(self, context):
        # 交易日历、因子数据、行情数据
        # current_date = context.current_date
        current_date = '20241220'
        pick = 15
        # start_date = TradingDate.get_offset_trading_date(current_date.strftime("%Y%m%d"), -60)
        # end_date = current_date.strftime("%Y%m%d")
        end_date = TradingDate.get_offset_trading_date(current_date, -1)
        start_date1 = TradingDate.get_offset_trading_date(end_date, -60)
        start_date2 = TradingDate.get_offset_trading_date(end_date, -120)

        # 提取行情数据，只能提取到截止T-1
        var_list = ['close', 'amount', 'limit', 'stopping', 'adjhigh', 'adjclose']
        df_dict = {}
        data = api_obj.wd_ashareeodprices_ori(start_date=start_date2, end_date=end_date, fields=['s_info_windcode', 'trade_dt']+['s_dq_'+var for var in var_list])
        data.columns = ['code', 'trade_date'] + var_list
        for var in var_list:
                df_dict[var] = data.pivot(index='trade_date', columns='code', values=var)

        # 有效样本(非涨跌停、停牌)
        valid = (df_dict['amount']>0)&(df_dict['close']!=df_dict['limit'])&(df_dict['close']!=df_dict['stopping'])
        # 尾盘抢筹，次日收益率>=2%
        Maxrt = ((df_dict['adjhigh'].shift(-1))/df_dict['adjclose']-1).dropna(axis=0,how='all') # 次日最高收益率作为未来一天的Y
        Maxrt_target = (Maxrt>=0.02).astype(int).iloc[-60:] # 截止到T-2

        Y = Maxrt_target[valid].stack().reset_index()
        Y.columns = ['trade_date','code','Y']

        f1 = api_obj.gj_ashare_dq_returns_keeptrend(start_date=start_date1, end_date=end_date, limit=None)[['s_info_windcode','trade_dt','s_dq_returns_l5d_keep_trend']].set_index(['trade_dt','s_info_windcode'])
        # f2 = api_obj.gj_ashare_dq_attachpeak(start_date=start_date1, end_date=end_date, limit=None)[['s_info_windcode','trade_dt','s_dq_adjclose_peak_value_lnd']].set_index(['trade_dt','s_info_windcode']).sort_index()
        f2 = api_obj.gj_ashare_dq_updown_limit(start_date=start_date1, end_date=end_date, limit=None)[['s_info_windcode','trade_dt','s_dq_limit_up_counts_l10d']].set_index(['trade_dt','s_info_windcode'])
        f3 = api_obj.gj_ashare_dq_closetrend(start_date=start_date1, end_date=end_date, limit=None)[['s_info_windcode','trade_dt','s_dq_close_ma10','s_dq_amount_ama120']].set_index(['trade_dt','s_info_windcode'])
        f4 = api_obj.gj_ashare_dq_ma_rate_keeptrend(start_date=start_date1, end_date=end_date, limit=None)[['s_info_windcode','trade_dt','s_dq_ma10_rate_keep_trend','s_dq_ama120_rate_keep_trend']].set_index(['trade_dt','s_info_windcode'])
        fc = pd.concat([f1,f2,f3,f4],axis=1).reset_index().rename({'trade_dt': 'trade_date', 's_info_windcode': 'code'}, axis=1) # 截止到T-1


        F = pd.merge(fc,Y,on=['code','trade_date'],how='right')
        F = F.dropna(axis=0,how='any') # 只包含feature和label全部非nan的样本，截止到T-2
        train = F
        
        X_train = train[['s_dq_returns_l5d_keep_trend', 's_dq_limit_up_counts_l10d', 's_dq_close_ma10', 
                         's_dq_amount_ama120', 's_dq_ma10_rate_keep_trend', 's_dq_ama120_rate_keep_trend']].values
        Y_train  = train['Y'].values

        nb = GaussianNB()
        nb.fit(X_train,Y_train)
        test = self.get_today(fc[fc['trade_date']==end_date], df_dict).dropna()
        X_test = test.values
        Y_test_pre = nb.predict(X_test)#测试集的预测值
        Y_test_prob = nb.predict_proba(X_test)[:, 1] # 测试集的预测概率
        result = pd.DataFrame({'pred': Y_test_pre, 'prob': Y_test_prob,
                               's_dq_ama120_rate_keep_trend': test['new_s_dq_ama120_rate_keep_trend']}, index=test.index)
        df_pick = result[result['pred']==1].sort_values(['pred','prob','s_dq_ama120_rate_keep_trend'])[-pick:]
        print(df_pick)

if __name__ == '__main__':
    RealProcess('EndofDayBuying', u'.\EndofDayBuying.py', 999, debug=True).run()

