# daily backtesting framework

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

MAX_INSTS = 10000

class BackTest:
    """
    cfg参数配置说明
    -----------------------------------
    key             value           comment

    startdate       str             回测起始日yyyymmdd
    enddate         str             回测终止日yyyymmdd
    alpha_id        str             回测的alpha pickle文件名, 通常是带_v1的原始版本
    tvr_day         list[str]       指定调仓日, None默认每日调仓, 输入'dd'的调仓日
    EmaDecay        int             EmaDecay参数
    RiskNeut        dict            RiskNeut参数
    IndNeut         dict            IndNeut参数(无参数)
    ProfitAna       dict            ProfitAna参数
    NLRiskNeut      dict            NLRiskNeut参数
    FactorAna       dict            FactorAna参数
    GroupFactorAna  dict            GroupFactorAna参数
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.alpha_id = cfg.get('alpha_id')

        self.calendar = pd.read_pickle('./data/calendar.pkl')
        self._datastart = self.get_offset_trd(self.startdate, -60) # alpha取数起始日, 多取60天方便decay
        self._alphastart = self.get_offset_trd(self.startdate, -1) # decay后的alpha截断日，需要多取一天防止区间首日信号为nan
        self.dateindex = self.calendar[(self.calendar>=self.startdate)&(self.calendar<=self.enddate)] # 回测区间
        self.alphaindex = self.calendar[(self.calendar>=self._alphastart)&(self.calendar<=self.enddate)] # alpha op相关的区间
        self.dataindex = self.calendar[(self.calendar>=self._datastart)&(self.calendar<=self.enddate)] # 取数区间

        self.alpha_df = pd.read_pickle(f'./dump/{self.alpha_id}.pkl').loc[self.dataindex]
        self.listdays = pd.read_pickle('./data/listdays.pkl').loc[self.alphaindex]
        self.suspend = pd.read_pickle('./data/is_suspend.pkl').loc[self.alphaindex]
        self.st = pd.read_pickle('./data/is_st.pkl').loc[self.alphaindex]
        self.price = pd.read_pickle('./data/close.pkl').loc[self.alphaindex]
        self.returns = pd.read_pickle('./data/returns.pkl').loc[self.dateindex]

        self.get_valid()

        # 确定固定调仓日
        self.tvr_day = cfg.get('tvr_day')
        if self.tvr_day:
            self.calendar_df = pd.DataFrame({'trade_date': self.dateindex})
            for i in range(len(self.tvr_day)):
                self.calendar_df[f'tvrday{i}'] = self.calendar_df['trade_date'].str[:-2]+self.tvr_day[i]
            valid_tvr_day = self.calendar_df.iloc[:, 1:].stack().unique()
            self.tvrindex = self.dateindex.iloc[self.dateindex.searchsorted(valid_tvr_day, side='left')] # 回测期内所有定点调仓日

        print(f'------------------------backtesting {self.alpha_id}-----------------------')
    
    def get_offset_trd(self, date, offset):
        return self.calendar[np.maximum(self.calendar.searchsorted(date)+offset, 0)]

    def exec_op(self):
        self.EmaDecay(self.cfg.get('EmaDecay')) # must needed step
        self.alpha_df.where(self.valid.shift(-1), inplace=True)
        for op in self.cfg.keys():
            if op in ['IndNeut', 'RiskNeut', 'NLRiskNeut']:
                exec(f'self.{op}(**self.cfg.get("{op}"))')

    def exec_ana(self):
        for op in self.cfg.keys():
            if op in ['ProfitAna', 'FactorAna', 'GroupFactorAna']:
                exec(f'self.{op}(**self.cfg.get("{op}"))')

    def __call__(self):
        self.exec_op()
        self.get_position()
        self.exec_ana()
        self.generate_pnl()
        self.stats()
    
    def EmaDecay(self, decay):
        self.alpha_df = self.alpha_df.ewm(span=decay, adjust=False).mean()
        self.alpha_df = self.alpha_df.loc[self.alphaindex]

    def get_valid(self):
            lp = self.price > 1 # exclude stocks whose price <= 1
            st = ~self.st # exclude st stocks
            rl = self.listdays >= 300 # exclude recently listed stocks and delisted stocks
            suspend = ~self.suspend # exclude suspended stocks
            bj = ~self.price.columns.str.contains('BJ')
            bj = pd.DataFrame(np.tile(bj, (len(self.price.index), 1)), index=self.price.index, columns=self.price.columns)
            valid = lp & st & rl & suspend & bj
            self.valid = valid

    def RiskNeut(self, factors: str):
        for factor in factors.split('|'):
            factor_df = pd.read_pickle(f'./data/{factor}.pkl').loc[self.alphaindex]
            factor_df.where(self.valid.shift(-1), inplace=True)
            mask = self.alpha_df.isna() | factor_df.isna()
            X = factor_df.copy()
            Y = self.alpha_df.copy()
            X.where(~mask, inplace=True)
            Y.where(~mask, inplace=True)
            meanX = X.mean(1)
            meanY = Y.mean(1)
            beta = (X.sub(meanX, 0)*Y.sub(meanY, 0)).sum(1) / ((X.sub(meanX, 0))**2).sum(1)
            alpha = meanY - beta * meanX
            self.alpha_df = Y.sub(alpha, 0).sub(X.mul(beta, 0))

    def IndNeut(self):
        ind_df = pd.read_pickle('./data/indint.pkl').loc[self.alphaindex]
        valid = self.valid.shift(-1) & (ind_df>=0)
        nInsts = ind_df.apply(lambda x: np.bincount(x[valid.loc[x.name]]), axis=1)
        nVals = ind_df.apply(lambda x: np.bincount(x[valid.loc[x.name]], 
                                                    self.alpha_df.loc[x.name][valid.loc[x.name]]), axis=1)
        IndMean = pd.DataFrame((nVals / nInsts).tolist(), index=ind_df.index).values
        self.alpha_df -= np.take_along_axis(IndMean, ind_df.values, 1)

    def NLRiskNeut(self, factors: str, group=2):
        def grouprank_row(row_group, row_alpha):
            df = pd.DataFrame({'group': row_group, 'alpha': row_alpha})
            return df.groupby('group')['alpha'].rank(pct=True)

        for factor in factors.split('|'):
            factor_df = pd.read_pickle(f'./data/{factor}.pkl').loc[self.alphaindex]
            factor_df.where(self.valid.shift(-1), inplace=True)
            mask = ~(self.alpha_df.isna() | factor_df.isna())
            factor_df.where(mask, inplace=True)
            self.alpha_df.where(mask, inplace=True)
            factor_df = factor_df.rank(1, pct=True)
            factor_group = np.floor(factor_df * group)
            factor_group[factor_group == group] -= 1
            self.alpha_df = self.alpha_df.apply(lambda row: grouprank_row(factor_group.loc[row.name], row), axis=1)

    def FactorAna(self, factors: str):
        exposure = {}
        for factor in factors.split('|'):
            factor_df = pd.read_pickle(f'./data/{factor}.pkl').loc[self.alphaindex]
            factor_df = factor_df.sub(factor_df.mean(1), axis=0).div(factor_df.std(1), axis=0)
            exp = self.alpha_rk.mul(factor_df).mean(1).mean()
            exposure[factor] = exp
        exposure = pd.DataFrame(exposure, index=['exposure'])
        output = 'total exposure: \n'
        for factor in exposure.columns:
            output += '{}: {:.4f}| '.format(factor, exposure[factor].iloc[0])
        output = output[:-2]
        print(output)

    def GroupFactorAna(self, factors: str, group=10):
        exposure = {}
        for factor in factors.split('|'):
            factor_df = pd.read_pickle(f'./data/{factor}.pkl').loc[self.alphaindex]
            pos_df = self.alpha_rk.copy().fillna(-1)
            gpos = np.floor((pos_df+0.5)*group)
            gpos[gpos==group] -= 1
            gpos = gpos.astype(int)
            valid = gpos>=0
            factor_df.where(valid, inplace=True)
            factor_df = factor_df.rank(axis=1, method='dense', pct=True)-0.5
            nInsts = gpos.apply(lambda x: np.bincount(x[valid.loc[x.name]]), axis=1)
            nVals = gpos.apply(lambda x: np.bincount(x[valid.loc[x.name]], factor_df.loc[x.name][valid.loc[x.name]]), axis=1)
            gexp = pd.DataFrame((nVals / nInsts).tolist(), index=gpos.index).mean()
            exposure[factor] = gexp
        exposure = pd.DataFrame(exposure)
        output = ''
        for factor in exposure.columns:
            output += f'{factor} group exposure: \n'
            for idx, row in exposure[factor].items():
                output += '{}: {:.2f}| '.format(idx, row)
            output = output[:-2] + '\n'
        output = output[:-1]
        print(output)

    def ProfitAna(self, retdays=5, group=10):
        pos_df = self.pos_df.copy().fillna(-1)
        ret_df = self.returns.rolling(retdays).sum().shift(-retdays).fillna(0)
        gpos = np.floor((pos_df+0.5)*group)
        gpos[gpos==group] -= 1
        gpos = gpos.astype(int)
        # group avgret
        valid = gpos>=0
        nInsts = gpos.apply(lambda x: np.bincount(x[valid.loc[x.name]]), axis=1)
        nVals = gpos.apply(lambda x: np.bincount(x[valid.loc[x.name]], ret_df.loc[x.name][valid.loc[x.name]]), axis=1)
        gret = pd.DataFrame((nVals / nInsts).tolist(), index=gpos.index).mean() * 250/retdays
        output = ''
        for idx, row in gret.items():
            output += '{}: {:.4%}| '.format(idx, row)
        output = output[:-2]
        print(f'group {retdays} days return:')
        print(output)
        # group winrate
        # gpos[ret_df<=0] = -1
        # valid = gpos>=0
        # vldidx = valid.index[valid.sum(1)>0]
        # nPosInsts = gpos.loc[vldidx].apply(lambda x: np.bincount(x[valid.loc[x.name]]), axis=1)
        # gwinr = pd.DataFrame((nPosInsts / nInsts[vldidx]).tolist(), index=vldidx).mean()
        # output = ''
        # for idx, row in gwinr.items():
        #     output += '{}: {:.2%}| '.format(idx, row)
        # output = output[:-2]
        # print(f'group {retdays} days winrate:')
        # print(output)

    def get_position(self):
        self.alpha_rk = self.alpha_df.rank(axis=1, method='dense', pct=True)-0.5
        pos_df = self.alpha_rk.shift(1).loc[self.dateindex]
        if self.tvr_day:
            self.pos_df = pd.DataFrame(np.nan, index=self.dateindex, columns=pos_df.columns)
            self.pos_df.loc[self.tvrindex] = pos_df.loc[self.tvrindex]
            self.pos_df.ffill(inplace=True)
            self.pos_df[pos_df.isna()] = np.nan
        else:
            self.pos_df = pos_df
        self.wgt_df = self.pos_df.div(self.pos_df.abs().sum(1), axis=0)

    def generate_pnl(self):
        ret_df = self.returns.shift(-1)
        pnl = self.wgt_df.mul(ret_df).sum(1).shift(1)*2
        self.pnl = pd.DataFrame({'pnl': pnl})
        self.pnl['nav'] = self.pnl['pnl'].cumsum()

    def stats(self):
        # total
        ret_T = self.pnl['pnl'].mean() * 250
        sp_T = self.pnl['pnl'].mean() / self.pnl['pnl'].std() * np.sqrt(250)
        self.pnl['dd_T'] = self.pnl['nav'].cummax() - self.pnl['nav']
        mdd_T = self.pnl['dd_T'].max()
        dd_end_T = self.pnl['dd_T'].idxmax()
        dd_start_T = self.pnl.loc[:dd_end_T, 'nav'].idxmax()
        dpos = self.wgt_df.fillna(0).diff()
        dpos[self.wgt_df.isna()&self.wgt_df.shift(1).isna()] = np.nan
        self.pnl['dpos'] = dpos.abs().sum(1)
        tvr_T = self.pnl['dpos'].mean()
        winr_T = (self.pnl['pnl']>0).mean()

        # yearly
        self.pnl['year'] = self.pnl.index.str[:4]
        gpnl = self.pnl.groupby('year')
        self.pnl['dd_y'] = gpnl['nav'].cummax() - self.pnl['nav']
        ret_y = gpnl['pnl'].mean() * 250
        sp_y = gpnl['pnl'].mean() / gpnl['pnl'].std() * np.sqrt(250)
        mdd_y = gpnl['dd_y'].max()
        dd_end_y = gpnl['dd_y'].idxmax()
        dd_start_y = gpnl.apply(lambda x: x.loc[:dd_end_y[x['year'].values[0]], 'nav'].idxmax())
        tvr_y = gpnl['dpos'].mean()
        winr_y = gpnl['pnl'].apply(lambda x: (x>0).mean())

        # output
        index1 = gpnl.head(1).index
        index2 = gpnl.tail(1).index
        out = pd.DataFrame({'ret': ret_y.values*100,
                            'sp': sp_y.values,
                            'dd': mdd_y.values*100,
                            'dd_start': dd_start_y.values,
                            'dd_end': dd_end_y.values,
                            'tvr': tvr_y.values*100,
                            'winr': winr_y.values*100}, index=pd.MultiIndex.from_arrays([index1, index2], names=['from', 'to']))
        out.loc[(self.dateindex.iloc[0], self.dateindex.iloc[-1]), :] = [ret_T*100, sp_T, mdd_T*100, dd_start_T, dd_end_T, tvr_T*100, winr_T*100]
        print(out)

    def sigbt(self, topN=200, weight_mode=0, benchmark: str=None, timing_ts: pd.Series=None):
        """
        纯多头信号回测参数说明
        ----------------------------------------------------------
        key             value           comment
        
        topN            int             多头持仓个数
        weight_mode     int             0-线性持仓, 1-等权持仓
        benchmark       str(None)       None计算绝对收益, 否则计算超额
        timing_ts       DataFrame       若输入则多头策略带择时信号, 决定每日持有股票数
        """
        total = (~self.pos_df.isna()).sum(1)

        if timing_ts is not None:
            # timing_th = pd.read_csv('./data_attached/大盘择时未来10日信号_涨1跌0震荡2.csv', header=None, index_col=0).squeeze()
            # timing_th.index = timing_th.index.astype('str')
            timing_ratio = timing_ts[self.dateindex]
            # timing_th = timing_th.map({1: 1, 0: 0.5, 2: 0.75})
            th = 0.5-timing_ratio*topN/total
        else:
            th = 0.5-topN/total
        pos = self.pos_df.values
        pos[pos<=th.values[:, np.newaxis]] = np.nan
        self.pos_df[:] = pos

        if weight_mode:
            nancount = self.pos_df.notna().sum(1).values
            wgt = np.divide(1, nancount, where=nancount!=0, out=np.full_like(nancount, np.nan, dtype=np.float64))
            self.wgt_df = self.pos_df.notna()*wgt[:, np.newaxis]
            self.wgt_df[self.wgt_df==0] = np.nan
        else:
            self.wgt_df = self.pos_df.div(self.pos_df.sum(1), axis=0)
        ret_df = self.returns.shift(-1)
        pnl = self.wgt_df.mul(ret_df).sum(1).shift(1)

        if benchmark:
            mkt = pd.read_pickle(f'./data/{benchmark}.pkl').loc[self.dateindex, 'pct_change'] / 100
            pnl -= mkt
        self.pnl = pd.DataFrame({'pnl': pnl})
        self.pnl['nav'] = self.pnl['pnl'].cumsum()
        self.stats()

    def dump(self):
        self.alpha_rk.to_pickle(f'./dump/{self.alpha_id[:-3]}_v2.pkl')
        print(f'------------------------{self.alpha_id[:-3]}_v2 dumped-----------------------')

class Alpha:
    """
    cfg参数配置说明
    --------------------------------------------------
    key             value           comment

    startdate       str             因子计算起始日
    enddate         str             因子计算终止日
    因子计算默认回溯120日, 保证回测起始日有因子值
    """
    def __init__(self, cfg):
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.calendar = pd.read_pickle('./data/calendar.pkl')
        _startdate = self.calendar.searchsorted(self.startdate, 'left')-120
        _enddate = self.calendar.searchsorted(self.enddate, side='right')
        _mindate = self.calendar.searchsorted('20140101', 'left')

        self.dateindex = self.calendar.iloc[np.maximum(_startdate, _mindate):_enddate]
        self.alpha_df = pd.DataFrame()
        
    def hf_to_daily(self):
        for didx in self.dateindex:
            daily = self.generate_daily(didx)
            self.alpha_df.loc[didx] = daily

    def generate_daily(self, didx):
        pass

    def generate_alpha(self):
        pass

    def dump(self):
        self.alpha_df.to_pickle(f'./dump/{self.__class__.__name__}_v1.pkl')