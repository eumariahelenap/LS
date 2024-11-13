from typing import Literal as Lit
import itertools as it
from numpy import nan
import numpy as np
import pandas as pd
import deps as _
import gc



def GET_HIST_DATA(TICKERS, SRC, UNTIL, start=None):
    
    Raw = _.yf_api_hdatas(TICKERS, interval=SRC, period=UNTIL, start=start)

    Src = pd.DataFrame()
    Src['N']      = Raw['N']
    Src['Ticker'] = Raw['Ticker']
    Src['Date']   = Raw['Date']
    Src['Price']  = Raw['Adj Close']
    return Src

def MAKE_PAIRS_KEYS(Src):
    Pairs = _.pd_make_pairs(Src['Ticker'], unique=1, prefix='Ticker.')
    Date  = Src[['Date']].drop_duplicates().sort_values('Date', ascending=False) . reset_index(drop=1)
    return pd.merge(Pairs, Date, 'cross')

def MAKE_PAIRS_SOURCE(Keys, Src, TF, FULL_CLOCK):
    
    Df = Keys.copy()
    Df['Trans']   = _.in_transit(Df, 'Date', TF, FULL_CLOCK)
    Df['Price.X'] = _.pd_relate(Keys, ['Ticker.X','Date'], Src, ['Ticker','Date'], 'Price') .add_suffix('.X')
    Df['Price.Y'] = _.pd_relate(Keys, ['Ticker.Y','Date'], Src, ['Ticker','Date'], 'Price') .add_suffix('.Y')
    Df = Df.dropna().reset_index(drop=1)
    
    Df.insert(0, 'N', _.Nth(Keys, 'Date', By='Ticker.XY', sign=(-1), asc=0, adj=(+1)))

    Df['LnPrc.X'] = _.log(Df['Price.X'])
    Df['LnPrc.Y'] = _.log(Df['Price.Y'])
    return Df



def CALCULATE_PRC_STATS(Src, K, PRDS, SLGS):
    P1, P2, P3, P4, P5, P6 = PRDS
    S1, S2, S3, S4, S5, S6 = SLGS

    Df = Src.loc[:, None:'LnPrc.Y'].copy()
    
    def _correlate(win, minWin):
        return _.pd_rollby_correlate('pandas',  Df,'LnPrc.Y','LnPrc.X',  By='Ticker.XY', win=win, minWin=minWin,  stp=(-1), _mlt=100, _rnd=1)

    Df['Cor']       = ''
    Df[f'Cor {S1}'] = _correlate(win=P1*K, minWin=P1*K)  # nan
    Df[f'Cor {S2}'] = _correlate(win=P2*K, minWin=P1*K)  # nan
    Df[f'Cor {S3}'] = _correlate(win=P3*K, minWin=P2*K)  # nan
    Df[f'Cor {S4}'] = _correlate(win=P4*K, minWin=P3*K)  # nan
    Df[f'Cor {S5}'] = _correlate(win=P5*K, minWin=P4*K)  # nan
    Df[f'Cor {S6}'] = _correlate(win=P6*K, minWin=P5*K)  # nan
    return Df

def CALCULATE_VARIATIONS(Src, K, PRDS, SLGS):
    P1, P2, P3, P4, P5, P6 = PRDS
    S1, S2, S3, S4, S5, S6 = SLGS
    
    Df = Src.copy()

    def _Return(Val, N):
        return _.Return(Df, Val, By='Ticker.XY', N=N, stp=(-1), mult=100, rnd=2)

    Df['Vars.X+']       = ''
    Df[f'Var.X +{S1}']  = _Return('Price.X', N=P1*K)
    Df[f'Var.X +{S2}']  = _Return('Price.X', N=P2*K)
    Df[f'Var.X +{S3}']  = _Return('Price.X', N=P3*K)
    Df[f'Var.X +{S4}']  = _Return('Price.X', N=P4*K)
    Df[f'Var.X +{S5}']  = _Return('Price.X', N=P5*K)
    Df[f'Var.X +{S6}']  = _Return('Price.X', N=P6*K)

    Df['Vars.Y+']       = ''
    Df[f'Var.Y +{S1}']  = _Return('Price.Y', N=P1*K)
    Df[f'Var.Y +{S2}']  = _Return('Price.Y', N=P2*K)
    Df[f'Var.Y +{S3}']  = _Return('Price.Y', N=P3*K)
    Df[f'Var.Y +{S4}']  = _Return('Price.Y', N=P4*K)
    Df[f'Var.Y +{S5}']  = _Return('Price.Y', N=P5*K)
    Df[f'Var.Y +{S6}']  = _Return('Price.Y', N=P6*K)

    def _Variat(Val, N):
        return _.Variat(Df, Val, By='Ticker.XY', N=N, stp=(-1), mult=100, rnd=2)

    Df['Vars.X-']       = ''
    Df[f'Var.X -{S1}']  = _Variat('Price.X', N=P1*K)
    Df[f'Var.X -{S2}']  = _Variat('Price.X', N=P2*K)
    Df[f'Var.X -{S3}']  = _Variat('Price.X', N=P3*K)
    Df[f'Var.X -{S4}']  = _Variat('Price.X', N=P4*K)
    Df[f'Var.X -{S5}']  = _Variat('Price.X', N=P5*K)
    Df[f'Var.X -{S6}']  = _Variat('Price.X', N=P6*K)

    Df['Vars.Y-']       = ''
    Df[f'Var.Y -{S1}']  = _Variat('Price.Y', N=P1*K)
    Df[f'Var.Y -{S2}']  = _Variat('Price.Y', N=P2*K)
    Df[f'Var.Y -{S3}']  = _Variat('Price.Y', N=P3*K)
    Df[f'Var.Y -{S4}']  = _Variat('Price.Y', N=P4*K)
    Df[f'Var.Y -{S5}']  = _Variat('Price.Y', N=P5*K)
    Df[f'Var.Y -{S6}']  = _Variat('Price.Y', N=P6*K)

    def _pct_point(Y, X):
        return _.pct_point(Df, Y, X, geo=['X','Y','Z'], pct=['X','Y','Z'], rnd=1)

    Df['Gap']       = ''
    Df[f'Gap {S1}'] = _pct_point(f'Var.Y -{S1}',  f'Var.X -{S1}')
    Df[f'Gap {S2}'] = _pct_point(f'Var.Y -{S2}',  f'Var.X -{S2}')
    Df[f'Gap {S3}'] = _pct_point(f'Var.Y -{S3}',  f'Var.X -{S3}')
    Df[f'Gap {S4}'] = _pct_point(f'Var.Y -{S4}',  f'Var.X -{S4}')
    Df[f'Gap {S5}'] = _pct_point(f'Var.Y -{S5}',  f'Var.X -{S5}')
    Df[f'Gap {S6}'] = _pct_point(f'Var.Y -{S6}',  f'Var.X -{S6}')

    return Df

def CALCULATE_VAR_STATS(Src, Vars, K, PRDS, SLGS):
    P1, P2, P3, P4, P5, P6 = PRDS
    S1, S2, S3, S4, S5, S6 = SLGS

    Df = Src.copy()


    def _roll_stats(algo, Val, win, minWin):
        return _.pd_roll_calc(algo, Vars, Val, By='Ticker.XY', win=win, minWin=minWin, stp=(-1), _in='p100', _out='m100', fig=1)

    def add_roll_stats(Name, algo, P, S):
        Df[f'Gap.{P} {Name}.{P1}'] = _roll_stats(algo, f'Gap {S}', win=P1*K, minWin=P1*K)
        Df[f'Gap.{P} {Name}.{P2}'] = _roll_stats(algo, f'Gap {S}', win=P2*K, minWin=P1*K)
        Df[f'Gap.{P} {Name}.{P3}'] = _roll_stats(algo, f'Gap {S}', win=P3*K, minWin=P2*K)
        Df[f'Gap.{P} {Name}.{P4}'] = _roll_stats(algo, f'Gap {S}', win=P4*K, minWin=P3*K)
        Df[f'Gap.{P} {Name}.{P5}'] = _roll_stats(algo, f'Gap {S}', win=P5*K, minWin=P4*K)
        Df[f'Gap.{P} {Name}.{P6}'] = _roll_stats(algo, f'Gap {S}', win=P6*K, minWin=P5*K)
        pass
    
    Df['Gap Avg'] = ''
    add_roll_stats('Avg', 'geo_mean', P1, S1)
    add_roll_stats('Avg', 'geo_mean', P2, S2)
    add_roll_stats('Avg', 'geo_mean', P3, S3)
    add_roll_stats('Avg', 'geo_mean', P4, S4)
    add_roll_stats('Avg', 'geo_mean', P5, S5)
    add_roll_stats('Avg', 'geo_mean', P6, S6)

    Df['Gap Dev'] = ''
    add_roll_stats('Dev', 'geo_std', P1, S1)
    add_roll_stats('Dev', 'geo_std', P2, S2)
    add_roll_stats('Dev', 'geo_std', P3, S3)
    add_roll_stats('Dev', 'geo_std', P4, S4)
    add_roll_stats('Dev', 'geo_std', P5, S5)
    add_roll_stats('Dev', 'geo_std', P6, S6)


    def _ZLevel(Val, Avg, Dev):
        return _.ZLevel(..., Vars[Val], Df[Avg], Df[Dev], Pct=[1,1,1], Log=[1,1,1], fig=3) 

    def add_ZLevel(P, S):
        Df[f'Gap.{P} Lvl.{P1}'] = _ZLevel(f'Gap {S}',  f'Gap.{P} Avg.{P1}',  f'Gap.{P} Dev.{P1}')
        Df[f'Gap.{P} Lvl.{P2}'] = _ZLevel(f'Gap {S}',  f'Gap.{P} Avg.{P2}',  f'Gap.{P} Dev.{P2}')
        Df[f'Gap.{P} Lvl.{P3}'] = _ZLevel(f'Gap {S}',  f'Gap.{P} Avg.{P3}',  f'Gap.{P} Dev.{P3}')
        Df[f'Gap.{P} Lvl.{P4}'] = _ZLevel(f'Gap {S}',  f'Gap.{P} Avg.{P4}',  f'Gap.{P} Dev.{P4}')
        Df[f'Gap.{P} Lvl.{P5}'] = _ZLevel(f'Gap {S}',  f'Gap.{P} Avg.{P5}',  f'Gap.{P} Dev.{P5}')
        Df[f'Gap.{P} Lvl.{P6}'] = _ZLevel(f'Gap {S}',  f'Gap.{P} Avg.{P6}',  f'Gap.{P} Dev.{P6}')
        pass

    Df['Gap Lvl'] = ''
    add_ZLevel(P1, S1)
    add_ZLevel(P2, S2)
    add_ZLevel(P3, S3)
    add_ZLevel(P4, S4)
    add_ZLevel(P5, S5)
    add_ZLevel(P6, S6)

    return Df

def LONG_SHORT_RETURNS(Src, Vars, PRDS, SLGS):
    P1, P2, P3, P4, P5, P6 = PRDS
    S1, S2, S3, S4, S5, S6 = SLGS
    
    Df = pd.DataFrame()

    # Df['N']         = Src2['N'] 
    # Df['Date']      = Src2['Date'] 
    # Df[' '] = ''
    # Df['Ticker.Y']  = Src2['Ticker.Y']
    # Df['Ticker.X']  = Src2['Ticker.X']
    # Df['  '] = ''
    # Df['Var.Y 1Y']  = Vars['Var.Y -1Y']
    # Df['Gap 1Y']    = Vars['Gap 1Y']
    # Df['Var.X 1Y']  = Vars['Var.X -1Y']
    # Df['   '] = ''

    is_ngtv_1, is_pstv_1 = (Vars[f'Gap {S1}'] < 0), (0 < Vars[f'Gap {S1}'])
    is_ngtv_2, is_pstv_2 = (Vars[f'Gap {S2}'] < 0), (0 < Vars[f'Gap {S2}'])
    is_ngtv_3, is_pstv_3 = (Vars[f'Gap {S3}'] < 0), (0 < Vars[f'Gap {S3}'])
    is_ngtv_4, is_pstv_4 = (Vars[f'Gap {S4}'] < 0), (0 < Vars[f'Gap {S4}'])
    is_ngtv_5, is_pstv_5 = (Vars[f'Gap {S5}'] < 0), (0 < Vars[f'Gap {S5}'])
    is_ngtv_6, is_pstv_6 = (Vars[f'Gap {S6}'] < 0), (0 < Vars[f'Gap {S6}'])

    Df['Long Short']  = ''
    Df[f'Long {S1}']  = np.select([is_ngtv_1, is_pstv_1], [Src['Ticker.Y'], Src['Ticker.X']], '')
    Df[f'Short {S1}'] = np.select([is_ngtv_1, is_pstv_1], [Src['Ticker.X'], Src['Ticker.Y']], '')

    Df[f'Long {S2}']  = np.select([is_ngtv_2, is_pstv_2], [Src['Ticker.Y'], Src['Ticker.X']], '')
    Df[f'Short {S2}'] = np.select([is_ngtv_2, is_pstv_2], [Src['Ticker.X'], Src['Ticker.Y']], '')

    Df[f'Long {S3}']  = np.select([is_ngtv_3, is_pstv_3], [Src['Ticker.Y'], Src['Ticker.X']], '')
    Df[f'Short {S3}'] = np.select([is_ngtv_3, is_pstv_3], [Src['Ticker.X'], Src['Ticker.Y']], '')

    Df[f'Long {S4}']  = np.select([is_ngtv_4, is_pstv_4], [Src['Ticker.Y'], Src['Ticker.X']], '')
    Df[f'Short {S4}'] = np.select([is_ngtv_4, is_pstv_4], [Src['Ticker.X'], Src['Ticker.Y']], '')

    Df[f'Long {S5}']  = np.select([is_ngtv_5, is_pstv_5], [Src['Ticker.Y'], Src['Ticker.X']], '')
    Df[f'Short {S5}'] = np.select([is_ngtv_5, is_pstv_5], [Src['Ticker.X'], Src['Ticker.Y']], '')

    Df[f'Long {S6}']  = np.select([is_ngtv_6, is_pstv_6], [Src['Ticker.Y'], Src['Ticker.X']], '')
    Df[f'Short {S6}'] = np.select([is_ngtv_6, is_pstv_6], [Src['Ticker.X'], Src['Ticker.Y']], '')
    
    Df['LS Vars']          = ''
    Df[f'Long Var +{S1}']  = np.select([is_ngtv_1, is_pstv_1], [Vars[f'Var.Y +{S1}'], Vars[f'Var.X +{S1}']], nan)
    Df[f'Short Var +{S1}'] = np.select([is_ngtv_1, is_pstv_1], [Vars[f'Var.X +{S1}'], Vars[f'Var.Y +{S1}']], nan)

    Df[f'Long Var +{S2}']  = np.select([is_ngtv_2, is_pstv_2], [Vars[f'Var.Y +{S2}'], Vars[f'Var.X +{S2}']], nan)
    Df[f'Short Var +{S2}'] = np.select([is_ngtv_2, is_pstv_2], [Vars[f'Var.X +{S2}'], Vars[f'Var.Y +{S2}']], nan)

    Df[f'Long Var +{S3}']  = np.select([is_ngtv_3, is_pstv_3], [Vars[f'Var.Y +{S3}'], Vars[f'Var.X +{S3}']], nan)
    Df[f'Short Var +{S3}'] = np.select([is_ngtv_3, is_pstv_3], [Vars[f'Var.X +{S3}'], Vars[f'Var.Y +{S3}']], nan)

    Df[f'Long Var +{S4}']  = np.select([is_ngtv_4, is_pstv_4], [Vars[f'Var.Y +{S4}'], Vars[f'Var.X +{S4}']], nan)
    Df[f'Short Var +{S4}'] = np.select([is_ngtv_4, is_pstv_4], [Vars[f'Var.X +{S4}'], Vars[f'Var.Y +{S4}']], nan)

    Df[f'Long Var +{S5}']  = np.select([is_ngtv_5, is_pstv_5], [Vars[f'Var.Y +{S5}'], Vars[f'Var.X +{S5}']], nan)
    Df[f'Short Var +{S5}'] = np.select([is_ngtv_5, is_pstv_5], [Vars[f'Var.X +{S5}'], Vars[f'Var.Y +{S5}']], nan)

    Df[f'Long Var +{S6}']  = np.select([is_ngtv_6, is_pstv_6], [Vars[f'Var.Y +{S6}'], Vars[f'Var.X +{S6}']], nan)
    Df[f'Short Var +{S6}'] = np.select([is_ngtv_6, is_pstv_6], [Vars[f'Var.X +{S6}'], Vars[f'Var.Y +{S6}']], nan)

    Df['LS Rtn']        = ''
    Df[f'LS Rtn +{S1}'] = ((Df[f'Long Var +{S1}'] - Df[f'Short Var +{S1}']) /2) .round(2)
    Df[f'LS Rtn +{S2}'] = ((Df[f'Long Var +{S2}'] - Df[f'Short Var +{S2}']) /2) .round(2)
    Df[f'LS Rtn +{S3}'] = ((Df[f'Long Var +{S3}'] - Df[f'Short Var +{S3}']) /2) .round(2)
    Df[f'LS Rtn +{S4}'] = ((Df[f'Long Var +{S4}'] - Df[f'Short Var +{S4}']) /2) .round(2)
    Df[f'LS Rtn +{S5}'] = ((Df[f'Long Var +{S5}'] - Df[f'Short Var +{S5}']) /2) .round(2)
    Df[f'LS Rtn +{S6}'] = ((Df[f'Long Var +{S6}'] - Df[f'Short Var +{S6}']) /2) .round(2)

    return Df

def CALCULATE_ABSOLUTES(Vars, VSts):
    Inds = pd.concat([ Vars.loc[:, 'Gap':None], VSts.loc[:, 'Gap Avg':None] ], axis=1)
    IAbs = Inds.map(pd.to_numeric, errors='coerce').abs().add_prefix('Abs ')
    return IAbs

def CALCULATE_GAP_VARIATION(Src, IAbs, K, PRDS, SLGS):
    P1, P2, P3, P4, P5, P6 = PRDS
    S1, S2, S3, S4, S5, S6 = SLGS

    Df = pd.concat([ Src.loc[:, None:'Date'], IAbs.loc[:, None:f'Abs Gap {S6}'] ], axis=1)

    def add_variat_2nd(Val, G):
        Df[f'Gap.{G} Var.{P1}'] = _.Variat_2nd(Df, Val, By='Ticker.XY', N=P1*K, stp=(-1), rnd=2)
        Df[f'Gap.{G} Var.{P2}'] = _.Variat_2nd(Df, Val, By='Ticker.XY', N=P2*K, stp=(-1), rnd=2)
        Df[f'Gap.{G} Var.{P3}'] = _.Variat_2nd(Df, Val, By='Ticker.XY', N=P3*K, stp=(-1), rnd=2)
        Df[f'Gap.{G} Var.{P4}'] = _.Variat_2nd(Df, Val, By='Ticker.XY', N=P4*K, stp=(-1), rnd=2)
        Df[f'Gap.{G} Var.{P5}'] = _.Variat_2nd(Df, Val, By='Ticker.XY', N=P5*K, stp=(-1), rnd=2)
        Df[f'Gap.{G} Var.{P6}'] = _.Variat_2nd(Df, Val, By='Ticker.XY', N=P6*K, stp=(-1), rnd=2)
        pass

    Df['Gap Var'] = ''
    add_variat_2nd(f'Abs Gap {S1}', P1)
    add_variat_2nd(f'Abs Gap {S2}', P2)
    add_variat_2nd(f'Abs Gap {S3}', P3)
    add_variat_2nd(f'Abs Gap {S4}', P4)
    add_variat_2nd(f'Abs Gap {S5}', P5)
    add_variat_2nd(f'Abs Gap {S6}', P6)
    return Df

def COMBINE_FRAMES(Src, LS, PSts, Vars, VSts, IAbs, GVar):
    return pd.concat([
        Src, 
        PSts.loc[:, 'Cor':None], 
        LS, 
        Vars.loc[:, 'Vars.X+':None], 
        VSts.loc[:, 'Gap Avg':None], 
        IAbs, 
        GVar.loc[:, 'Gap Var':None]
    ], axis=1)








def CALCULATE_RANKS(Frames, PRDS, SLGS):

    P1, P2, P3, P4, P5, P6 = PRDS
    S1, S2, S3, S4, S5, S6 = SLGS

    Rnk_Rtn = _.HRnk(Frames, _.pd_columns(Frames, f'LS Rtn +{S1}', f'LS Rtn +{S6}'), By='Date', prefix='High ', _as='float')
    Rnk_Rtn.insert(0, 'High LS Rtn Rank', '')

    Rnk_Cor = _.HRnk(Frames, _.pd_columns(Frames, f'Cor {S1}', f'Cor {S6}'), By='Date', prefix='High ', _as='float')
    Rnk_Cor.insert(0, 'High Cor Rank', '')
    
    Rnk_Inds = _.RANK('BANDED', Frames, feats=['Gap',None], Keys=['Ticker.X','Ticker.Y','Ticker.XY'], By='Date', method='dense')

    return Rnk_Rtn, Rnk_Cor, Rnk_Inds

def COMBINE_RANKS(Src, Rnk_Rtn, Rnk_Cor, Rnk_Inds):
    return pd.concat([
        Src.loc[:, None:'Trans'], 
        Rnk_Rtn, 
        Rnk_Cor, 
        Rnk_Inds.loc[:, 'High Rank':None], 
    ], axis=1)

def COMBINE_DATAS(Frames, Ranks):
    return pd.concat([
        Frames,
        Ranks.loc[:, 'High LS Rtn Rank':None], 
    ], axis=1)

def SORT_SCREENER(Data):
    Scrn = Data.sort_values('Date', ascending=False) . reset_index(drop=1)
    Scrn['N'] = _.Nth(Scrn, 'Date', By='', sign=(-1), asc=0, adj=(+1))
    return Scrn



def SELECT(Scrn, GRAN):
    Df = Scrn[(Scrn['N']<=(-GRAN)) & (Scrn['Trans']==1)] . reset_index(drop=1)
    Df['N'] = _.Nth(Df, 'Date', By='', sign=(-1), asc=0, adj=0)
    return Df

def RANK_COMBINATION(Src,  Keys, By,  Ind_1, Ind_2,  WGHTS, NORM_WGHTS=False):

    WGHTS = _.TotalNorm(WGHTS, norm=NORM_WGHTS)
    Keys  = np.ravel(Keys).tolist()
    By    = np.ravel(By).tolist()
    Df    = Src[[*Keys,*By]].copy()

    Df['Rank Comb'] = ''
    Df['Art Comb']  = (Src[Ind_1] *  WGHTS[0]  +  Src[Ind_2] *  WGHTS[1])
    Df['Pwr Comb']  = (Src[Ind_1] ** WGHTS[0]  +  Src[Ind_2] ** WGHTS[1])
    Df['Geo Comb']  = (Src[Ind_1] ** WGHTS[0]  *  Src[Ind_2] ** WGHTS[1])

    Df['High Art']  = Df.groupby(By)['Art Comb'].rank(method='dense', ascending=1)
    Df['Low Art']   = Df.groupby(By)['Art Comb'].rank(method='dense', ascending=0)

    Df['High Pwr']  = Df.groupby(By)['Pwr Comb'].rank(method='dense', ascending=1)
    Df['Low Pwr']   = Df.groupby(By)['Pwr Comb'].rank(method='dense', ascending=0)

    Df['High Geo']  = Df.groupby(By)['Geo Comb'].rank(method='dense', ascending=1)
    Df['Low Geo']   = Df.groupby(By)['Geo Comb'].rank(method='dense', ascending=0)
    return Df

def TOP_CLASSIFCATION(RnkComb, RankBy, TOP):
    return _.pd_isin_top(RnkComb, Keys=['N','Ticker.XY','Date'], Rank=RankBy, TOP=TOP,  with_RnkVal=1)

def COMBINE_CLASS_RCOMB(InTop, RnkComb):
    return pd.concat([  InTop, RnkComb.loc[:,'Rank Comb':None]  ], axis=1)

def INSERT_CLASS_RCOMB_IN_DFRAME(Src, Class_RComb): 
    return _.pd_insert_df(Src, 'before','High LS Rtn Rank',  Class_RComb.loc[:, 'Rnk Val':None])

def MAKE_AGGREGATION_KEYS(Src, TOP):
    TLINE = Src['N'].sort_values(ascending=0).unique()
    return pd.DataFrame(it.product(TLINE, TOP), columns=['N', 'Top'])

def ADD_WALLET_PERFORMANCE(Agg, Data, TF, pwr, COLS):

    def _filter(Df, TL, TOP, Rtn):
        COND_TLINE = (Df['N'] == TL)
        COND_TOP   = (Df[f'InTop {TOP}'] == 1)
        COND_ALL   = (COND_TLINE & COND_TOP)
        return Df[COND_ALL][Rtn]

    def _mean(S, rnd, alt=nan):
        try:    return S.mean().round(rnd)
        except: return alt


    Agg['Count'] = Agg.apply(lambda x: (
        _filter(Data, TL=x.N, TOP=int(x.Top), Rtn=f'LS Rtn +1{TF}').count()
    ), axis=1)

    Agg['Return'] = Agg.apply(lambda x: (
        _mean(_filter(Data, TL=x.N, TOP=int(x.Top), Rtn=f'LS Rtn +1{TF}'), rnd=2)
    ), axis=1)

    Agg['CAGR']  = _.CAGR(Agg,  'Return', By='Top', stp=(-1), pwr=pwr)
    Agg['Worth'] = _.Worth(Agg, 'Return', By='Top', stp=(-1))
    Agg['Expnt'] = _.Expnt(Agg,  'Worth', By='Top', stp=(-1))

    Agg['Inds']  = ''
    Agg[COLS]    = Agg.apply(lambda x: (
        _mean(_filter(Data, TL=x.N, TOP=int(x.Top), Rtn=COLS), rnd=2)
    ), axis=1)

    return Agg[['N','Top', 'Count','Expnt','Worth','CAGR','Return', 'Inds',*COLS]]

def FINAL_RESULTS(Agg, TOP, COLS):

    Top = pd.DataFrame({ 'Top':TOP })

    Top['Count'] = Agg.groupby('Top').apply(lambda By: (
        By['Count'].mean() 
    )).reset_index(drop=1) .apply(round)

    Top[['Expnt', 'Worth', 'CAGR']] = Agg.groupby('Top').apply(lambda By: (
        By[['Expnt', 'Worth', 'CAGR']].head(1)
    )).reset_index(drop=1)

    Top['Return'] = Agg.groupby('Top').apply(lambda By: (
        np.round(By['Return'].mean(), 3)
    )).reset_index(drop=1)

    Top['Inds'] = ''
    Top[COLS]   = Agg.groupby('Top').apply(lambda By: (
        np.round(By[COLS].mean(), 3)
    )).reset_index(drop=1)

    return Top




def GET_SOURCE(toggle, TICKERS, TF, SRC, UNTIL, FULL_CLOCK, start=None):
    if not toggle:  return None
    Src  = GET_HIST_DATA(TICKERS, SRC, UNTIL, start)
    Keys = MAKE_PAIRS_KEYS(Src)
    Src2 = MAKE_PAIRS_SOURCE(Keys, Src, TF, FULL_CLOCK)
    
    del Src, Keys;  gc.collect()
    return Src2

def COMPUTE_DATA(toggle, Src2, GRAN, PRDS, SLGS):
    if not toggle:  return None
    PSts    = CALCULATE_PRC_STATS(Src2, GRAN, PRDS, SLGS)
    Vars    = CALCULATE_VARIATIONS(Src2, GRAN, PRDS, SLGS)
    VSts    = CALCULATE_VAR_STATS(Src2, Vars, GRAN, PRDS, SLGS)
    LS      = LONG_SHORT_RETURNS(Src2, Vars, PRDS, SLGS)
    IAbs    = CALCULATE_ABSOLUTES(Vars, VSts)
    GVar    = CALCULATE_GAP_VARIATION(Src2, IAbs, GRAN, PRDS, SLGS)
    Frames  = COMBINE_FRAMES(Src2, LS, PSts, Vars, VSts, IAbs, GVar)
    
    del PSts, Vars, VSts, LS, IAbs, GVar;  gc.collect()

    Rnk_Rtn, Rnk_Cor, Rnk_Inds = CALCULATE_RANKS(Frames, PRDS, SLGS)
    Ranks   = COMBINE_RANKS(Src2, Rnk_Rtn, Rnk_Cor, Rnk_Inds)
    
    del Src2, Rnk_Rtn, Rnk_Cor, Rnk_Inds;  gc.collect()
    
    Data    = COMBINE_DATAS(Frames, Ranks)
    
    del Frames, Ranks;  gc.collect()

    Scrn    = SORT_SCREENER(Data)

    del Data;  gc.collect()

    # del PSts, Vars, VSts, LS, IAbs, GVar, Frames, Rnk_Rtn, Rnk_Cor, Rnk_Inds, Ranks, Data;  gc.collect()
    return Scrn

def RANKED_FOLIOS(Scrn, TF, TOP, GRAN, ANUM,  RCOMB, IND_1,IND_2,RNK_1,RNK_2,  WGHTS, NORM_WGHTS):
    Slc         = SELECT(Scrn, GRAN)

    del Scrn;  gc.collect()

    RnkComb     = RANK_COMBINATION(Slc,  ['N','Ticker.XY'],['Date'],  RNK_1, RNK_2,  WGHTS, NORM_WGHTS)
    InTop       = TOP_CLASSIFCATION(RnkComb, RankBy=RCOMB, TOP=TOP)
    Class_RComb = COMBINE_CLASS_RCOMB(InTop, RnkComb)

    del InTop, RnkComb;  gc.collect()

    DFrame      = INSERT_CLASS_RCOMB_IN_DFRAME(Slc, Class_RComb)

    del Class_RComb;  gc.collect()

    Agg         = MAKE_AGGREGATION_KEYS(Slc, TOP)

    del Slc;  gc.collect()

    Agg         = ADD_WALLET_PERFORMANCE(Agg, DFrame, TF, ANUM, COLS=[IND_1, IND_2, RNK_1, RNK_2])

    del DFrame;  gc.collect()

    Top         = FINAL_RESULTS(Agg, TOP, COLS=[IND_1, IND_2, RNK_1, RNK_2])

    # del Slc, RnkComb, InTop, Class_RComb, DFrame;  gc.collect()
    return Top, Agg


