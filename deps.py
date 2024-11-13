import warnings
from typing import Literal as Lit
import os
import requests as req
import sqlite3 as sqlite
import itertools as it
import functools as ft
import datetime as dt
from numpy import nan
import numpy as np
import scipy as sci
import pandas as pd
import yfinance as yf



# ============================================ #
# ================ DTYPE CAST ================ #
# ============================================ #
def pd_str_cast(Data, left='$', middle=',', right='%',   
    numeric:Lit['skip','to','apply']='skip', num_errors:Lit['ignore','coerce','raise']='coerce', 
    mult=None, div=None, rnd=None, apply:Lit['skip','int','round']='skip', 
    err_rtn:Lit['void','entry','none','nan','na','0','1','alt']='na', alt=pd.NA, catch=True, logErr=False, 
    as_type='', as_errors:Lit['ignore','raise']='ignore', 
):

    # ======================== HELPERS ======================== #
    def Lambda(x):
        try:
            if apply == 'int':      return   int(x)
            if apply == 'round':    return round(x)

        except Exception as Error:
            if not catch:   raise Error
            if logErr:      print(Error)

            if err_rtn == 'entry':  return x
            if err_rtn == 'none':   return None
            if err_rtn == 'nan':    return np.nan
            if err_rtn == 'na':     return pd.NA
            if err_rtn == '0':      return 0
            if err_rtn == '1':      return 1
            if err_rtn == 'alt':    return alt

    Pipe = Data
    
    # ======================== REPLACE ======================== #
    if left:                Pipe = Pipe.str.replace(left,   '')
    if middle:              Pipe = Pipe.str.replace(middle, '')
    if right:               Pipe = Pipe.str.replace(right,  '')

    # ======================== CAST NUMERIC ======================== #
    if numeric == 'to':     Pipe = pd.to_numeric(Pipe,                     errors=num_errors)
    if numeric == 'apply':  Pipe =               Pipe.apply(pd.to_numeric, errors=num_errors)

    # ======================== FIGURES AND TYPE ======================== #
    if div   !=  None :     Pipe = Pipe / div
    if mult  !=  None :     Pipe = Pipe * mult
    if rnd   !=  None :     Pipe = Pipe.round(rnd)
    if apply != 'skip':     Pipe = Pipe.apply(Lambda)

    # ======================== SET TYPE ======================== #
    if as_type:             Pipe =               Pipe.astype(as_type,      errors= as_errors)

    return Pipe


# =========================================== #
# ================ OPERATORS ================ #
# =========================================== #
def coalesce(x, y):
    return x if (x != None) else y

def replaces(data, befores, after):
    return ft.reduce(lambda pipe, before: pipe.replace(before, after), befores, data)


# ======================================= #
# ================ CALCS ================ #
# ======================================= #
def exp(x, base=np.e, _in:Lit['p1','p100','100']='', _out:Lit['m1','m100','100']='', 
    fx:Lit['exp','pwr']='exp', sci=False,  toggle=True, skip=False, fig=None, 
    err_rtn:Lit['void','none','0','1','nan','na','','[]','array','series','alt']='nan', alt=..., catch=True, logErr=False, 
    flt_warn=True, *args, **kwargs
):
    if not toggle or skip:
        return x
    
    if flt_warn:  
        warnings.filterwarnings('ignore', 'overflow encountered')

    try:
        np_log = np.lib.scimath.log if sci else np.log
        _log   = lambda x:         np_log(x,       *args,**kwargs)
        _exp   = lambda x:         np.exp(x,       *args,**kwargs)
        _pwr   = lambda x, base: np.power(base, x, *args,**kwargs)

        if fx == 'exp':     func = lambda x, base: _exp(x * _log(base))
        if fx == 'pwr':     func = lambda x, base: _pwr(x,       base )


        pipe = x

        if _in == 'p1':     pipe = (1+pipe    )
        if _in == 'p100':   pipe = (1+pipe/100)
        if _in == '100':    pipe = (  pipe/100)

        pipe = func(x, base)

        if _out == 'm1':    pipe = (pipe-1)
        if _out == 'm100':  pipe = (pipe-1)*100
        if _out == '100':   pipe = (pipe  )*100

        if fig != None:     pipe = pipe.round(fig)
    
        return pipe

    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)

        if err_rtn == 'entry':  return x
        if err_rtn == 'none':   return None
        if err_rtn == '0':      return 0
        if err_rtn == '1':      return 1
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '':       return ""
        if err_rtn == '[]':     return []
        if err_rtn == 'array':  return np.array([])
        if err_rtn == 'series': return pd.Series([])
        if err_rtn == 'alt':    return alt

def log(x, base=np.e, _in:Lit['p1','p100','100']='', _out:Lit['m1','m100','100']='', 
    sci=False, toggle=True, skip=False, fig=None, 
    err_rtn:Lit['void','none','0','1','nan','na','','[]','array','series','alt']='nan', alt=..., catch=True, logErr=False, 
    flt_warn=True, *args, **kwargs
):
    if not toggle or skip:
        return x

    if flt_warn:  
        warnings.filterwarnings('ignore', 'invalid value encountered')
        warnings.filterwarnings('ignore', 'divide by zero encountered')

    try:
        np_log = np.lib.scimath.log if sci else np.log
        _log   = lambda x: np_log(x, *args, **kwargs)
        pipe   = x

        if _in == 'p1':     pipe = (1+pipe    )
        if _in == 'p100':   pipe = (1+pipe/100)
        if _in == '100':    pipe = (  pipe/100)

        pipe = _log(x) / _log(base)

        if _out == 'm1':    pipe = (pipe-1)
        if _out == 'm100':  pipe = (pipe-1)*100
        if _out == '100':   pipe = (pipe  )*100

        if fig != None:     pipe = pipe.round(fig)
    
        return pipe

    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)

        if err_rtn == 'entry':  return x
        if err_rtn == 'none':   return None
        if err_rtn == '0':      return 0
        if err_rtn == '1':      return 1
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '':       return ""
        if err_rtn == '[]':     return []
        if err_rtn == 'array':  return np.array([])
        if err_rtn == 'series': return pd.Series([])
        if err_rtn == 'alt':    return alt

def pct_point(Df, Y, X, geo:Lit['X','Y','Z']=[], pct:Lit['X','Y','Z']=[], mlt=None, rnd=None, _as=''):
    
    has_df = isinstance(Df, pd.DataFrame)

    if has_df:
        Y, X = Df[Y], Df[X]

    if 'Y' in pct:  Y = (1+Y/100)
    if 'X' in pct:  X = (1+X/100)

    if 'Y' in geo:  Y = log(Y)
    if 'X' in geo:  X = log(X)

    Z = (Y - X)

    if 'Z' in geo:  Z = exp(Z)
    if 'Z' in pct:  Z = (Z-1)*100

    if mlt != None: Z = Z * mlt
    if rnd != None: Z = Z.round(rnd)
    if _as:         Z = Z.astype(_as)

    return Z


def np_calc(algo:Lit['len','sum','prod','mean','median','max','std','min',  'geo_mean','geo_median','geo_std'], X, skip_nan=False, 
    _in:Lit['p1','p100','100']='', Log=0, geo=0, Exp=0, _out:Lit['m1','m100','100']='', 
    base=np.e, pwr=None, mlt=None, rnd=None, 
    err_rtn:Lit['void','none','0','1','nan','na','','alt']='nan', alt=..., catch=True, logErr=False,  *a,**b,
):
    try:
        # ======================== Step 1 ======================== #
        if 'geo' in algo:   geo = True

        pipe = X

        if _in == 'p1':     pipe = (1+pipe    )
        if _in == 'p100':   pipe = (1+pipe/100)
        if _in == '100':    pipe = (  pipe/100)

        if Log or geo:      pipe = log(pipe, base, catch=catch, logErr=logErr)


        # ======================== Step 2 ======================== #
        _sum    = np.nansum    if skip_nan else np.sum
        _prod   = np.nanprod   if skip_nan else np.prod
        _mean   = np.nanmean   if skip_nan else np.mean
        _median = np.nanmedian if skip_nan else np.median
        _max    = np.nanmax    if skip_nan else np.max
        _std    = np.nanstd    if skip_nan else np.std
        _min    = np.nanmin    if skip_nan else np.min

        if algo == 'len':       pipe =  len(pipe)
        if algo == 'sum':       pipe = _sum(pipe    *a,**b)
        if algo == 'prod':      pipe = _prod(pipe   *a,**b)
        if algo == 'mean':      pipe = _mean(pipe   *a,**b)
        if algo == 'median':    pipe = _median(pipe *a,**b)
        if algo == 'max':       pipe = _max(pipe    *a,**b)
        if algo == 'std':       pipe = _std(pipe    *a,**b)
        if algo == 'min':       pipe = _min(pipe    *a,**b)


        # ======================== Step 3 ======================== #
        if Exp or geo:      pipe = exp(pipe, base, catch=catch, logErr=logErr)
        if pwr != None:     pipe = pipe ** pwr

        if _out == 'm1':    pipe = (pipe-1)
        if _out == 'm100':  pipe = (pipe-1)*100
        if _out == '100':   pipe = (pipe  )*100

        if mlt != None:     pipe = pipe * mlt
        if rnd != None:     pipe = np.round(pipe, rnd)

        return pipe


        # ======================== Error ======================== #
    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)

        if err_rtn == 'none':   return None
        if err_rtn == '0':      return 0
        if err_rtn == '1':      return 1
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '':       return ""
        if err_rtn == 'alt':    return alt

def pd_calc(algo:Lit['len','sum','prod','mean','median','max','std','min'], X,  
    _in:Lit['p1','p100','100']='', Log=0, geo=0, Exp=0, _out:Lit['m1','m100','100']='', 
    base=np.e, mlt=None, pwr=None, rnd=None, 
    err_rtn:Lit['void','none','0','1','nan','na','','alt']='nan', alt=..., catch=True, logErr=False, *a,**b,
):
    try:
        # ======================== Step 1 ======================== #
        is_series = isinstance(X, pd.Series|pd.DataFrame)
        if (    is_series):  pipe = X
        if (not is_series):  pipe = pd.Series(X)

        if _in == 'p1':     pipe = (1+pipe    )
        if _in == 'p100':   pipe = (1+pipe/100)
        if _in == '100':    pipe = (  pipe/100)

        if Log or geo:      pipe = log(pipe, base, catch=catch, logErr=logErr)

        # ======================== Step 2 ======================== #
        if 'count'  in algo:    pipe = pipe.count( *a,**b)
        if 'sum'    in algo:    pipe = pipe.sum(   *a,**b)
        if 'prod'   in algo:    pipe = pipe.prod(  *a,**b)
        if 'mean'   in algo:    pipe = pipe.mean(  *a,**b)
        if 'median' in algo:    pipe = pipe.median(*a,**b)
        if 'max'    in algo:    pipe = pipe.max(   *a,**b)
        if 'std'    in algo:    pipe = pipe.std(   *a,**b)
        if 'min'    in algo:    pipe = pipe.min(   *a,**b)

        # ======================== Step 3 ======================== #
        if Exp or geo:      pipe = exp(pipe, base, catch=catch, logErr=logErr)
        if pwr != None:     pipe = pipe ** pwr

        if _out == 'm1':    pipe = (pipe-1)
        if _out == 'm100':  pipe = (pipe-1)*100
        if _out == '100':   pipe = (pipe  )*100

        if mlt != None:     pipe = pipe * mlt
        if rnd != None:     pipe = pipe.round(rnd)

        return pipe

        # ======================== Error ======================== #
    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)

        if err_rtn == 'none':   return None
        if err_rtn == '0':      return 0
        if err_rtn == '1':      return 1
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '':       return ""
        if err_rtn == 'alt':    return alt

def pd_roll_by(Src, By, win, minWin, cb, Small=None,  stp=(0),  cast_wins:Lit['int','rnd']='', 
    dummy_key=None,  gsort=0, gkeys=0, gdrop=0, 
    logErr=1, throw=0,  to:Lit['list','numpy','series']='list', 
):
    # ================================================ DEPS ================================================ #
    def is_float(x):  return isinstance(x, float)


    def step(Data, stp):
        if isinstance(stp, int|float) and (stp < 0):
                return Data[::stp]
        else:   return Data


    def pd_groupby(Df, By, iter_rows=False, with_dummy=False, dummy_key=None,      
        gsort=0, gkeys=0, gdrop=0, *a,**b
    ):
        if By:          return              Df.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop,  *a,**b)
        if iter_rows:   return              Df.iterrows()
        if with_dummy:  return [(dummy_key, Df)]
        else:           return              Df


    def pd_window(Df, win=None, minWin=..., iter_rows=True, iter_idx=False, 
        logErr=1, throw=0, ERR_MISS_WIN_PARAMS='MISSING_BOTH_WIN_PARAMS', *a,**b
    ):
        # ================ Methods ================ #  
        if win and minWin:  return Df.rolling(win, minWin,  *a,**b)
        if minWin:          return Df.expanding(   minWin,  *a,**b)
        
        # ================ Error ================ #  
        if throw:           raise  Exception(ERR_MISS_WIN_PARAMS)
        if logErr:                     print(ERR_MISS_WIN_PARAMS)          

        # ================ Default ================ #
        if (not iter_idx and iter_rows):    return Df.iterrows()[1]
        if (    iter_idx and iter_rows):    return Df.iterrows()
        else:                               return Df


    def Empty(K, Key, R, Roll):
        if callable(Small): return Small(K, Key, R, Roll)
        else:               return Small


    # ================================================ MAIN ================================================ #
    has_minWin = isinstance(minWin, int|float) and (0 < minWin)

    if win in ['inf','exp']:   win = 100_000_000
    if minWin == 'eq':      minWin = win
    if minWin == 'half':    minWin = win/2

    if (cast_wins == 'int') and is_float(win):      win    =   int(win)
    if (cast_wins == 'int') and is_float(minWin):   minWin =   int(minWin)

    if (cast_wins == 'rnd') and is_float(win):      win    = round(win)
    if (cast_wins == 'rnd') and is_float(minWin):   minWin = round(minWin)


    Df = step(Src, stp)

    pipe = []
    # ======================== LOOPING GROUP ======================== #
    for K, (Key, Sec) in enumerate(pd_groupby(Df, By,  iter_rows=False, with_dummy=True, dummy_key=dummy_key,  gsort=gsort, gkeys=gkeys, gdrop=gdrop)):
        # ======================== LOOPING WINDOW ======================== #
        for R, Roll in enumerate(pd_window(Sec,  win, minWin,  iter_rows=True, iter_idx=False,  logErr=logErr, throw=throw)):
            
            if has_minWin and (minWin <= len(Roll)):
                pipe.append(cb(K, Key, R, Roll))
            else:   
                pipe.append(Empty(K, Key, R, Roll))
        pass
    pass

    pipe = step(pipe, stp)
    if to == 'numpy':   pipe = np.array(pipe)    
    if to == 'series':  pipe = pd.Series(pipe)
    return pipe

def pd_roll_calc(algo:Lit['count','sum','prod','mean','max','median','min','std', 'geo_mean','geo_median','geo_std'], 
    Df, Val, By='', win=0, minWin=0, stp=0, cast_wins:Lit['int','rnd']='', 
    _in:Lit['p1','p100','100']='', Log=0, geo=0, Exp=0, _out:Lit['m1','m100','100']='', 
    mult=None, pwr=None, fig=None, _as='',  
    err_rtn:Lit['entry','none','nan','na','alt','void']='nan', alt=..., catch=True, logErr=True, 
    gsort=0, gkeys=0, gdrop=0,
):
    try: 
        def is_float(x):  return isinstance(x, float)
        
        if minWin == 'eq':      minWin = win
        if minWin == 'half':    minWin = win/2

        if (cast_wins == 'int') and is_float(win):      win    =   int(win)
        if (cast_wins == 'int') and is_float(minWin):   minWin =   int(minWin)

        if (cast_wins == 'rnd') and is_float(win):      win    = round(win)
        if (cast_wins == 'rnd') and is_float(minWin):   minWin = round(minWin)

        if win in ['exp', 'inf', float('inf')]:  win = 100_000_000


        if 'geo' in algo:   geo = True

        copy = (_in) or (Log) or (geo)
        Pipe = Df.copy() if copy else Df

        if _in == 'p1':         Pipe[Val] =   1+Pipe[Val]
        if _in == 'p100':       Pipe[Val] =   1+Pipe[Val]/100
        if _in == '100':        Pipe[Val] =     Pipe[Val]/100
        if (Log | geo):         Pipe[Val] = log(Pipe[Val])

        if stp:                 Pipe = Pipe[::stp]
        if By:                  Pipe = Pipe.groupby(By, sort=gsort, group_keys=gkeys, dropna=gdrop)

        Pipe = Pipe[Val]

        if   (win and minWin):  Pipe = Pipe.rolling(win, minWin)
        elif (win           ):  Pipe = Pipe.rolling(win)
        elif (        minWin):  Pipe = Pipe.expanding(minWin)
        else:                   Pipe = Pipe.expanding()

        if   'count'  in algo:  Pipe = Pipe.count()
        elif 'sum'    in algo:  Pipe = Pipe.sum()
        elif 'prod'   in algo:  Pipe = Pipe.prod()
        elif 'mean'   in algo:  Pipe = Pipe.mean()
        elif 'median' in algo:  Pipe = Pipe.median()
        elif 'max'    in algo:  Pipe = Pipe.max()
        elif 'std'    in algo:  Pipe = Pipe.std()
        elif 'min'    in algo:  Pipe = Pipe.min()

        if stp:                 Pipe = Pipe[::stp]
        Pipe = Pipe.reset_index(drop=1)

        if (Exp | geo):         Pipe = exp(Pipe)
        if pwr  != None:        Pipe = Pipe ** pwr

        if _out == 'm1':        Pipe =    (Pipe-1)
        if _out == 'm100':      Pipe =    (Pipe-1)*100
        if _out == '100':       Pipe =    (Pipe  )*100

        if mult != None:        Pipe = Pipe * mult
        if fig  != None:        Pipe = Pipe.round(fig)
        if _as:                 Pipe = Pipe.astype(_as)
        return Pipe


    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)
            
        if err_rtn == 'entry':  return Df[Val]
        if err_rtn == 'none':   return None
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == 'alt':    return alt



# ======================================== #
# ================ FILTER ================ #
# ======================================== #
def np_filter(X, C, eq=1):
    if not isinstance(X, np.ndarray|pd.Series|pd.DataFrame):    X = np.array(X) 
    if not isinstance(C, np.ndarray|pd.Series|pd.DataFrame):    C = np.array(C) 
    return X[C==eq]

def np_cross_filter(X, Y, toggle=True, skip=False, flt_warn=True):
    
    # ================ Logs ================ #
    if flt_warn:  warnings.filterwarnings('ignore', 'invalid value encountered')

    # ================ Helpers ================ #
    def is_fin(x):      return isinstance(x, int|float) and (float('-inf') < x < float('+inf'))
    def np_is_fin(x):   return np.vectorize(is_fin)(x)
    def is_arr(x):      return isinstance(x, np.ndarray|pd.Series)

    # ================ Skip ================ #
    if not toggle or skip:  
        return X, Y

    # ================ Init ================ #
    if not is_arr(X):   X = np.array(X)
    if not is_arr(Y):   Y = np.array(Y)

    # ================ Main ================ #
    X_check  = np_is_fin(X)
    Y_check  = np_is_fin(Y)
    XY_check = np.multiply(X_check, Y_check)

    X_flt, Y_flt = X[XY_check], Y[XY_check]

    return X_flt, Y_flt

def TotalNorm(X, norm=True, K=None, fig=None):
    X_total = np.sum(X)
    X_norm  = X / X_total

    result = X

    if norm:                result = X_norm
    if (K is not None):     result = np.multiply(result, K)
    if (fig is not None):   result = np.round(result, fig)

    return result

def pd_between(Df, Val, By='', Ref='N', win=1, _min=0, cb=None, current=True,  
    to:Lit['series','series.T','S.values','S.values.T','dframe','dframe.T','df.values','df.values.T']='S.values',
    if_small:Lit['raise','return']='return', small_rtn:Lit['none','nan','na','nat','0','1','','[]','np','pd','alt']='nan', small_alt=..., 
    ERR_SMALL='NO_MIN_WINDOW', 
):
    By      = np.ravel(By)
    C       = current
    is_exp  = win in ['exp', 'inf', np.inf]
    is_roll = not is_exp
    _Vals_  = np.ravel(Val)
    
    def Lambda(row):
        if By:  BY = (Df[By] == row[By]).prod(axis=1)
        else:   BY = 1

        if is_roll and (current):  WIN = (row[Ref]-win <  Df[Ref]) * (Df[Ref] <= row[Ref])
        if is_roll and (not C  ):  WIN = (row[Ref]-win <= Df[Ref]) * (Df[Ref] <  row[Ref])
        if is_exp  and (current):  WIN =                             (Df[Ref] <= row[Ref])
        if is_exp  and (not C  ):  WIN =                             (Df[Ref] <  row[Ref])

        Flt = Df[BY * WIN == 1]
        is_small_win = (len(Flt) < _min)

        if is_small_win:
            if if_small == 'raise':   raise Exception(ERR_SMALL)
            if if_small == 'return':  return { 'none':None, 'nan':np.nan, 'na':pd.NA, 'nat':pd.NaT, '0':0, '1':1, '':"", '[]':[], 'np':np.array([]), 'pd':pd.Series([]), 'alt':small_alt }[small_rtn]

        if   to == 'series':        Pipe = Flt[Val]
        elif to == 'series.T':      Pipe = Flt[Val].T
        elif to == 'S.values':      Pipe = Flt[Val].values
        elif to == 'S.values.T':    Pipe = Flt[Val].values.T
        elif to == 'dframe':        Pipe = Flt[_Vals_]
        elif to == 'dframe.T':      Pipe = Flt[_Vals_].T
        elif to == 'df.values':     Pipe = Flt[_Vals_].values
        elif to == 'df.values.T':   Pipe = Flt[_Vals_].values.T
        if cb:                      Pipe = cb(Pipe)
        return Pipe

    return Df.apply(Lambda, axis=1)




# ======================================= #
# ================ STATS ================ #
# ======================================= #
def ZLevel(Df, Value, Mean, Deviat, Pct=[0,0,0], Log=[0,0,0], fig=None, 
    err_rtn:Lit['void','none','nan','na','','alt']='nan', alt=..., catch=True, logErr=True,  
):
    try:
        # ================ Check ================ #
        has_df  = isinstance(Df, pd.DataFrame)
        has_fig = fig != None


        # ================ Values ================ #
        if has_df:
            val, avg, dev = Df[Value], Df[Mean], Df[Deviat]
        else:
            val, avg, dev = Value, Mean, Deviat


        # ================ Adjusts ================ #
        if Pct[0]:  val = (1+val/100)
        if Pct[1]:  avg = (1+avg/100)
        if Pct[2]:  dev = (1+dev/100)

        if Log[0]:  val = log(val)
        if Log[1]:  avg = log(avg)
        if Log[2]:  dev = log(dev)


        # ================ Formula ================ #
        Calc = (val - avg) / dev
        

        # ================ Return ================ #
        if has_fig:  return np.round(Calc, fig)
        else:        return Calc


        # ================ Error ================ #
    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)

        if err_rtn == 'none':   return None
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '':       return ""
        if err_rtn == 'alt':    return alt

def correlate(lib:Lit['numpy','scipy','pandas'],  Df, Y, X, 
    # numpy
    rowvar:bool=..., 
    # scipy
    alternative:Lit['two-sided','less','greater']='two-sided',
    # pandas
    method:Lit['pearson','kendall','spearman']='pearson', min_periods=None,  

    _mlt=None, _rnd=..., _int=None, _as='',  
    err_rtn:Lit['none','nan','na','0','+1','-1','','alt']='nan', alt=..., catch=True, logErr=True, 
):
    try: 
        has_df = isinstance(Df, pd.DataFrame)

        if has_df:
            Y = Df[Y]
            X = Df[X]

        if lib == 'pandas':
              Y = pd.Series(Y)
              X = pd.Series(X)


        if lib == 'numpy':
            Z = np.corrcoef(X, Y,  rowvar=rowvar)[0, 1]

        if lib == 'scipy':  
            Z = sci.stats.pearsonr(x=X, y=Y,  alternative=alternative)[0]
        
        if lib == 'pandas': 
            Z = Y.corr(X,  method=method, min_periods=min_periods)


        if (_mlt != None):  Z =      (Z * _mlt)
        if (_rnd != ... ):  Z = round(Z,  _rnd)
        if (_int):          Z =   int(Z)
        if (_as):           Z =       Z.astype(_as)

        return Z
    

    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print('def:correlate', Error)

        if err_rtn == 'none':   return None
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '0':      return 0
        if err_rtn == '+1':     return +1
        if err_rtn == '-1':     return -1
        if err_rtn == '':       return ""
        if err_rtn == 'alt':    return alt

def pd_rollby_correlate(lib:Lit['numpy','scipy','pandas'],  Df, Y, X,  By, win, minWin,  
    stp=(0), cast_wins:Lit['int','rnd']='', Small=None, 
    # numpy
    rowvar:bool=..., 
    # scipy
    alternative:Lit['two-sided','less','greater']='two-sided',
    # pandas
    method:Lit['pearson','kendall','spearman']='pearson', min_periods=None,  

    _mlt=None, _rnd=..., _int=None, _as='',  
    err_rtn:Lit['none','nan','na','0','+1','-1','','alt']='nan', alt=..., throw=True, catch=True, logErr=True,
    dummy_key=None,  gsort=0, gkeys=0, gdrop=0, 
):
    
    return pd_roll_by(Df, By, win, minWin, lambda K, Key, R, Roll: (
        correlate(lib, Roll, Y, X, rowvar, alternative, method, min_periods, _mlt, _rnd, _int, _as, err_rtn, alt, catch, logErr)
    ), Small, stp, cast_wins, dummy_key, gsort, gkeys, gdrop, logErr, throw, to='series')


# ============================================ #
# ================ DATAFRAMES ================ #
# ============================================ #
class pd_DataFrame_fromRows():
    def __init__(my, cols, data):
        my.cols = cols
        my.data = data

    def append(my, row):
        my.data.append(row)

    def instance(my, *args, **kwargs):
        return pd.DataFrame(my.data, columns=my.cols, *args, **kwargs)

def pd_reduce_rows(Keys, Lambda, cols, data, verbose=True, *args, **kwargs):
    
    Df = pd_DataFrame_fromRows(cols, data)
    L  = len(Keys)

    for i, key in enumerate(Keys):
        if verbose:  print('Start,', f'iLoop: {i}/{L},', 'Key:',key)
        Df.append(Lambda(i, key))
        if verbose:  print('Finish,', f'iLoop: {i}/{L},', 'Key:',key)
    pass
    return Df.instance(*args, **kwargs)


class pd_DataFrame_fromSecs():
    def __init__(my, Sections):
        my.Sections = Sections

    def append(my, Sec):
        my.Sections.append(Sec)

    def concat(my, axis=0, *args, **kwargs):
        return pd.concat(my.Sections, axis=axis, *args, **kwargs)
    
    def concat_reset(my, axis=0, drop=1, *args, **kwargs):
        return my.concat(axis=axis).reset_index(drop=drop, *args, **kwargs)

def pd_reduce_Secs(Keys, Lambda, Initial, axis=0, drop=1, verbose=True):
    
    Df = pd_DataFrame_fromSecs(Initial)
    L  = len(Keys)

    for i, key in enumerate(Keys):
        if verbose:  print('Start,', f'iLoop: {i}/{L},', 'Key:',key)
        Df.append(Lambda(i, key))
        if verbose:  print('Finish,', f'iLoop: {i}/{L},', 'Key:',key)
    pass
    return Df.concat_reset(axis, drop)


def pd_columns(Df, A=None, Z=None, to:Lit['numpy','list']='numpy'): 
    cols = Df.loc[:, A:Z].columns
    if (to == 'list'):   return cols.to_list()
    if (to == 'numpy'):  return cols.to_numpy()

def pd_col_insert(Src, Label, values, on:Lit['init','before','after','final'], Ref='', 
    if_exists:Lit['skip','assign','duplicate','raise','delete']='raise', copy=False, 
    ERR_ALREADY_EXISTS='LABEL_ALREADY_EXISTS'
):

    # ======================== INIT VARS ======================== #
    Df   = Src.copy() if copy else Src
    iloc = ...
    allow_duplicates = ...

    # ======================== GUARD ======================== #
    if (Label in Df) and (if_exists == 'skip'):
        return Df

    if (Label in Df) and (if_exists == 'assign'):
        Df[Label] = values
        return Df

    if (Label in Df) and (if_exists == 'duplicate'):
        allow_duplicates == True
    
    if (Label in Df) and (if_exists == 'raise'):
        raise Exception(ERR_ALREADY_EXISTS)

    if (Label in Df) and (if_exists == 'delete'):
        del Df[Label]


    # ======================== DEFINE POSITION ======================== #
    if on == 'init':    iloc = 0
    if on == 'before':  iloc = Df.columns.get_loc(Ref)
    if on == 'after':   iloc = Df.columns.get_loc(Ref) +1
    if on == 'final':   iloc = len(list(Df))


    # ======================== INSERT ======================== #
    Df.insert(iloc, Label, values, allow_duplicates=allow_duplicates)
    return Df

def pd_insert_df(Src, on:Lit['before','after'], Ref, Add):

    iRef = Src.columns.get_loc(Ref)

    adj = { 'before':0, 'after':1 }[on]

    Left, Right = Src.iloc[:, None:iRef+adj], Src.iloc[:, iRef+adj:None]

    return pd.concat([Left, Add, Right], axis=1)


def pd_book(Data, head=5, tail=0, page=1,  reset=False, drop=True,  set_option=True, width=None, max_columns=None,  *args, **kwargs):
    
    if set_option:
        pd.set_option('display.width',       width)
        pd.set_option('display.max_columns', max_columns)


    Df = pd.DataFrame(index=range(len(Data)))

    Df['head+']  = Df.index[::+1] +1
    Df['head-']  = Df.index[::-1] +1
    Df['tail+']  = Df.index[::-1] +1
    Df['tail-']  = Df.index[::+1] +1

    Df['H+']     = (Df['head+']-1) //abs(head) +1
    Df['H-']     = (Df['head-']-1) //abs(head) +1
    Df['T+']     = (Df['tail+']-1) //abs(tail) +1
    Df['T-']     = (Df['tail-']-1) //abs(tail) +1

    Df['HP++']   = (Df['H+'] == abs(page))
    Df['HP+-']   = (Df['H-'] == abs(page))
    Df['HP-+']   = (Df['H-'] == abs(page))
    Df['HP--']   = (Df['H+'] == abs(page))

    Df['TP++']   = (Df['T+'] == abs(page))
    Df['TP+-']   = (Df['T-'] == abs(page))
    Df['TP-+']   = (Df['T-'] == abs(page))
    Df['TP--']   = (Df['T+'] == abs(page))

    Df['Filter'] = True

    if (head > 0) and (page > 0):   Df['Filter'] = Df['HP++']
    if (head > 0) and (page < 0):   Df['Filter'] = Df['HP+-']
    if (head < 0) and (page > 0):   Df['Filter'] = Df['HP-+']
    if (head < 0) and (page < 0):   Df['Filter'] = Df['HP--']

    if (tail > 0) and (page > 0):   Df['Filter'] = Df['TP++']
    if (tail > 0) and (page < 0):   Df['Filter'] = Df['TP+-']
    if (tail < 0) and (page > 0):   Df['Filter'] = Df['TP-+']
    if (tail < 0) and (page < 0):   Df['Filter'] = Df['TP--']

    Pipe = Data[Df['Filter']]
    Pipe = Pipe.reset_index(drop=drop, *args, **kwargs) if reset else Pipe
    return Pipe



# ============================================== #
# ================ DF RELATIONS ================ #
# ============================================== #
def pd_relate(L, X, R, Y, Z, kind='left', *args, **kwargs):
    
    X, Y, Z = np.ravel(X), np.ravel(Y), np.ravel(Z)
    
    LEFT  = L[X].add_prefix('L.')
    RIGHT = R[Y].add_prefix('R.')
    RIGHT = RIGHT.assign(**R[Z])

    return pd.merge(LEFT, RIGHT, kind,  left_on=[f'L.{x}' for x in X],  right_on=[f'R.{y}' for y in Y],  *args, **kwargs)[Z] 


# =============================================== #
# ================ CLASSFICATION ================ #
# =============================================== #
def np_remove_prefix(X, remove=['Rank ','High ','Low '], after=''):
    
    for before in remove:
        X = np.char.replace(X, before, after)
    pass
    return X

def Nth(Df, Ref, By,  sign, asc, adj,  
    method:Lit['average','min','max','first','dense']='dense', _as:Lit['int','Int64']='int',  
    gsort=0, gkeys=0, gdrop=0,  *a,**b
):
    Pipe = Df
    if By:      Pipe = Pipe.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop)
    Pipe = Pipe[Ref].rank(ascending=asc, method=method,  *a,**b)
    if sign:    Pipe = sign * Pipe 
    if adj:     Pipe = Pipe + adj
    if _as:     Pipe = Pipe.astype(_as)
    return Pipe

def HRnk(Df, Ref, By='', asc=0, 
    method:Lit['average','min','max','first','dense']='dense', _as:Lit['int','Int64']='int',  
    prefix='', suffix='',  gsort=0, gkeys=0, gdrop=0,  *a,**b
):
    Pipe = Df
    if By:      Pipe = Pipe.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop)
    Pipe = Pipe[Ref].rank(ascending=asc, method=method,  *a,**b)
    if _as:     Pipe = Pipe.astype(_as)

    if prefix:  Pipe = Pipe.add_prefix(prefix)
    if suffix:  Pipe = Pipe.add_suffix(suffix)
    return Pipe

def LRnk(Df, Ref, By='', asc=1, 
    method:Lit['average','min','max','first','dense']='dense', _as:Lit['int','Int64']='int',  
    prefix='', suffix='',  gsort=0, gkeys=0, gdrop=0,  *a,**b
):
    Pipe = Df
    if By:      Pipe = Pipe.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop)
    Pipe = Pipe[Ref].rank(ascending=asc, method=method,  *a,**b)
    if _as:     Pipe = Pipe.astype(_as)
    
    if prefix:  Pipe = Pipe.add_prefix(prefix)
    if suffix:  Pipe = Pipe.add_suffix(suffix)
    return Pipe

def pd_banded_HLRanks(HRnk, LRnk, _slice=['High Rank','Low Rank']):
    H, L = HRnk.loc[:, _slice[0]:None], LRnk.loc[:, _slice[1]:None]
    HLRanks     = pd.concat([H, L], axis=1)
    BANDED_COLS = np.hstack(list(zip(H.columns, L.columns)))
    return HLRanks[BANDED_COLS]

def RANK(mode:Lit['BANDED','HIGH_LOW'], Src, feats, Keys=[], By=[],  dsc=['High ',''], asc=['Low ',''],  with_keys=True, with_by=True, with_feats=False,  
    method:Lit['average','min','max','first','dense']='dense', pct=False, as_Int64=False, 
    gsort=0, gkeys=0, gdrop=0,  flt_warn=True,  *a,**b, 
):
    # ================ Helpers ================ #
    def _unique(X):
        return pd.Series(X).unique()

    # ================ Warnings ================ #
    if flt_warn: 
        warnings.filterwarnings('ignore', 'DataFrame is highly fragmented')

    # ================ Main ================ #
    Keys  = np.ravel(Keys)
    By    = np.ravel(By)
    A, Z  = feats
    Feats = Src.loc[:, A:Z].columns.values

    Cols = _unique([*Keys, *By, *Feats])
    Slc  = Src[Cols].copy()
    Grp  = Slc.groupby(By.tolist(),  sort=gsort, group_keys=gkeys, dropna=gdrop) if By else Slc

    ASC = Grp[Feats].rank(method=method, ascending=1, pct=pct,  *a,**b)
    DSC = Grp[Feats].rank(method=method, ascending=0, pct=pct,  *a,**b)

    if as_Int64:
        ASC = ASC.round(0).astype('Int64')
        DSC = DSC.round(0).astype('Int64')

    # ================ HL ================ #
    HLcols = _unique([
        *(Keys   if with_keys   else []),  
        *(By     if with_by     else []),
        *(Feats  if with_feats  else []), 
    ])

    HIGH = Slc[HLcols].copy()
    HIGH[dsc[0]+'Rank'+dsc[1]]   = ''
    if dsc[0]:      HIGH = HIGH.assign(**DSC.add_prefix(dsc[0]))
    if dsc[1]:      HIGH = HIGH.assign(**DSC.add_suffix(dsc[1]))

    LOW = Slc[HLcols].copy()
    LOW[asc[0]+'Rank'+asc[1]]   = ''
    if asc[0]:      LOW = LOW.assign(**ASC.add_prefix(asc[0]))
    if asc[1]:      LOW = LOW.assign(**ASC.add_suffix(asc[1]))

    if mode == 'HIGH_LOW':  return HIGH, LOW
    
    # ================ Banded ================ #
    HL = Slc[HLcols].copy()
    HL = HL.assign(**pd_banded_HLRanks(HIGH, LOW))
    
    if mode == 'BANDED':  return HL 

def pd_isin_top(Src, Rank, Keys=[], TOP=[3,5,10], _as:Lit['bool','num']='num',  with_RnkVal=0, 
    Label=(lambda T: f'InTop {T}')
):
    # ================ Helpes ================ #
    def cast(x):
        if _as == 'bool':   return x
        if _as == 'num':    return x * 1
    
    # ================ Vars ================ #
    Keys = np.ravel(Keys)

    # ================ Main ================ #
    Df = pd.DataFrame(Src[Keys])
    
    if with_RnkVal:  
        Df['Rnk Val'] = Src[Rank]

    Df['Isin Top'] = ''
    for T in TOP:
        Df[Label(T)] = cast(Src[Rank] <= T)
    pass
    return Df


# ==================================================== #
# ================ LINEAR COMBINATION ================ #
# ==================================================== #
def LinearCombinator(X, W, algo:Lit['sum_mult','sum_pwr','prod_pwr']='sum_mult', 
    filt=False, norm=True, emptyRtn=np.nan, fig=None
):
    X, W = np_cross_filter(X, W, toggle=filt)
    W    = TotalNorm(   W, norm=norm)

    if not len(X): 
        return emptyRtn

    result = np.nan
    if (algo == 'sum_mult'):    result = np.sum( np.multiply(X, W))
    if (algo == 'sum_pwr' ):    result = np.sum( np.power(   X, W))
    if (algo == 'prod_pwr'):    result = np.prod(np.power(   X, W))
    if (fig != None):           result = np.round(result, fig)
    
    return result



# ====================================================== #
# ======================== CRUD ======================== #
# ====================================================== #
def google_colab_drive_connect(base='/content/drive', path='/My Drive/Colab Notebooks/'):
    from google.colab import drive;  drive.mount(base)
    import sys;  sys.path.append(base + path)

def google_colab_drive_reconnect(base='/content/drive', force_remount=True,  toggle=True, skip=False):
    if toggle and not skip:
        from google.colab import drive
        drive.flush_and_unmount()
        drive.mount(base, force_remount=force_remount)

def environment(env:Lit['local','google_colab','kaggle']='local', base='', path='', force_remount=True,  toggle=True, skip=False):
    if (env == 'google_colab'): 
        if (not base):  base = '/content/drive'
        if (not path):  path = '/My Drive/Colab Notebooks/'
        google_colab_drive_reconnect(base, force_remount,  toggle, skip)

    if (env == 'kaggle'): 
        if (not base):  base = '/kaggle/input'
        if (not path):  path = '/uploads/'

    return base + path


def pd_check_exists(left, Dir, File, Tab, fmt:Lit['excel','sqlite','parquet'],  con=None, con_close=False, 
    to:Lit['bool','tuple','dict']='bool',  
    ext_excel='xlsx', ext_sqlite='db', ext_parquet='parquet',  
    QUERY=lambda Tab: f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tab}'", 
    catch=True, logErr=True
):
    # ================ Response ================ #
    def Return(has_file, has_tab):
        if to == 'bool':    return has_tab
        if to == 'tuple':   return has_file, has_tab
        if to == 'dict':    return { 'has_file':has_file, 'has_tab':has_tab }

    try:
        # ================ Case Excel ================ #
        if (fmt == 'excel'):
            file  =  pd.ExcelFile(f'{left}{Dir}{File}.{ext_excel}')
            check = (Tab in file.sheet_names)

        # ================ Case SQLite ================ #
        if (fmt == 'sqlite'): 
            if not con:     con = sqlite.connect(f'{left}{Dir}{File}.{ext_sqlite}')
            res   =  pd.read_sql_query(QUERY(Tab), con)
            check = (not res.empty)
            if con_close:   con.close()

        # ================ Case Parquet ================ #
        if (fmt == 'parquet'): 
            if not  os.path.exists(f'{left}{Dir}{File}/'):   raise Exception('DIR_NOT_EXISTS')
            check = os.path.exists(f'{left}{Dir}{File}/{Tab}.{ext_parquet}')

        return Return(has_file=True, has_tab=check)

    # ================ Error ================ #
    except Exception as Error:
        if not catch:   raise Error
        if logErr:      print(Error)
        return Return(has_file=False, has_tab=None)

def pd_sync_storage(mode:Lit['READ','SAVE','LOAD'],  Df, Dir, File, Tab,  fmt:Lit['excel','sqlite','parquet'], 
    index=False, toggle=True, skip=False, LIMIT=None, 
    if_exists:Lit['OVERWRITE','RAISE']='OVERWRITE',  if_not_exists:Lit['READ','SAVE','RAISE']='RAISE', 
    env:Lit['local','google_colab','kaggle']='local', base='', path='', force_remount=True,  re_toggle=True, re_skip=False,
    ext_excel='xlsx', ext_sqlite='db', ext_parquet='parquet',  
    ERR_EXISTS='FILE_ALREADY_EXISTS', ERR_NOT_EXISTS='FILE_NOT_EXISTS',  *a,**b, 
):
    # ======================== TOGGLE ======================== # 
    if (not toggle) or (skip):  return Df


    # ======================== VARS ======================== # 
    LEFT = environment(env, base, path, force_remount,  re_toggle, re_skip)


    # ======================== HELPERS ======================== # 
    def pd_read():
        if (fmt == 'excel'):
            return pd.read_excel(f'{LEFT}{Dir}{File}.{ext_excel}', Tab,  *a,**b)

        if (fmt == 'sqlite'):
            con = sqlite.connect(f'{LEFT}{Dir}{File}.{ext_sqlite}')
            if LIMIT:   query = f'SELECT * FROM [{Tab}] LIMIT {LIMIT}'
            else:       query = f'SELECT * FROM [{Tab}]'
            return pd.read_sql(query, con,  *a,**b)

        if (fmt == 'parquet'):
            return pd.read_parquet(f'{LEFT}{Dir}{File}/{Tab}.{ext_parquet}',  *a,**b)

    def pd_write():
        if (fmt == 'excel'):
            with pd.ExcelWriter(f'{LEFT}{Dir}{File}.{ext_excel}') as Writer:
                Df.to_excel(Writer, Tab, index=index,  *a,**b)

        if (fmt == 'sqlite'): 
            con = sqlite.connect(f'{LEFT}{Dir}{File}.{ext_sqlite}')
            Df.to_sql(Tab, con, if_exists='replace', index=index,  *a,**b)

        if (fmt == 'parquet'):
            Df.to_parquet(f'{LEFT}{Dir}{File}/{Tab}.{ext_parquet}', index=index,  *a,**b)


    # ======================== EMPTY INPUT ======================== # 
    has_df = isinstance(Df, pd.DataFrame)
    if not has_df:
        return pd_read()


    # ======================== COMMANDS ======================== # 
    exists = pd_check_exists(LEFT, Dir, File, Tab, fmt=fmt)

    if mode == 'READ':                              return Df

    if mode == 'SAVE':
        if not exists:                  pd_write(); return Df
        elif if_exists == 'RAISE':                  raise  Exception(ERR_EXISTS)
        elif if_exists == 'OVERWRITE':  pd_write(); return Df

    if mode == 'LOAD':
        if exists:                                  return pd_read()
        elif if_not_exists == 'RAISE':              raise  Exception(ERR_NOT_EXISTS)
        elif if_not_exists == 'READ':               return Df    
        elif if_not_exists == 'SAVE':   pd_write(); return Df    

class pd_Storage():

    # ================ Initial ================ #
    def join(x, left, Dir, File, sfx, ext):
        return f'{left}{Dir}{File}{sfx}.{ext}'

    def __init__(x, Dir='', File='', sfx='', ext='parquet', 
        env:Lit['local','google_colab','kaggle']='local', base='', path='', force_remount=True,  re_toggle=True, re_skip=False, 
    ):
        left      = environment(env, base, path, force_remount,  re_toggle, re_skip)
        x.address = x.join(left, Dir, File, sfx, ext)


    # ================ Funcs ================ #
    def empty(x, *a,**b):
        return pd.DataFrame(*a,**b)

    def save(x, Df, *a,**b):
        Df.to_parquet(x.address, *a,**b)

    def load(x, *a,**b):
        return pd.read_parquet(x.address, *a,**b)


    # ================ Methods ================ #
    def create_new_empty(x, *a,**b):
        Df = x.empty(*a,**b)
        x.save(Df)

    def add_column(x, Col, Vals):
        Df      = x.load()
        Df[Col] = Vals
        x.save(Df)












# =========================================================================================== #
# ======================================== PROVIDERS ======================================== #
# =========================================================================================== #

# ======================== INDICE ======================== #
def get_SNP_500_components(url='https://www.slickcharts.com/sp500', headers=None, rtn_df:Lit['Raw','Custom']='Custom'):
    
    emulator = lambda: ({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36', 
        'X-Requested-With': 'XMLHttpRequest'
    })

    res = req.get(url, headers=coalesce(headers, emulator()))
    Raw = pd.read_html(res.content)[0]

    if rtn_df == 'Raw': 
        return Raw

    if rtn_df == 'Custom':
        Df              = pd.DataFrame()
        Df['N']         = Raw['#']
        Df['Ticker']    = Raw['Symbol']
        Df['Asset']     = Raw['Company']
        Df['Comp']      = Raw['Portfolio%'].str.rstrip('%').astype('float')
        Df['Price']     = Raw['Price']
        Df['Chg']       = Raw['Chg']
        Df['Var']       = Raw['% Chg'].apply(lambda x: replaces(x, ['(','%',')'], '')).astype('float')
        return Df


# ======================== CRYPTOS ======================== #
def cg_crypto_screener():
        
    # ======================================== STEP 1: SCRAP ======================================== #
    def SCRAP(URL='https://www.coingecko.com'):
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36', 
            'X-Requested-With': 'XMLHttpRequest'
        }
        res   = req.get(URL, headers=headers)
        Page  = pd.read_html(res.content)
        Table = Page[0]
        Frame = Table.loc[1:None, '#':None].reset_index(drop=1)
        Raw   = Frame.drop('Unnamed: 3', axis=1)

        Raw['#']            = Raw['#'].astype(int)
        Raw['Price']        = pd_str_cast(Raw['Price'],         as_type=float)

        Raw['1h']           = pd_str_cast(Raw['1h'],            as_type=float)
        Raw['24h']          = pd_str_cast(Raw['24h'],           as_type=float)
        Raw['7d']           = pd_str_cast(Raw['7d'],            as_type=float)
        Raw['30d']          = pd_str_cast(Raw['30d'],           as_type=float)

        Raw['24h Volume']   = pd_str_cast(Raw['24h Volume'],    numeric='to',   as_type='Int64')
        Raw['Market Cap']   = pd_str_cast(Raw['Market Cap'],    numeric='to',   as_type='Int64')
        Raw['FDV']          = pd_str_cast(Raw['FDV'],           numeric='to',   as_type='Int64')

        return Raw

    Raw = SCRAP()
    
    # ======================================== STEP 2: FORMAT ======================================== #
    def FORMAT(Raw):
        Src                 =  pd.DataFrame()
        Src['N']            =  Raw['#']
        Src['Ticker']       =  Raw['Coin'].str.split(' ').str[-1]
        Src['Asset']        =  Raw['Coin'].str.split(' ').apply(lambda x: ' '.join(x[:-1]))
        Src['Price']        =  Raw['Price']

        Src['Liq']          = ''
        Src['Capital']      =  Raw['Market Cap']
        Src['Cap']          = (Raw['Market Cap'] /1000**2).round(3)
        Src['Volume 1D']    =  Raw['24h Volume']
        Src['Vol 1D']       = (Raw['24h Volume'] /1000**2).round(3)

        Src['Vars']         =  ''
        Src['Var 1H']       =  Raw['1h']
        Src['Var 1D']       =  Raw['24h']
        Src['Var 1W']       =  Raw['7d']
        Src['Var 1M']       =  Raw['30d']
        return Src

    Src = FORMAT(Raw)

    return Src

def cs_crypto_screener():

    # ======================== STEP 1: SCRAP ======================== #
    def SCRAP(url='https://cryptoslate.com/coins/'):
        res  = req.get(url)
        Page = pd.read_html(res.content)
        Src  = Page[0]
        return Src

    Raw = SCRAP()


    # ======================== STEP 2: FORMAT ======================== #
    def FORMAT(Raw):
        Src = pd.DataFrame()
        Src['N']        = Raw['#'] 
        Src['Ticker']   = Raw['Name'].str.split(' ').str[-1]
        Src['Asset']    = Raw['Name'].str.split(' ').apply(lambda x: ' '.join(x[:-1]))
        Src['Price']    = pd_str_cast(Raw['Price'],       as_type=float)

        Src['Liq']      = ''
        Src['Capital']  = pd_str_cast(Raw['Market Cap'],  as_type='Int64')
        Src['Cap']      = pd_str_cast(Raw['Market Cap'],  as_type='Int64').div(1000**2).round(3)
        Src['Volume 1D']= pd_str_cast(Raw['24H Vol'],     as_type='Int64')
        Src['Vol 1D']   = pd_str_cast(Raw['24H Vol'],     as_type='Int64').div(1000**2).round(3)

        Src['Vars']     = ''
        Src['Var 1D']   = pd_str_cast(Raw['24H %'],       as_type=float)
        Src['Var 1W']   = pd_str_cast(Raw['7D %'],        as_type=float)
        Src['Var 1M']   = pd_str_cast(Raw['30D %'],       as_type=float)
        return Src

    Src = FORMAT(Raw)

    return Src


# ======================== YAHOO ======================== #
YF_TF = Lit['mo','wk','d']

def yf_url(TICKER, TF:YF_TF, start=1, end=10_000_000_000): 
    return (
        f'https://query1.finance.yahoo.com/v7/finance/download/{TICKER}?' + 
        f'&events=history' +
        f'&interval=1{TF}' + 
        f'&period1={start}' + 
        f'&period2={end}' + 
        f'&includeAdjustedClose=true'
    )

def yf_web_hdata(TICKER,  TF:YF_TF, start=1, end=10_000_000_000, offset={ 'n':0 },  tries=1, catch=True, logErr=False):

    Df = pd.DataFrame({ 'N':[], 'Ticker':[], 'Date':[], 'Open':[], 'High':[], 'Low':[], 'Close':[], 'Adj Close':[], 'Volume':[] })

    while tries:
        try:
            Csv = pd.read_csv(yf_url(TICKER, TF, start, end))
            Src = Csv.sort_values('Date', ascending=0).reset_index(drop=1)
            Df['N']      = -Src.index
            Df['Ticker'] = TICKER
            Df['Date']   = Src['Date'].astype('datetime64[ns]') + pd.DateOffset(**offset)
            Df = Df.assign(**Src.iloc[:, 1:])
            return Df

        except Exception as Error:
            if not catch: raise Error
            if logErr: print('Tries:',tries, 'Ticker:',TICKER, 'Error:',Error)
            tries -= 1

        if not tries:
            return Df

def yf_web_hdatas(TICKERS,  TF:YF_TF, start=1, end=10_000_000_000, offset={ 'n':0 },  tries=2, catch=True, logErr=False, verbose=False):
    return pd_reduce_Secs(np.ravel(TICKERS), lambda i, Ticker: (
        yf_web_hdata(Ticker, TF, start, end, offset, tries, catch, logErr)
    ), Initial=[], verbose=verbose)

def yf_api_hdata(
    TICKER      = '^SPX',
    group_by    : Lit['ticker', 'column'] = 'ticker', 
    ColDt       : Lit['Date', 'Datetime'] = 'Date', 
    interval    : Lit['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'] = '1mo', 
    period      : Lit['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'] = '5y', 
    end         = None, # dt.datetime(2025, 12, 31) 
    start       = None, # dt.datetime(2020,  1,  1) 
    offset      = { 'n':0 },
    prepost     = False,
    ignore_tz   = True, 
    tries       = 1, 
    catch       = True, 
    logErr      = False,
    flt_warn    = True,  
):
    if flt_warn:  warnings.filterwarnings('ignore', "'T' is deprecated")

    Df = pd.DataFrame({ 
        'N':         [], 'Ticker': [], ColDt: [], 
        'Open':      [], 'High':   [], 'Low': [], 'Close': [], 
        'Adj Close': [], 'Volume': [], 
    })

    while tries:
        try: 
            Raw = yf.download(tickers=TICKER, start=start, end=end, ignore_tz=ignore_tz, group_by=group_by, period=period, interval=interval, prepost=prepost)

            Src = Raw.reset_index()
            Src = Src.sort_values(Src.columns[0], ascending=False).reset_index(drop=1)

            Df['N']      = -Src.index
            Df['Ticker'] =  TICKER
            Df[ColDt]    =  Src.iloc[:, 0] + pd.DateOffset(**offset)
            Df           =  Df.assign(**Src.iloc[:, 1:])
            return Df

        except Exception as Error:
            if not catch: raise Error
            if logErr: print('Tries:',tries, 'Ticker:',TICKER, 'Error:',Error)
            tries -= 1

        if not tries:
            return Df

def yf_api_hdatas(
    TICKERS     = ['^SPX'],
    group_by    : Lit['ticker', 'column'] = 'ticker', 
    ColDt       : Lit['Date', 'Datetime'] = 'Date', 
    interval    : Lit['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'] = '1mo', 
    period      : Lit['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'] = '5y', 
    end         = None, # dt.datetime(2025, 12, 31) 
    start       = None, # dt.datetime(2020,  1,  1) 
    offset      = { 'n':0 },
    prepost     = False,
    ignore_tz   = True, 
    tries       = 2, 
    catch       = True, 
    logErr      = False, 
    verbose     = False,  
):
    return pd_reduce_Secs(np.ravel(TICKERS), lambda i, Ticker: (
        yf_api_hdata(Ticker, group_by, ColDt, interval, period, end, start, offset, prepost, ignore_tz, tries, catch, logErr)
    ), Initial=[], verbose=verbose)











# ========================================================================================= #
# ======================================== FINANCE ======================================== #
# ========================================================================================= #
ETFS    = ['SPY','DIA','QQQ','XLF','XLC','XLE','XLU','XLP','XLK','XLV','XLI','XLY','XLB','XOP','XBI','XRT','XHB','XME','XLRE','IWM','IYR','NOBL','AMLP','OIH','KRE','VNQ','ITB','KBE','IBB','SMH']
TECH_1  = ['MSFT']
TECH_3  = ['MSFT','AAPL','GOOG']
TECH_5  = ['MSFT','AAPL','GOOG','META','AMZN']
TECH_7  = ['MSFT','AAPL','GOOG','AMZN','META','NVDA','NFLX']
BEST_20 = ['AAPL','MSFT','GOOGL','META','UNH','V','MA','INTU','CPRT','IDXX','ODFL','SHW','EW','ANET','RMD','WST','MPWR','POOL','EPAM','MKTX']
DJI_30  = ['UNH','MSFT','GS','HD','CAT','CRM','MCD','AMGN','V','TRV','AXP','BA','HON','IBM','JPM','AAPL','AMZN','JNJ','PG','CVX','MRK','DIS','NKE','MMM','KO','WMT','DOW','CSCO','INTC','VZ']
NDX_100 = ['MSFT','AAPL','AMZN','META','AVGO','GOOGL','GOOG','COST','TSLA','NFLX','AMD','PEP','QCOM','TMUS','ADBE','LIN','CSCO','AMAT','TXN','AMGN','INTU','CMCSA','ISRG','MU','HON','BKNG','INTC','LRCX','VRTX','NVDA','ADI','REGN','KLAC','ADP','PANW','PDD','SBUX','MDLZ','ASML','SNPS','MELI','GILD','CDNS','CRWD','PYPL','NXPI','CTAS','MAR','ABNB','CSX','CEG','ROP','MRVL','ORLY','MRNA','PCAR','MNST','CPRT','MCHP','ROST','KDP','AZN','AEP','ADSK','FTNT','WDAY','DXCM','PAYX','DASH','TTD','KHC','IDXX','CHTR','LULU','VRSK','ODFL','EA','FAST','EXC','GEHC','CCEP','FANG','DDOG','CTSH','BIIB','BKR','CSGP','ON','XEL','CDW','ANSS','TTWO','ZS','GFS','TEAM','DLTR','WBD','ILMN','MDB','WBA','SIRI']
SNP_500 = ['MSFT','AAPL','NVDA','AMZN','META','GOOGL','GOOG','BRK.B','LLY','AVGO','JPM','XOM','TSLA','UNH','V','PG','MA','JNJ','COST','MRK','HD','ABBV','WMT','NFLX','CVX','CRM','BAC','AMD','PEP','KO','TMO','QCOM','ADBE','WFC','LIN','ORCL','ACN','CSCO','MCD','INTU','DIS','AMAT','ABT','GE','TXN','CAT','DHR','VZ','AMGN','PFE','IBM','NOW','PM','NEE','CMCSA','GS','UNP','SPGI','ISRG','RTX','MU','COP','ETN','AXP','HON','BKNG','UBER','ELV','INTC','LRCX','LOW','T','MS','C','PGR','ADI','VRTX','TJX','SYK','NKE','BLK','BSX','MDT','SCHW','CB','REGN','KLAC','ADP','MMC','UPS','LMT','CI','BA','DE','PANW','PLD','MDLZ','FI','SNPS','SBUX','BX','AMT','TMUS','CMG','BMY','SO','GILD','APH','MO','CDNS','ZTS','DUK','ICE','CL','WM','CME','TT','ANET','TDG','FCX','MCK','EOG','EQIX','SHW','CEG','NXPI','CVS','PH','GD','BDX','TGT','CSX','PYPL','SLB','NOC','ITW','MPC','EMR','MCO','HCA','USB','ABNB','PSX','PNC','MSI','CTAS','ECL','APD','ROP','ORLY','FDX','MAR','PCAR','AON','WELL','VLO','MMM','AIG','MRNA','AJG','CARR','MCHP','EW','COF','NSC','TFC','GM','HLT','JCI','WMB','DXCM','TRV','AZO','SRE','NEM','F','SPG','AEP','OKE','CPRT','TEL','ADSK','DLR','AFL','FIS','URI','ROST','KMB','BK','A','MET','GEV','D','AMP','PSA','O','HUM','CCI','ALL','IDXX','SMCI','DHI','PRU','LHX','NUE','GWW','IQV','HES','CNC','OXY','PAYX','PWR','DOW','AME','OTIS','STZ','GIS','PCG','CTVA','MNST','FTNT','MSCI','CMI','IR','YUM','LEN','RSG','ACGL','FAST','KVUE','KMI','EXC','PEG','COR','SYY','VRSK','MPWR','MLM','KDP','CSGP','KR','IT','RCL','LULU','XYL','FANG','CTSH','VMC','DD','GEHC','FICO','DAL','EA','ED','ADM','VST','HWM','HAL','MTD','BKR','BIIB','RMD','CDW','DVN','PPG','ON','DFS','ODFL','DG','TSCO','WAB','HIG','HSY','EXR','ROK','XEL','VICI','EL','EFX','ANSS','KHC','EIX','HPQ','GLW','EBAY','AVB','GPN','FSLR','FTV','CHTR','TROW','CBRE','CHD','WTW','DOV','WEC','TRGP','KEYS','FITB','GRMN','AWK','MTB','LYB','WST','ZBH','IFF','TTWO','DLTR','PHM','WDC','BR','NTAP','CAH','NVR','HPE','NDAQ','RJF','DECK','ETR','IRM','DTE','STT','STE','APTV','EQR','WY','VLTO','PTC','BALL','HUBB','TER','PPL','BRO','BLDR','TYL','GPC','LDOS','SBAC','CTRA','STLD','FE','ES','WAT','INVH','MOH','AXON','CPAY','HBAN','CBOE','VTR','TDY','COO','OMC','CNP','AEE','ARE','ULTA','CINF','AVY','NRG','MKC','STX','ALGN','PFG','CMS','DRI','SYF','RF','DPZ','J','HOLX','TSN','BAX','TXT','NTRS','WBD','UAL','EXPD','ATO','EG','ILMN','LH','ZBRA','FDS','ESS','LVS','EQT','CFG','CLX','IEX','K','PKG','WRB','ENPH','LUV','DGX','MAA','IP','VRSN','JBL','MAS','CE','MRO','CF','CCL','BG','EXPE','SWKS','CAG','ALB','SNA','AMCR','AKAM','TRMB','POOL','GEN','RVTY','AES','L','PNR','BBY','DOC','WRK','KEY','LYV','SWK','JBHT','NDSN','HST','ROL','TECH','VTRS','LNT','LW','KIM','EVRG','JKHY','IPG','PODD','UDR','EMN','WBA','LKQ','NI','SJM','CPT','CRL','JNPR','BBWI','KMX','UHS','EPAM','INCY','ALLE','MGM','AOS','MOS','FFIV','HII','HRL','TAP','CTLT','NWSA','CHRW','REG','TFX','TPR','QRVO','HSIC','DAY','APA','WYNN','AAL','CPB','GNRC','AIZ','PNW','PAYC','BXP','BF.B','SOLV','BWA','MKTX','FOXA','MTCH','HAS','FMC','ETSY','FRT','DVA','RHI','IVZ','CZR','GL','RL','CMA','BEN','NCLH','BIO','MHK','PARA','FOX','NWS']


TP_ETFS    = Lit['SPY','DIA','QQQ','XLF','XLC','XLE','XLU','XLP','XLK','XLV','XLI','XLY','XLB','XOP','XBI','XRT','XHB','XME','XLRE','IWM','IYR','NOBL','AMLP','OIH','KRE','VNQ','ITB','KBE','IBB','SMH']
TP_TECH_1  = Lit['MSFT']
TP_TECH_3  = Lit['MSFT','AAPL','GOOG']
TP_TECH_5  = Lit['MSFT','AAPL','GOOG','META','AMZN']
TP_TECH_7  = Lit['MSFT','AAPL','GOOG','AMZN','META','NVDA','NFLX']
TP_BEST_20 = Lit['AAPL','MSFT','GOOGL','META','UNH','V','MA','INTU','CPRT','IDXX','ODFL','SHW','EW','ANET','RMD','WST','MPWR','POOL','EPAM','MKTX']
TP_DJI_30  = Lit['UNH','MSFT','GS','HD','CAT','CRM','MCD','AMGN','V','TRV','AXP','BA','HON','IBM','JPM','AAPL','AMZN','JNJ','PG','CVX','MRK','DIS','NKE','MMM','KO','WMT','DOW','CSCO','INTC','VZ']
TP_NDX_100 = Lit['MSFT','AAPL','AMZN','META','AVGO','GOOGL','GOOG','COST','TSLA','NFLX','AMD','PEP','QCOM','TMUS','ADBE','LIN','CSCO','AMAT','TXN','AMGN','INTU','CMCSA','ISRG','MU','HON','BKNG','INTC','LRCX','VRTX','NVDA','ADI','REGN','KLAC','ADP','PANW','PDD','SBUX','MDLZ','ASML','SNPS','MELI','GILD','CDNS','CRWD','PYPL','NXPI','CTAS','MAR','ABNB','CSX','CEG','ROP','MRVL','ORLY','MRNA','PCAR','MNST','CPRT','MCHP','ROST','KDP','AZN','AEP','ADSK','FTNT','WDAY','DXCM','PAYX','DASH','TTD','KHC','IDXX','CHTR','LULU','VRSK','ODFL','EA','FAST','EXC','GEHC','CCEP','FANG','DDOG','CTSH','BIIB','BKR','CSGP','ON','XEL','CDW','ANSS','TTWO','ZS','GFS','TEAM','DLTR','WBD','ILMN','MDB','WBA','SIRI']
TP_SNP_500 = Lit['MSFT','AAPL','NVDA','AMZN','META','GOOGL','GOOG','BRK.B','LLY','AVGO','JPM','XOM','TSLA','UNH','V','PG','MA','JNJ','COST','MRK','HD','ABBV','WMT','NFLX','CVX','CRM','BAC','AMD','PEP','KO','TMO','QCOM','ADBE','WFC','LIN','ORCL','ACN','CSCO','MCD','INTU','DIS','AMAT','ABT','GE','TXN','CAT','DHR','VZ','AMGN','PFE','IBM','NOW','PM','NEE','CMCSA','GS','UNP','SPGI','ISRG','RTX','MU','COP','ETN','AXP','HON','BKNG','UBER','ELV','INTC','LRCX','LOW','T','MS','C','PGR','ADI','VRTX','TJX','SYK','NKE','BLK','BSX','MDT','SCHW','CB','REGN','KLAC','ADP','MMC','UPS','LMT','CI','BA','DE','PANW','PLD','MDLZ','FI','SNPS','SBUX','BX','AMT','TMUS','CMG','BMY','SO','GILD','APH','MO','CDNS','ZTS','DUK','ICE','CL','WM','CME','TT','ANET','TDG','FCX','MCK','EOG','EQIX','SHW','CEG','NXPI','CVS','PH','GD','BDX','TGT','CSX','PYPL','SLB','NOC','ITW','MPC','EMR','MCO','HCA','USB','ABNB','PSX','PNC','MSI','CTAS','ECL','APD','ROP','ORLY','FDX','MAR','PCAR','AON','WELL','VLO','MMM','AIG','MRNA','AJG','CARR','MCHP','EW','COF','NSC','TFC','GM','HLT','JCI','WMB','DXCM','TRV','AZO','SRE','NEM','F','SPG','AEP','OKE','CPRT','TEL','ADSK','DLR','AFL','FIS','URI','ROST','KMB','BK','A','MET','GEV','D','AMP','PSA','O','HUM','CCI','ALL','IDXX','SMCI','DHI','PRU','LHX','NUE','GWW','IQV','HES','CNC','OXY','PAYX','PWR','DOW','AME','OTIS','STZ','GIS','PCG','CTVA','MNST','FTNT','MSCI','CMI','IR','YUM','LEN','RSG','ACGL','FAST','KVUE','KMI','EXC','PEG','COR','SYY','VRSK','MPWR','MLM','KDP','CSGP','KR','IT','RCL','LULU','XYL','FANG','CTSH','VMC','DD','GEHC','FICO','DAL','EA','ED','ADM','VST','HWM','HAL','MTD','BKR','BIIB','RMD','CDW','DVN','PPG','ON','DFS','ODFL','DG','TSCO','WAB','HIG','HSY','EXR','ROK','XEL','VICI','EL','EFX','ANSS','KHC','EIX','HPQ','GLW','EBAY','AVB','GPN','FSLR','FTV','CHTR','TROW','CBRE','CHD','WTW','DOV','WEC','TRGP','KEYS','FITB','GRMN','AWK','MTB','LYB','WST','ZBH','IFF','TTWO','DLTR','PHM','WDC','BR','NTAP','CAH','NVR','HPE','NDAQ','RJF','DECK','ETR','IRM','DTE','STT','STE','APTV','EQR','WY','VLTO','PTC','BALL','HUBB','TER','PPL','BRO','BLDR','TYL','GPC','LDOS','SBAC','CTRA','STLD','FE','ES','WAT','INVH','MOH','AXON','CPAY','HBAN','CBOE','VTR','TDY','COO','OMC','CNP','AEE','ARE','ULTA','CINF','AVY','NRG','MKC','STX','ALGN','PFG','CMS','DRI','SYF','RF','DPZ','J','HOLX','TSN','BAX','TXT','NTRS','WBD','UAL','EXPD','ATO','EG','ILMN','LH','ZBRA','FDS','ESS','LVS','EQT','CFG','CLX','IEX','K','PKG','WRB','ENPH','LUV','DGX','MAA','IP','VRSN','JBL','MAS','CE','MRO','CF','CCL','BG','EXPE','SWKS','CAG','ALB','SNA','AMCR','AKAM','TRMB','POOL','GEN','RVTY','AES','L','PNR','BBY','DOC','WRK','KEY','LYV','SWK','JBHT','NDSN','HST','ROL','TECH','VTRS','LNT','LW','KIM','EVRG','JKHY','IPG','PODD','UDR','EMN','WBA','LKQ','NI','SJM','CPT','CRL','JNPR','BBWI','KMX','UHS','EPAM','INCY','ALLE','MGM','AOS','MOS','FFIV','HII','HRL','TAP','CTLT','NWSA','CHRW','REG','TFX','TPR','QRVO','HSIC','DAY','APA','WYNN','AAL','CPB','GNRC','AIZ','PNW','PAYC','BXP','BF.B','SOLV','BWA','MKTX','FOXA','MTCH','HAS','FMC','ETSY','FRT','DVA','RHI','IVZ','CZR','GL','RL','CMA','BEN','NCLH','BIO','MHK','PARA','FOX','NWS']

TP_CRYPTOS      = Lit['BTC','ETH','USDT','BNB','SOL','STETH','USDC','XRP','DOGE','TON','ADA','SHIB','AVAX','WBTC','TRX','LINK','BCH','DOT','UNI','NEAR','MATIC','LTC','WEETH','LEO','DAI','ICP','PEPE','FET','ETC','KAS','EZETH','APT','USDE','RNDR','XMR','STX','FIL','HBAR','FDUSD','ATOM','MNT','XLM','IMX','CRO','INJ','OKB','ARB','WIF','FLOKI','SUI','GRT','TAO','OP','AR','VET','MKR','RETH','THETA','FTM','JASMY','NOT','BONK','RUNE','ONDO','TIA','METH','BRETT','LDO','BGB','CORE','PYTH','STRK','EOS','WBT','SEI','ALGO','AAVE','GALA','JUP','QNT','ENA','ORDI','FLOW','FLR','CHEEL','CHZ','BEAM','GT','RSETH','DYDX','BSV','AXS','BTT','W','TKX','KCS','AKT','WLD','RON','EGLD']
LI_CRYPTOS      =           ['BTC','ETH','USDT','BNB','SOL','STETH','USDC','XRP','DOGE','TON','ADA','SHIB','AVAX','WBTC','TRX','LINK','BCH','DOT','UNI','NEAR','MATIC','LTC','WEETH','LEO','DAI','ICP','PEPE','FET','ETC','KAS','EZETH','APT','USDE','RNDR','XMR','STX','FIL','HBAR','FDUSD','ATOM','MNT','XLM','IMX','CRO','INJ','OKB','ARB','WIF','FLOKI','SUI','GRT','TAO','OP','AR','VET','MKR','RETH','THETA','FTM','JASMY','NOT','BONK','RUNE','ONDO','TIA','METH','BRETT','LDO','BGB','CORE','PYTH','STRK','EOS','WBT','SEI','ALGO','AAVE','GALA','JUP','QNT','ENA','ORDI','FLOW','FLR','CHEEL','CHZ','BEAM','GT','RSETH','DYDX','BSV','AXS','BTT','W','TKX','KCS','AKT','WLD','RON','EGLD']

TP_CRYPTOS_USD  = Lit['BTC-USD','ETH-USD','BNB-USD','SOL-USD','STETH-USD','XRP-USD','DOGE-USD','TON-USD','ADA-USD','SHIB-USD','AVAX-USD','WBTC-USD','TRX-USD','LINK-USD','BCH-USD','DOT-USD','UNI-USD','NEAR-USD','MATIC-USD','LTC-USD','WEETH-USD','LEO-USD','DAI-USD','ICP-USD','PEPE-USD','FET-USD','ETC-USD','KAS-USD','EZETH-USD','APT-USD','RNDR-USD','XMR-USD','STX-USD','FIL-USD','HBAR-USD','ATOM-USD','MNT-USD','XLM-USD','IMX-USD','CRO-USD','INJ-USD','OKB-USD','ARB-USD','WIF-USD','FLOKI-USD','SUI-USD','GRT-USD','TAO-USD','OP-USD','AR-USD','VET-USD','MKR-USD','RETH-USD','THETA-USD','FTM-USD','JASMY-USD','NOT-USD','BONK-USD','RUNE-USD','ONDO-USD','TIA-USD','METH-USD','BRETT-USD','LDO-USD','BGB-USD','CORE-USD','PYTH-USD','STRK-USD','EOS-USD','WBT-USD','SEI-USD','ALGO-USD','AAVE-USD','GALA-USD','JUP-USD','QNT-USD','ENA-USD','ORDI-USD','FLOW-USD','FLR-USD','CHEEL-USD','CHZ-USD','BEAM-USD','GT-USD','RSETH-USD','DYDX-USD','BSV-USD','AXS-USD','BTT-USD','W-USD','TKX-USD','KCS-USD','AKT-USD','WLD-USD','RON-USD','EGLD-USD']
LI_CRYPTOS_USD  =           ['BTC-USD','ETH-USD','BNB-USD','SOL-USD','STETH-USD','XRP-USD','DOGE-USD','TON-USD','ADA-USD','SHIB-USD','AVAX-USD','WBTC-USD','TRX-USD','LINK-USD','BCH-USD','DOT-USD','UNI-USD','NEAR-USD','MATIC-USD','LTC-USD','WEETH-USD','LEO-USD','DAI-USD','ICP-USD','PEPE-USD','FET-USD','ETC-USD','KAS-USD','EZETH-USD','APT-USD','RNDR-USD','XMR-USD','STX-USD','FIL-USD','HBAR-USD','ATOM-USD','MNT-USD','XLM-USD','IMX-USD','CRO-USD','INJ-USD','OKB-USD','ARB-USD','WIF-USD','FLOKI-USD','SUI-USD','GRT-USD','TAO-USD','OP-USD','AR-USD','VET-USD','MKR-USD','RETH-USD','THETA-USD','FTM-USD','JASMY-USD','NOT-USD','BONK-USD','RUNE-USD','ONDO-USD','TIA-USD','METH-USD','BRETT-USD','LDO-USD','BGB-USD','CORE-USD','PYTH-USD','STRK-USD','EOS-USD','WBT-USD','SEI-USD','ALGO-USD','AAVE-USD','GALA-USD','JUP-USD','QNT-USD','ENA-USD','ORDI-USD','FLOW-USD','FLR-USD','CHEEL-USD','CHZ-USD','BEAM-USD','GT-USD','RSETH-USD','DYDX-USD','BSV-USD','AXS-USD','BTT-USD','W-USD','TKX-USD','KCS-USD','AKT-USD','WLD-USD','RON-USD','EGLD-USD']

def pd_make_pairs(Keys, unique=None, prefix=''):

    if unique:  Keys = pd.Series(Keys).unique()

    Df = pd.DataFrame(it.combinations(Keys, 2), columns=['X','Y'])
    Df['XY'] = Df['X'] +'-'+ Df['Y']
    
    if prefix:  Df = Df.add_prefix(prefix)

    return Df


# ========================================= #
# ================ METRICS ================ #
# ========================================= #
def Prev(Df, Value, By='', N=1, stp=1,  gsort=0, gkeys=0, gdrop=0):
    Pipe = Df
    if (stp < 0):   Pipe = Pipe[::stp]
    if By:          Pipe = Pipe.groupby(By, sort=gsort, group_keys=gkeys, dropna=gdrop)
    return Pipe[Value].shift(N)

def Next(Df, Value, By='', N=1, stp=1,  gsort=0, gkeys=0, gdrop=0):
    Pipe = Df
    if (stp < 0):   Pipe = Pipe[::stp]
    if By:          Pipe = Pipe.groupby(By, sort=gsort, group_keys=gkeys, dropna=gdrop)
    return Pipe[Value].shift(-N)

def Return(Df, Value, By='', N=1, stp=1,  mult=None, rnd=None,  gsort=0, gkeys=0, gdrop=0):
    crnt  = Df[Value]
    next_ = Next(Df, Value, By, N, stp,  gsort, gkeys, gdrop)
    pipe  = next_ / crnt -1
    if mult != None:  pipe = pipe * mult
    if rnd  != None:  pipe = pipe.round(rnd)
    return pipe

def Variat(Df, Value, By='', N=1, stp=1,  mult=None, rnd=None,  gsort=0, gkeys=0, gdrop=0):
    crnt = Df[Value]
    prev = Prev(Df, Value, By, N, stp,  gsort, gkeys, gdrop)
    pipe = crnt / prev -1
    if mult != None:  pipe = pipe * mult
    if rnd  != None:  pipe = pipe.round(rnd)
    return pipe

def Variat_2nd(Df, Val, By='', N=1, stp=1,  geo=['X','Y','Z'], pct=['X','Y','Z'], 
    mlt=None, rnd=None,  gsort=0, gkeys=0, gdrop=0, 
):
    crnt = Df[Val]
    prev = Prev(Df, Val, By, N, stp,  gsort, gkeys, gdrop)

    pipe = pct_point(..., crnt, prev, geo=geo, pct=pct)

    if (mlt != None):   pipe = pipe * mlt
    if (rnd != None):   pipe = pipe.round(rnd)
    return pipe

def High(Df, Val, By='', win=..., minWin=1, stp=(+1),  gsort=0, gkeys=0, gdrop=0):
    Pipe = Df
    if (stp < 0):   Pipe = Pipe[::stp]
    if By:          Pipe = Pipe.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop)
    if win:         Pipe = Pipe.rolling(win, minWin)
    else:           Pipe = Pipe.expading(minWin)
    Pipe = Pipe[Val].max()
    Pipe = Pipe[::stp].reset_index(drop=1)
    return Pipe

def Low(Df, Val, By='', win=..., minWin=1, stp=(+1),  gsort=0, gkeys=0, gdrop=0):
    Pipe = Df
    if (stp < 0):   Pipe = Pipe[::stp]
    if By:          Pipe = Pipe.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop)
    if win:         Pipe = Pipe.rolling(win, minWin)
    else:           Pipe = Pipe.expading(minWin)
    Pipe = Pipe[Val].min()
    Pipe = Pipe[::stp].reset_index(drop=1)
    return Pipe


def Accum_Volume(Df, Val, By, win, minWin=1, stp=(+1),  mult=None, rnd=None, _as='',  
    gsort=0, gkeys=0, gdrop=0
):
    Pipe = Df
    if (stp < 0):       Pipe = Pipe[::stp]
    if By:              Pipe = Pipe.groupby(By, sort=gsort, group_keys=gkeys, dropna=gdrop)
    
    if win:             Pipe = Pipe.rolling(win, minWin)
    else:               Pipe = Pipe.expanding(minWin)
    Pipe = Pipe[Val].sum()

    if (stp < 0):       Pipe = Pipe[::stp]
    Pipe = Pipe.reset_index(drop=1)

    if (mult != None):  Pipe = Pipe * mult
    if (rnd  != None):  Pipe = Pipe.round(rnd)
    if _as:             Pipe = Pipe.astype(_as)
    return Pipe

def Evolut(Df, Value, By='', stp=(+1),  base=10, mult=10, add=(1), rnd=1, _as='',
    I=0, T='first',  gsort=0, gkeys=0, gdrop=0
):
    def extreme(S):
        try:    return S.dropna().iloc[I]
        except: return np.nan
    
    is_rev = (stp < 0)

    pipe = Df
    if is_rev:  pipe = pipe[::stp]
    if By:      pipe = pipe.groupby(By,  sort=gsort, group_keys=gkeys, dropna=gdrop)

    pipe = pipe[Value].transform(lambda S:  S / extreme(S))
    pipe = log(pipe, base)
    
    if (mult != None):  pipe = pipe * mult 
    if (add  != None):  pipe = pipe + add
    if (rnd  != None):  pipe = pipe.round(rnd)
    if _as:             pipe = pipe.astype(_as)
    return pipe












# ========================================================================================== #
# ======================================== BACKTEST ======================================== #
# ========================================================================================== #
TOP_30  = [5, 10, 15, 20, 25, 30]
TOP_100 = [5, 10, 25, 50, 100]
TOP_500 = [5, 10, 25, 50, 100, 250, 500]

TP_TF = Lit['Y','Q','M','W','D', 'Hf','Td','Sx','Tv','Tf']

MAP_SFX   = { 'Y':'Yr',  'Q':'Qr',  'M':'Mo',  'W':'Wk',  'D':'Da',    'Hf':'Hf',   'Td':'Td',   'Sx':'Sx',   'Tv':'Tv',   'Tf':'Tf'   }
MAP_SRC   = { 'Y':'1mo', 'Q':'1mo', 'M':'1d',  'W':'1d',  'D':'1d',    'Hf':'30m',  'Td':'15m',  'Sx':'15m',  'Tv':'5m',   'Tf':'5m'   } 
MAP_UNTIL = { 'Y':'max', 'Q':'50Y', 'M':'50Y', 'W':'50Y', 'D':'50Y',   'Hf':'60D',  'Td':'60D',  'Sx':'60D',  'Tv':'60D',  'Tf':'60D'  } 

MAP_GRAN  = { 'Y':12, 'Q':3, 'M':21, 'W':5,  'D':1,     'Hf':6,     'Td':8,     'Sx':4,     'Tv':6,      'Tf':3      } 
MAP_ANUM  = { 'Y':1,  'Q':4, 'M':12, 'W':52, 'D':252,   'Hf':2*252, 'Td':3*252, 'Sx':6*252, 'Tv':12*252, 'Tf':24*252 } 

def PRESET(TF:TP_TF, to:Lit['tuple','dict']='tuple'):
    TF    = TF
    SFX   = MAP_SFX[TF]
    SRC   = MAP_SRC[TF]
    UNTIL = MAP_UNTIL[TF]
    GRAN  = MAP_GRAN[TF]
    ANUM  = MAP_ANUM[TF]

    if to == 'tuple':   return TF, SFX, SRC, UNTIL, GRAN, ANUM
    if to == 'dict':    return { 'TF':TF, 'SFX':SFX, 'SRC':SRC, 'UNTIL':UNTIL, 'GRAN':GRAN, 'ANUM':ANUM }


# ================ CRUD ================ #
TP_SYNC    = Lit['READ','SAVE','LOAD']
TP_FORMATS = Lit['excel','sqlite','parquet']



# ================ Rebalance ================ #
TRANS_Yr = [1]
TRANS_Qr = [1,4,7,10]
TRANS_Mo = [1]
TRANS_Wk = [1]

TRANS_Hf = ['10:00:00','13:00:00']
TRANS_Td = ['10:00:00','12:00:00','14:00:00']
TRANS_Sx = ['10:00:00','11:00:00','12:00:00', '13:00:00','14:00:00','15:00:00']
TRANS_Tv = [00,30]
TRANS_Tf = [00,15,30,45]

TRANS_Hf_full = [*['01:00:00','04:00:00','07:00:00'],                       *['10:00:00','13:00:00'],            *['16:00:00','19:00:00','22:00:00']]
TRANS_Td_full = [*['00:00:00','02:00:00','04:00:00','06:00:00','08:00:00'], *['10:00:00','12:00:00','14:00:00'], *['16:00:00','18:00:00','20:00:00','22:00:00']]
TRANS_Sx_full = [*['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00'],  *['10:00:00','11:00:00','12:00:00', '13:00:00','14:00:00','15:00:00'],  *['16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']]
TRANS_Tv_full = [00,30]
TRANS_Tf_full = [00,15,30,45]

def in_transit(Df, Col, TF, full=False):
    
    A = TRANS_Hf_full if full else TRANS_Hf
    B = TRANS_Td_full if full else TRANS_Td
    C = TRANS_Sx_full if full else TRANS_Sx
    D = TRANS_Tv_full if full else TRANS_Tv
    E = TRANS_Tf_full if full else TRANS_Tf

    has_df = isinstance(Df, pd.DataFrame)

    if has_df:  Col = Df[Col]

    if (TF == 'Y'):     pipe = 1* Col.dt.month.isin(TRANS_Yr)
    if (TF == 'Q'):     pipe = 1* Col.dt.month.isin(TRANS_Qr)
    if (TF == 'M'):     pipe = 1* Col.dt.day.isin(  TRANS_Mo)
    if (TF == 'W'):     pipe = 1* Col.dt.isocalendar().day.isin(TRANS_Wk)
    if (TF == 'D'):     pipe = 1                                                            

    if (TF == 'Hf'):    pipe = 1* Col.dt.time.apply(lambda x: str(x) in A)
    if (TF == 'Td'):    pipe = 1* Col.dt.time.apply(lambda x: str(x) in B)
    if (TF == 'Sx'):    pipe = 1* Col.dt.time.apply(lambda x: str(x) in C)
    if (TF == 'Tv'):    pipe = 1* Col.dt.minute.isin(D)
    if (TF == 'Tf'):    pipe = 1* Col.dt.minute.isin(E)

    return pipe



# ================ Time Window ================ #
def Prds(TF:Lit['Y','Q','M','W','D',  'Hf','Td','Sx','Tv','Tf'], Z):
    return {
        'Y':  [1,  3,  5, 10,  20,  40], 
        'Q':  [1,  2,  4,  8,  16,  32], 
        'M':  [1,  3,  6, 12,  24,  48], 
        'W':  [1,  2,  4,  8,  16,  32], 
        'D':  [1,  3,  5, 10,  20,  40], 

        'Hf': [1,  2,  4,  8,  16,  32], 
        'Td': [1,  3,  6, 12,  24,  48], 
        'Sx': [1,  3,  6, 12,  24,  48], 
        'Tv': [1,  3,  6, 12,  24,  48], 
        'Tf': [1,  3,  6, 12,  24,  48], 
    }[TF][Z-1]

def Slug(TF:Lit['Y','Q','M','W','D',  'Hf','Td','Sx','Tv','Tf'], Z): 
    return str(Prds(TF, Z)) + TF

def aPrds(TF:Lit['Y','Q','M','W','D',  'Hf','Td','Sx','Tv','Tf'], Z):
    return np.vectorize(Prds)(TF, Z)

def aSlug(TF:Lit['Y','Q','M','W','D',  'Hf','Td','Sx','Tv','Tf'], Z): 
    return np.vectorize(Slug)(TF, Z)



# ================ Indicators ================ #
def CAGR(Df, Val, By, stp, pwr, rnd=2):
    def Lambda(x):
        x = log(1+x/100)
        x = x.expanding().mean()
        x = exp(x)
        x = x ** pwr
        x = (x-1)*100
        return x
    return Df[::stp].groupby(By)[Val].transform(Lambda).round(rnd)

def Worth(Df, Val, By, stp, mlt=10, rnd=3):
    def Lambda(x):
        x = log(1+x/100)
        x = x.expanding().sum()
        x = exp(x) * mlt
        return x
    return Df[::stp].groupby(By)[Val].transform(Lambda).round(rnd)

def Expnt(Df, Val, By, stp,  base=10, mlt=10, add=1, rnd=1,  iloc=0):

    def extreme(S):
        try:    return S.dropna().iloc[iloc]
        except: return nan
    
    def Lambda(x):
        y = x / extreme(x)
        y = log(y, base)
        y = y * mlt + add
        return y

    return Df[::stp].groupby(By)[Val].transform(Lambda).round(rnd)



# ================ Helpers ================ #
def combine_label(W, C):
    if (np.prod(W) == 1):
        return ' + '.join([f'{c}' for w, c in zip(W, C)])
    else:
        return ' + '.join([f'{w} {c}' for w, c in zip(W, C)])



