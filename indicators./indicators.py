### Indicators $$$

def get_avg(data, type='SMA', period=14, weights=np.arange(14)):
    if type == 'SMA':
        return pd.Series(data).rolling(period).mean()
    if type == 'WMA':
        return pd.Series(data).rolling(period).apply(lambda x: (x*weights)/weights.sum())
    #if type == 'EMA':
    #    return pd.Series(data).ewm(com=1, adjust=False).mean()
    if type == 'SMMA':
        return pd.Series(data).ewm(com=period-1, adjust=False).mean()

#v# ATR ## https://en.wikipedia.org/wiki/Average_true_range
def get_atr(data, average='SMMA', period=14):
    tr = pd.Series(np.maximum(
        (data.HIGH - data.LOW).values, 
        np.maximum(
            (data.HIGH - data.CLOSE.shift(1)).abs().values, 
            (data.LOW - data.CLOSE.shift(1)).abs().values
        )
    ))
    atr = get_avg(tr, type=average, period=period)
    
    atr = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'atr' : atr, 'atr_tr': tr})
    return atr

#v# AD ## https://en.wikipedia.org/wiki/Accumulation/distribution_index
def get_ad(data):    
    clv = ((data.CLOSE - data.LOW) - (data.HIGH - data.CLOSE))/(data.HIGH - data.LOW)
    accdist = clv * data.VOLUME
    accdist = accdist.cumsum()
    
    accdist = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ad' : accdist, 'ad_clv_e' : clv})
    return accdist

#v# ABANDS ## https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/
def get_abands(data, alpha=4, period=30):
    upper_band = (data.HIGH*(1+alpha*((data.HIGH-data.LOW)/(data.HIGH+data.LOW)))).rolling(period).mean()
    middle_band = data.CLOSE.rolling(period).mean()
    lower_band = (data.LOW*(1-alpha*((data.HIGH-data.LOW)/(data.HIGH+data.LOW)))).rolling(period).mean()
    
    abands = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'abands_ub' : upper_band, 'abands_mb' : middle_band, 'abands_lb' : lower_band})
    return abands

#v# ADX ## https://en.wikipedia.org/wiki/Average_directional_movement_index
def get_adx(data, period=14):
    upmove = data.HIGH - data.HIGH.shift(1)
    downmove = data.LOW.shift(1) - data.LOW
    
    pDM = ((upmove > downmove) & (upmove > 0)).values.astype(int)*upmove
    nDM = ((downmove > upmove) & (downmove > 0)).values.astype(int)*downmove
    
    pDM_avg = get_avg(pDM, type='SMMA', period=period)
    nDM_avg = get_avg(nDM, type='SMMA', period=period)
    
    atr = get_atr(data, period=period).atr
    pDI = 100*pDM_avg/atr
    nDI = 100*nDM_avg/atr
    
    adx = get_avg(100*(pDI-nDI).abs()/(pDI+nDI), type='SMMA', period=period)
    adx = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'adx' : adx, 
                        'adx_pDM_e' : pDM, 'adx_pDM_e' : pDM,
                        'adx_pDM_avg_e' : pDM_avg, 'adx_pDM_avg_e' : pDM_avg,
                        'adx_pDI_e' : pDI, 'adx_pDI_e' : pDI,  })
    return adx

#v# ALMA ## https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/
def get_alma(data, source='CLOSE', period=20, sigma=1, offset=0.85):
    m = offset*(period-1)
    s = period/sigma
    
    wts = np.array([np.exp(-((k-m)**2)/(2*s*s)) for k in range(period)])
    alma = data[source].rolling(period).apply(lambda x: (x*wts).sum()/wts.sum())
    
    alma = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'alma' : alma})
    return alma

#v# AO ## https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
def get_ao(data, period1=5, period2=34):
    s1 = get_avg((data.HIGH + data.LOW)/2, type='SMA', period=period1)
    s2 = get_avg((data.HIGH + data.LOW)/2, type='SMA', period=period2)
    ao = s1 - s2
    
    ao = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ao' : ao})
    return ao

#v# AROON ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon_oscillator
def get_aroon(data, source='CLOSE', period=25):
    arup = 100 * (period - data[source].rolling(period).apply(lambda x: np.argmax(x)))/period
    ardn = 100 * (period - data[source].rolling(period).apply(lambda x: np.argmin(x)))/period
    ar = arup - ardn
    
    aroon = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'aroon' : ar, 'ar_up': arup, 'ar_dn': ardn})
    return aroon

#v# ASI ## https://www.investopedia.com/terms/a/asi.asp
def get_asi(data):
    _r1 = data.HIGH - data.CLOSE.shift(1)
    _r2 = data.LOW - data.CLOSE.shift(1)
    _r3 = data.HIGH - data.LOW
    _r = np.vstack([_r1.values, _r2.values, _r3.values])
    
    r1 = (data.HIGH - data.CLOSE.shift(1)) - 0.5*(data.LOW - data.CLOSE.shift(1)) + 0.25*(data.CLOSE.shift(1) - data.OPEN.shift(1))
    r2 = (data.LOW - data.CLOSE.shift(1)) - 0.5*(data.HIGH - data.CLOSE.shift(1)) + 0.25*(data.CLOSE.shift(1) - data.OPEN.shift(1))
    r3 = (data.HIGH - data.LOW) + 0.25*(data.CLOSE.shift(1) - data.OPEN.shift(1))
    r = np.vstack([r1.values, r2.values, r3.values]).T
    
    r = r[np.arange(r.shape[0]), np.argmax(_r, axis=0)]
    si_nm = ((data.CLOSE.shift(1) - data.CLOSE) + 0.5*(data.CLOSE.shift(1) - data.OPEN.shift(1)) + 0.25*(data.CLOSE - data.OPEN))
    si = 50*si_nm*np.maximum(data.HIGH.shift(1)-data.CLOSE, data.LOW.shift(1)-data.CLOSE)/r/(data.HIGH - data.LOW)
    si_avg = pd.Series(si).cumsum()
    asi = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'asi' : pd.Series(si).cumsum(), 
                       'asi_si': si, 'asi_r1_e': r1, 'asi_r2_e': r2, 'asi_r3_e':r3, 'asi_nm_e':si_nm})
    return asi

#v# BB ## https://en.wikipedia.org/wiki/Bollinger_Bands
def get_bb(data, period=20, k=2):
    mb = get_avg(data.CLOSE, type='SMA', period=period)
    std = data.CLOSE.rolling(period).stdev()
    ub = mb + k*std
    lb = mb - k*std
    
    pb = (data.CLOSE - lb) / (ub - lb)
    bbw = (ub - lb) / mb
    
    bb = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 
                        'bb_mb' : mb, 'bb_ub' : ub, 
                        'bb_lb' : lb, 'bb_pb' : pb,  
                      'bb_bbw' : bbw})
    return bb

#v# CC ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve
def get_cc(data, source='CLOSE', period1=11, period2=14, period_wma=10):
    p1roc = get_roc(data, source=source, period=period1).roc
    p2roc = get_roc(data, source=source, period=period2).roc
    
    cc = get_avg(p1roc+p2roc, type='WMA', period=period_wma)
    cc = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'cc' : cc, 'cc_p1roc_e':p1roc, 'cc_p2roc_e':p2roc})
    return cc

#v# CCI ## https://en.wikipedia.org/wiki/Commodity_channel_index
def get_cci(data, period=14):
    tp = (data.HIGH + data.LOW + data.CLOSE)/3
    sma_tp = get_avg(tp, type='SMA', period=period)
    mad_tp = (sma_tp - tp).abs().rolling(period).mean()
    
    cci = (1.0/0.015)*((tp-sma_tp)/mad_tp)
    
    cci = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'cci':cci, 'cci_tp_e':tp,
                       'cci_sma_tp_e':sma_tp, 'cci_mad_tp_e':mad_tp})
    return cci

#v# CE ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chandelier_exit
def get_ce(data, period=22, k=30):
    atr = get_atr(data, average='SMMA', period=period).atr
    ce_long = data.HIGH.rolling(period).max() - atr*k
    ce_shrt = data.LOW.rolling(period).min() + atr*k
    
    ce = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ce_long' : ce_long, 'ce_shrt': ce_shrt})
    return ce

#v# CHOP ## https://www.tradingview.com/wiki/Choppiness_Index_(CHOP)
def get_chop(data, period=14):
    atr = get_atr(data, period=1).atr
    atr_sum = atr.rolling(period).sum()
    maxhi = data.HIGH.rolling(period).max()
    minlo = data.LOW.rolling(period).min()
    
    raw_chop = atr_sum/(maxhi - minlo)
    
    chop = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'chop' : raw_chop})
    return chop

#v# CMO ## https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/chande-momentum-oscillator-cmo/
def get_cmo(data, source='CLOSE', period=30):
    ps = ((data[source] > data[source].shift(1)).astype(int)*(data[source] - data[source].shift(1))).rolling(period).sum()
    ns = ((data[source] < data[source].shift(1)).astype(int)*(data[source].shift(1) - data[source])).rolling(period).sum()
    cmo = (ps-ns) / (ps+ns) * 100
    
    cmo = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'cmo' : cmo, 'cmo_ps':ps, 'cmo_ns': ns})
    return cmo

#v# CRSI ## https://www.tradingview.com/wiki/Connors_RSI_(CRSI)
def get_csri(data, source='CLOSE', rsi_period=3, updown_length=2, roc_period=100):
    rsi = get_rsi(data, period=rsi_period).rsi
    
    udl = np.zeros_like(data[source].values)
    trend = 1
    idx = 1
    while idx < len(udl):
        if trend == 1:
            if data[source][idx] > data[source][idx-1]:
                udl[idx] = udl[idx-1]+1
            else:
                udl[idx] = 1
                trend = 0
        elif trend == 0:
            if data[source][idx] < data[source][idx-1]:
                udl[idx] = udl[idx-1]+1
            else:
                udl[idx] = 1
                trend = 1
        idx = idx+1
    udl_rsi = get_rsi(pd.DataFrame({'OPEN_TIME': data.OPEN_TIME, 'udl':udl}), source='udl', period=updown_length).rsi
    udl_rsi = udl_rsi.fillna(value=0)
    
    roc = get_roc(data, source=source, period=1).roc
    mroc = roc.rolling(roc_period).apply(lambda x: (x < x[-1]).sum()/float(roc_period)*100)
    
    crsi = (rsi + udl_rsi + mroc)/3
    crsi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'crsi':crsi, 'crsi_rsi':rsi, 'crsi_udl':udl,
                         'crsi_udl_rsi':udl_rsi, 'crsi_mroc':mroc})
    return crsi

#v# DPO ## https://en.wikipedia.org/wiki/Detrended_price_oscillator
def get_dpo(data, period=14):
    dpo = data.CLOSE - get_avg(data.CLOSE, type='SMA', period=period).shift(int(period/2)+1)
    
    dpo = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'dpo':dpo})
    return dpo

#v# DC ## https://www.tradingview.com/wiki/Donchian_Channels_(DC)
def get_dc(data, period=30):
    uc = data.HIGH.rolling(period).max()
    lc = data.LOW.rolling(period).min()
    mc = (uc+lc)/2
    
    dc = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'dc_uc': uc,
                      'dc_lc': lc, 'dc_mc': mc})
    return dc

#v# EOM ## https://www.tradingview.com/wiki/Ease_of_Movement_(EOM)
def get_eom(data, period=14):
    dm = (data.HIGH+data.LOW)/2 - (data.HIGH.shift(1)+data.LOW.shift(1))/2
    br = data.VOLUME/1000000/(data.HIGH - data.LOW)
    eom = dm/br
    eomma = get_avg(eom, type='SMA', period=period)
    
    eom = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'eom':eomma,
                       'eom_dm_e': dm, 'eom_br_e': br})
    return eom

#v# EFI ## https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
def get_efi(data, period=13):
    efi = (data.CLOSE - data.CLOSE.shift(1))*data.VOLUME
    efima = get_avg(efi, type='SMMA', period=period)
    
    efi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'efi':efima,
                       'efi_raw_e': efi})
    return efi

#v# ENV ## https://www.tradingview.com/wiki/Envelope_(ENV)
def get_env(data, source='CLOSE', period=9, multiplier=0.01):
    basis = get_avg(data[source], type='SMA', period=period)
    ue = basis + multiplier*basis
    le = basis - multiplier*basis
    
    env = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'env':basis, 
                        'env_ue':ue, 'env_le': le})
    return env

#v# FI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
def get_fi(data, period=13):
    fi = (data.CLOSE - data.CLOSE.shift(1))*data.VOLUME
    fi = get_avg(fi, type='SMMA', period=period)
    
    fi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'fi':fi})
    return fi

#v# KAMA ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:kaufman_s_adaptive_moving_average
def get_kama(data, period_er=10, period1_sc=2, period2_sc=30):
    chng = (data.CLOSE - data.CLOSE.shift(period_er)).abs()
    vol = (data.CLOSE - data.CLOSE.shift(1)).abs().rolling(period_er).sum()
    er = chng/vol
    
    sc = (er*(2.0/(1+period1_sc) - 2.0/(1+period2_sc)) + 2.0/(1+period2_sc))**2
    kama = sc*data.CLOSE 
    kama = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'kama':kama, 'kama_er':er, 'kama_sc_e':sc})
    return kama

#v# KC ## https://www.tradingview.com/wiki/Keltner_Channels_(KC)
def get_kc(data, period_basis=30, period_atr=14, multiplier=2):
    basis = get_avg(data.CLOSE, type='SMMA', period=period_basis)
    atr = get_atr(data, average='SMMA', period=period_atr).atr
    ue = basis + multiplier*atr
    le = basis - multiplier*atr
    
    kc = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'kc_ue':ue,
                         'kc_le': le, 'kc': basis})
    return kc

#v# KST ## https://www.tradingview.com/wiki/Know_Sure_Thing_(KST)
def get_kst(data, source='CLOSE', pROC1=10, pROC2=15, pROC3=20, pROC4=25, pSMA1=10, pSMA2=10, pSMA3=10, pSMA4=15, period=9):
    roc1 = get_roc(data, source=source, period=pROC1).roc
    roc2 = get_roc(data, source=source, period=pROC2).roc
    roc3 = get_roc(data, source=source, period=pROC3).roc
    roc4 = get_roc(data, source=source, period=pROC4).roc
    
    rocma1 = get_avg(roc1, type='SMA', period=pSMA1)
    rocma2 = get_avg(roc2, type='SMA', period=pSMA2)
    rocma3 = get_avg(roc3, type='SMA', period=pSMA3)
    rocma4 = get_avg(roc4, type='SMA', period=pSMA4)
    
    kst = rocma1*1 + rocma2*2 + rocma3*3 + rocma4*4
    kstma = get_avg(kst, type='SMA', period=period)
    
    kst = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'kst':kstma,
                         'kst_raw_e': kst})
    return kst

#v# MACD ## https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
def get_macd(data, source='CLOSE', sig_period=9, period1=12, period2=26):
    p1ema = get_avg(data[source], type='EMA', period=period1)
    p2ema = get_avg(data[source], type='EMA', period=period2)
    macd_line = p1ema - p2ema
    sig_line = get_avg(macd_line, type='EMA', period=sig_period)
    macd_hist = macd_line - sig_line
    
    macd = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'macd':macd_line,
                         'macd_sig': sig_line, 'macd_hist': macd_hist})
    return macd

#v# MI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
def get_mi(data, period1=9, period2=25):
    hl = data.HIGH - data.LOW
    sema = get_avg(hl, type='SMMA', period=period1)
    dema = get_avg(sema, type='SMMA', period=period2)
    ema_ratio = sema/dema
    mi = ema_ratio.rolling(period2).sum()
    
    mi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'mi':mi, 'mi_sema_e':sema, 'mi_dema_e':dema,
                       'mi_ema_ratio_e':ema_ratio})
    return mi
    
#v# MFI ## https://www.tradingview.com/wiki/Money_Flow_(MFI)
def get_mfi(data, period=14):
    tp = (data.HIGH + data.LOW + data.CLOSE)/3
    rmf = tp * data.VOLUME
    
    pnmf = pd.Series(((rmf > rmf.shift(1)).values.astype(float) - 0.5)*2*rmf)
    
    pmf = pnmf.rolling(period).apply(lambda x: ((x>0).astype(int)*x).sum())
    nmf = pnmf.rolling(period).apply(lambda x: -((x<0).astype(int)*x).sum())
    
    mfr = pmf/nmf
    mfi = 100 - 100/(1+mfr)
    
    mfi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'mfi':mfi, 'mfi_mfr_e':mfr,
                       'mfi_pmf_e':pmf, 'mfi_nmf_e':nmf, 'mfi_tp_e': tp,
                       'mfi_rmf_e': rmf})
    return mfi

#t# NVI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde
def get_nvi(data, source='VOLUME', period=255):
    vo_id = (data.VOLUME < data.VOLUME.shift(1)).values.astype(int)
    chng = (data[source] - data[source].shift(1))/data[source].shift(1)
    
    nvi = (vo_id*chng).cumsum()
    nvi = get_avg(nvi, type='SMMA', period=period)
    
    nvi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'nvi':nvi})
    return nvi

#t# OBV ## https://www.tradingview.com/wiki/On_Balance_Volume_(OBV)
def get_obv(data):
    obv = ((data.CLOSE - data.CLOSE.shift(1)).values.astype(float) - 0.5)*2*data.VOLUME
    obv = obv.cumsum()
    obv = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'obv':obv})
    return obv

#v# SAR ## https://www.tradingview.com/wiki/Parabolic_SAR_(SAR)
def get_sar(data, af=0.02, af_inc=0.01, af_max=0.2):
    sar = np.zeros_like(data.CLOSE.values)
    trd = np.zeros_like(data.CLOSE.values)
    
    if data.CLOSE[0] < data.CLOSE[1]:
        trend = 1
        trd[0] = 1
        sar[0] = data.HIGH[0]
        ext = data.LOW[0]
    else:
        trend = 0
        trd[0] = 0
        sar[0] = data.LOW[0]
        ext = data.HIGH[0]
    
    idx = 1
    af_curr = af
    while idx < len(sar):
        trd[idx] = trend
        sar[idx] = sar[idx-1] + af_curr * (ext - sar[idx-1])
        af_curr = min(af_curr + af_inc, af_max)
        if trend == 1:
            if data.HIGH[idx] > sar[idx]:
                trend = 0
                sar[idx] = ext
                ext = data.HIGH[idx]
                af_curr = af
            else:
                ext = min(ext, data.LOW[idx])
        elif trend == 0:
            if data.LOW[idx] < sar[idx]:
                trend = 1
                sar[idx] = ext
                ext = data.LOW[idx]
                af_curr = af
            else:
                ext = max(ext, data.HIGH[idx])
        idx = idx + 1
    
    sar = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'sar':sar, 'sar_trd':trd})
    return sar

#v# PO ## https://www.tradingview.com/wiki/Price_Oscillator_(PPO)
def get_po(data, source='CLOSE', period1=12, period2=26, period_sig=9):
    po1 = get_avg(data[source], type='SMMA', period=period1)
    po2 = get_avg(data[source], type='SMMA', period=period2)
    
    po = (po1 - po2) / po2
    
    sig = get_avg(po, type='SMMA', period=period_sig)
    
    hist = po - sig
    
    po = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'po':po, 'po_sig':sig, 'po_hist':hist, 'po_po1_e':po1, 'po_po2_e':po2})
    return po

#v# PVT ## https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)
def get_pvt(data):
    raw_pvt = data.VOLUME * (data.CLOSE - data.CLOSE.shift(1))/data.CLOSE.shift(1) * data.VOLUME
    pvt = raw_pvt.cumsum()
    pvt = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'pvt':pvt,
                       'pvt_raw_e': raw_pvt})
    return pvt
    
#v# ROC ## https://www.tradingview.com/wiki/Rate_of_Change_(ROC)
def get_roc(data, source='CLOSE', period=14):
    roc = 100*(data[source] - data[source].shift(period))/data[source].shift(period)
    roc = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'roc':roc})
    return roc

#v# RSI ## https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)
def get_rsi(data, source='CLOSE', period=14):
    diff = (data[source] - data[source].shift(1)).values
    gain = (diff > 0).astype(int) * diff
    loss = (diff < 0).astype(int) * -diff
    
    gain_ma = get_avg(gain, type='SMA', period=period)
    loss_ma = get_avg(loss, type='SMA', period=period)
    
    rs = gain_ma/loss_ma
    
    rsi = 100 - (100 / (1 + rs))
    rsi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'rsi':rsi, 'rsi_rs_e':rs,
                       'rsi_diff_e': diff, 'rsi_gain_e': gain,
                       'rsi_loss_e': loss, 'rsi_gain_ma_e': gain_ma,
                       'rsi_loss_ma_e': loss_ma})
    return rsi

#v# SCTR ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:sctr
def get_sctr(data, source='CLOSE', period1ab=200, period1roc=125, period2ab=50, period2roc=20, periodrsi=14):
    pab200 = (data[source] - get_avg(data[source], type='SMA', period=period1ab))/get_avg(data[source], type='SMA', period=period1ab)
    roc125 = get_roc(data, source=source, period=period1roc).roc
    
    pab50 = (data[source] - get_avg(data[source], type='SMA', period=period2ab))/get_avg(data[source], type='SMA', period=period2ab)
    roc20 = get_roc(data, source=source, period=period2roc).roc
    
    rsi = get_rsi(data, source=source, period=periodrsi).rsi
    
    sctr = 0.3*pab200 + 0.3*roc125 + 0.15*pab50 + 0.15*roc20 + 0.1*rsi
    
    sctr = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'sctr':sctr})
    return sctr

## SFS
#v# STOCH ## https://www.tradingview.com/wiki/Stochastic_(STOCH)
def get_stoch(data, source='CLOSE', period_k=14, smooth_k=3, smooth_d=3):
    pK_series = (data[source] - data.LOW.rolling(period_k).min())/(data.HIGH.rolling(period_k).max() - data.LOW.rolling(period_k).min())
    pK = get_avg(pK_series, type='SMA', period=smooth_k)
    pD = get_avg(pK, type='SMA', period=smooth_d)
    
    stoch = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'stoch_pK':pK,
                       'stoch_pD': pD, 'stoch_pK_s_e': pK_series})
    return stoch

#v# TSI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:true_strength_index
def get_tsi(data, period1=13, period2=25):
    pc = data.CLOSE - data.CLOSE.shift(1)
    pc_fs = get_avg(pc, type='SMMA', period=period1)
    pc_ss = get_avg(pc_fs, type='SMMA', period=period2)
    
    apc = (data.CLOSE - data.CLOSE.shift(1)).abs()
    apc_fs = get_avg(apc, type='SMMA', period=period1)
    apc_ss = get_avg(apc_fs, type='SMMA', period=period2)
    
    tsi = 100 * pc_ss/apc_ss
    tsi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'tsi':tsi, 'tsi_pc_ss_e':pc_ss, 'tsi_apc_ss_e':apc_ss})
    return tsi
    
#v# TRIX ## https://www.tradingview.com/wiki/TRIX
def get_trix(data, source='CLOSE', period=18):
    ssmma = get_avg(data[source], type='SMMA', period=period)
    dsmma = get_avg(ssmma, type='SMMA', period=period)
    tsmma = get_avg(dsmma, type='SMMA', period=period)
    
    pdiff = (tsmma - tsmma.shift(1))/tsmma.shift(1)
    
    trix = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'trix':pdiff, 
                         'trix_ssmma_e':ssmma,
                         'trix_dsmma_e':dsmma,
                         'trix_tsmma_e':tsmma})
    return trix

#v# UI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index
def get_ui(data, period=14):
    pd = (data.CLOSE - data.CLOSE.rolling(period).max())/data.CLOSE.rolling(period).max()*100
    sa = pd.rolling(14).std()
    sa = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'sa':sa, 'sa_pd':pd})
    return sa

#v# UO ## https://www.tradingview.com/wiki/Ultimate_Oscillator_(UO)
def get_uo(data, period1=7, period2=14, period3=28):
    tl = pd.Series(np.minimum(data.LOW, data.CLOSE.shift(1))) # true low
    bp = data.CLOSE - tl # buying pressure
    
    tr = pd.Series(np.maximum(data.HIGH, data.CLOSE.shift(1)) - np.minimum(data.LOW, data.CLOSE.shift(1)))
    
    s1 = bp.rolling(period1).sum()/tr.rolling(period1).sum()
    s2 = bp.rolling(period2).sum()/tr.rolling(period2).sum()
    s3 = bp.rolling(period3).sum()/tr.rolling(period3).sum()
    
    min_period = min(period1, min(period2, period3))
    max_period = max(period1, max(period2, period3))
    mid_period = period1+period2+period3 - min_period - max_period
    
    if min_period==period1:
        if period2 > period3:
            _ = s2
            s2 = s3
            s3 = _
    elif min_period==period2:
        _ = s1
        s1 = s2
        if period1 < period3:
            s2 = _
        else:
            s2 = s3
            s3 = _
    elif min_period==period3:
        _ = s1
        s1 = s3
        if period1 < period2:
            s3 = s2
            s2 = _
        else:
            s3 = _
    
    uo = 100*(max_period*s1 + mid_period*s2 + min_period*s3)/(max_period+mid_period+min_period)
    uo = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'uo':uo, 'uo_tl_e':tl,
                       'uo_bp_e':bp, 'uo_tr_e':tr, 'uo_s1_e':s1,
                       'uo_s2_e':s2, 'uo_s3_e':s3})
    return uo

#v# VA ## http://www.onlinetradingconcepts.com/TechnicalAnalysis/VolumeAccumulation.html
def get_va(data):
    va = data.VOLUME*(data.CLOSE - (data.HIGH+data.LOW)/2)
    va = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'va':va})
    return va
    
#v# VI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
def get_vi(data, period=14):
    pVM = (data.HIGH - data.LOW.shift(1)).abs().rolling(period).sum()
    nVM = (data.LOW - data.HIGH.shift(1)).abs().rolling(period).sum()
    
    tr = get_atr(data, average='SMMA', period=period).atr_tr.rolling(period).sum()
    
    pVI = pVM/tr
    nVI = nVM/tr
    
    vi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'vi_pVI':pVI, 'vi_nVI':nVI,
                      'vi_pVM_e':pVM, 'vi_nVM_e':nVM})
    return vi

#v# VWAP ## https://www.tradingview.com/wiki/Volume_Weighted_Average_Price_(VWAP)
def get_vwap(data, period=30):
    tp = (data.HIGH + data.LOW + data.CLOSE)/3
    tp_v = tp*data.VOLUME
    
    vwap = tp_v.rolling(period).sum()/tp.rolling(period).sum()
    vwap = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'vwap':vwap, 'vwap_tp_e':tp,
                       'vwap_tp_v_e':tp_v})
    return vwap

#v# WR ## https://www.tradingview.com/wiki/Williams_%25R_(%25R)
def ger_wr(data, period=30):
    hh = data.HIGH.rolling(period).max()
    ll = data.LOW.rolling(period).min()
    
    wr = (hh - data.CLOSE)/(hh - ll)
    wr = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'wr':wr, 'wr_hh_e':hh, 'wr_ll_e':ll})
    
    return wr
