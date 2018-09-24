### Indicators $$$

def get_avg(data, type='SMA', period=14, weights=np.arange(14)):
    if type == 'SMA':
        return pd.Series(data).rolling(period).mean()
    if type == 'WMA':
        return pd.Series(data).rolling(period).apply(lambda x: (x*weights)/weights.sum())
    if type == 'EMA':
        return pd.Series(data).ewm(com=1, adjust=False).mean()
    if type == 'SMMA':
        return pd.Series(data).ewm(com=period-1, adjust=False).mean()
        
## MA ## https://en.wikipedia.org/wiki/Moving_average
def get_ma(data, item='CLOSE', period=30):
    itm = data[item]
    ma = itm.rolling(period).mean()
    
    ma = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ma' : ma})
    return ma 

## ATR ## https://en.wikipedia.org/wiki/Average_true_range
def get_atr(data, average='SMMA', period=14):
    tr = pd.Series(np.maximum(
        (data.HIGH - data.LOW).values, 
        np.maximum(
            (data.HIGH - data.CLOSE.shift(-1)).abs().values, 
            (data.LOW - data.CLOSE.shift(-1)).abs().values
        )
    ))
    atr = get_avg(tr, type=average, period=period)
    
    atr = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'atr' : atr, 'tr': tr})
    return atr

## AD ## https://en.wikipedia.org/wiki/Accumulation/distribution_index
def get_ad(data):
    o = data.OPEN
    h = data.HIGH
    l = data.LOW
    c = data.CLOSE
    v = data.VOLUME
    
    clv = ((c - l) - (h - c))/(h - l)
    accdist = clv * v
    accdist = accdist.rolling(2).sum()
    
    accdist = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ad' : accdist, 'ad_clv_e' : clv})
    return accdist

## ABANDS ## https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/
def get_abands(data, alpha=4, period=30):
    upper_band = (data.HIGH*(1+alpha*((data.HIGH-data.LOW)/(data.HIGH+data.LOW)))).rolling(period).mean()
    middle_band = get_ma(data, period=period).ma
    lower_band = (data.LOW*(1-alpha*((data.HIGH-data.LOW)/(data.HIGH+data.LOW)))).rolling(period).mean()
    
    abands = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'abands_ub' : upper_band, 'abands_mb' : middle_band, 'abands_lb' : lower_band})
    return abands

## ADX ## https://en.wikipedia.org/wiki/Average_directional_movement_index
def get_adx(data, period=14):
    upmove = data.HIGH - data.HIGH.shift(-1)
    downmove = data.LOW - data.LOW.shift(-1)
    
    pDM = ((upmove > downmove) & (upmove > 0)).values.astype(int)*upmove
    nDM = ((downmove > upmove) & (downmove > 0)).values.astype(int)*downmove
    
    pDM = get_avg(pDM, type='SMMA', period=period)
    nDM = get_avg(nDM, type='SMMA', period=period)
    
    atr = get_atr(data, period=period)
    pDI = 100*pDM/atr.atr
    nDI = 100*nDM/atr.atr
    
    adx = get_avg(100*(pDI-nDI).abs()/(pDI+nDI), type='SMMA', period=period)
    adx = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'adx' : adx, 
                        'adx_pDM_e' : pDM, 'adx_pDM_e' : pDM, 
                        'adx_pDI_e' : pDI, 'adx_pDI_e' : pDI,  })
    return adx

## AO ## https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
def get_ao(data, period1=5, period2=34):
    s1 = get_avg((data.HIGH + data.LOW)/2, type='SMA', period=period1)
    s2 = get_avg((data.HIGH + data.LOW)/2, type='SMA', period=period2)
    ao = s1 - s2
    
    ao = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ao' : ao})
    return ao

## AROON ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon_oscillator
def get_aroon(data, source='CLOSE', period=25):
    arup = 100 * (period - data[source].rolling(period).apply(lambda x: np.argmax(x)))/period
    ardn = 100 * (period - data[source].rolling(period).apply(lambda x: np.argmin(x)))/period
    ar = arup - ardn
    
    aroon = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'aroon' : ar, 'ar_up': arup, 'ar_dn': ardn})
    return aroon

## BB ## https://www.tradingview.com/wiki/Bollinger_Bands_%25B_(%25B)
def get_bb(data, period=20, k=2):
    mb = get_avg(data.CLOSE, type='SMA', period=period)
    std = data.CLOSE.rolling(period).stdev()
    ub = mb + k*std
    lb = mb - k*std
    
    pb = (data.CLOSE - lb) / (ub - lb)
    bbw = (ub - lb) / mb
    
    bb = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'bb' : bb, 
                        'bb_mb' : mb, 'bb_ub' : ub, 
                        'bb_lb' : lb, 'bb_pb' : pb,  
                      'bb_bbw' : bbw})
    return bb

## CE ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chandelier_exit
def get_ce(data, period=22, k=3):
    atr = get_atr(data, average='SMMA', period=period).atr
    ce_long = data.HIGH.rolling(period).max() - atr*k
    ce_shrt = data.LOW.rolling(period).min() + atr*k
    
    ce = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'ce_long' : ce_long, 'ce_shrt': ce_shrt})
    return ce

## CHOP ## https://www.tradingview.com/wiki/Choppiness_Index_(CHOP)
def get_chop(data, period=14):
    atr = get_atr(data, period=1).atr
    atr_avg = get_avg(atr, type='SMA', period=period)
    maxhi = data.HIGH.rolling(period).max()
    minlo = data.LOW.rolling(period).min()
    
    raw_chop = atr_avg/(maxhi - minlo)
    
    chop = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'chop' : raw_chop})
    return chop

## CC ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve
def get_cc(data, source='CLOSE', period1=11, period2=14, period_wma=10):
    p1roc = get_roc(data, source=source, period=period1).roc
    p2roc = get_roc(data, source=source, period=period2).roc
    
    cc = get_avg(p1roc+p2roc, type='WMA', period=period_wma)
    cc = pd.DataFrame({'OPEN_TIME' : data.OPEN_TIME, 'cc' : cc, 'cc_p1roc_e':p1roc, 'cc_p2roc_e':p2roc})
    return cc

## CCI ## https://en.wikipedia.org/wiki/Commodity_channel_index
def get_cci(data, period=14):
    tp = (data.HIGH + data.LOW + data.CLOSE)/3
    sma_tp = get_avg(tp, type='SMA', period=period)
    mad_tp = (sma_tp - tp).abs()
    
    cci = (1.0/0.015)*((tp-sma_tp)/mad_tp)
    
    cci = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'cci':cci, 'cci_tp_e':tp,
                       'cci_sma_tp_e':sma_tp, 'cci_mad_tp_e':mad_tp})
    return cci

## CRSI ## https://www.tradingview.com/wiki/Connors_RSI_(CRSI)
def get_csri(data, rsi_period=3, updown_length=2, roc_period=100):
    rsi = get_rsi(data, period=rsi_period)
    return

## DPO ## https://en.wikipedia.org/wiki/Detrended_price_oscillator
def get_dpo(data, period=14):
    dpo = data.CLOSE - get_avg(data.CLOSE, type='SMA', period=period).shift(int(period/2)+1)
    
    dpo = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'dpo':dpo})
    return dpo

## DC ## https://www.tradingview.com/wiki/Donchian_Channels_(DC)
def get_dc(data, period=30):
    uc = data.HIGH.rolling(period).max()
    lc = data.LOW.rolling(period).min()
    mc = (uc+lc)/2
    
    dc = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'dc_uc': uc,
                      'dc_lc': lc, 'dc_mc': mc})
    return dc

## EOM ## https://www.tradingview.com/wiki/Ease_of_Movement_(EOM)
def get_eom(data, period=14):
    dm = (data.HIGH+data.LOW)/2 - (data.HIGH.shift(1)+data.LOW.shift(1))/2
    br = data.VOLUME/1000000/(data.HIGH - data.LOW)
    eom = dm/br
    eomma = get_avg(eom, type='SMA', period=period)
    
    eom = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'eom':eomma,
                       'eom_dm_e': dm, 'eom_br_e': br})
    return eom

## EFI ## https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
def get_efi(data, period=13):
    efi = (data.CLOSE - data.CLOSE.shift(1))*data.VOLUME
    efima = get_avg(efi, type='EMA', period=period)
    
    efi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'efi':efima,
                       'efi_raw_e': efi})
    return efi

## ENV ## https://www.tradingview.com/wiki/Envelope_(ENV)
def get_env(data, source='CLOSE', period=9, multiplier=0.1):
    basis = get_avg(data[source], type='SMA', period=period)
    ue = basis + multiplier*basis
    le = basis - multiplier*basis
    
    env = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'env':basis, 
                        'env_ue':ue, 'env_le': le})
    return env

## FI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
def get_fi(data, period=13):
    fi = (data.CLOSE - data.CLOSE.shift(1))*data.VOLUME
    fi = get_avg(fi, type='SMMA', period=period)
    
    fi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'fi':fi})
    return fi

## KAMA ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:kaufman_s_adaptive_moving_average
def get_kama(data, er_period=10, sc_period1=2, sc_period2=30):
    chng = (data.CLOSE - data.CLOSE.shift(er_period)).abs()
    vol = (data.CLOSE - data.CLOSE.shift(1)).abs().rolling(period).sum()
    er = chng/vol
    
    sc = (er*(2/(1+sc_period1) - 2/(1+sc_period2)) + 2/(1+sc_period2))**2
    kama = sc*data.CLOSE 
    return

## KC ## https://www.tradingview.com/wiki/Keltner_Channels_(KC)
def get_kc(data, basis_period=30, multiplier=2):
    basis = get_avg(data.CLOSE, type='SMA', period=basis_period)
    atr = get_atr(data, average='SMMA', period=14).atr
    ue = basis + multiplier*atr
    le = basis - multiplier*atr
    
    kc = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'kc_ue':ue,
                         'kc_le': kc, 'kc': basis})
    return kc

## KST ## https://www.tradingview.com/wiki/Know_Sure_Thing_(KST)
def get_kst(data, source='CLOSE', pROC1=10, pROC2=15, pROC3=20, pROC4=25, pSMA1=10, pSMA2=10, pSMA3=10, pSMA4=15, period=9):
    roc1 = get_roc(data, source='CLOSE', period=pROC1).roc
    roc2 = get_roc(data, source='CLOSE', period=pROC2).roc
    roc3 = get_roc(data, source='CLOSE', period=pROC3).roc
    roc4 = get_roc(data, source='CLOSE', period=pROC4).roc
    
    rocma1 = get_avg(roc1, type='SMA', period=pSMA1)
    rocma2 = get_avg(roc2, type='SMA', period=pSMA2)
    rocma3 = get_avg(roc3, type='SMA', period=pSMA3)
    rocma4 = get_avg(roc4, type='SMA', period=pSMA4)
    
    kst = rocma1*1 + rocma2*2 + rocma3*3 + rocma4*4
    kstma = get_avg(kst, type='SMA', period=period)
    
    kst = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'kst':kstma,
                         'kst_raw_e': kst})
    return kst

## MACD ## https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
def get_macd(data, source='CLOSE', sig_period=9, period1=12, periodd2=26):
    p1ema = get_avg(data[source], type='EMA', period=period1)
    p2ema = get_avg(data[source], type='EMA', period=period2)
    macd_line = p1ema - p2ema
    sig_line = get_avg(macd_line, type='EMA', period=sig_period)
    macd_hist = macd_line - sig_line
    
    macd = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'macd':macd_line,
                         'macd_sig': sig_line, 'macd_hist': macd_hist})
    return macd

## MI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
def get_mi(data, period1=9, period2=25):
    hl = data.HIGH - data.LOW
    sema = get_avg(hl, type='SMMA', period=period1)
    dema = get_avg(sema, type='SMMA', period=period2)
    ema_ratio = sema/dema
    mi = ema_ratio.rolling(period2).sum()
    
    mi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'mi':mi, 'mi_sema_e':sema, 'mi_dema_e':dema,
                       'mi_ema_ratio_e':ema_ratio})
    return mi
    
## MFI ## https://www.tradingview.com/wiki/Money_Flow_(MFI)
def get_mfi(data, period=14):
    tp = (data.HIGH + data.LOW + data.CLOSE)/3
    rmf = tp * data.VOLUME
    
    pnmf = pd.Series((rmf > rmf.shift(1)).values.astype(int)*rmf)
    
    pmf = pnmf.rolling(period).apply(lambda x: ((x>0).values.astype(int)*x).sum())
    nmf = pnmf.rolling(period).apply(lambda x: -((x<0).values.astype(int)*x).sum())
    
    mfr = pmf/nmf
    mfi = 100 - 100/(1+mfr)
    
    mfi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'mfi':mfi, 'mfi_mfr_e':mfr,
                       'mfi_pmf_e':pmf, 'mfi_nmf_e':nmf, 'mfi_tp_e': tp,
                       'mfi_rmf_e': rmf})
    return mfi

## NVI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde
def get_nvi(data, source='VOLUME', period=255):
    vo_id = (data.VOLUME < data.VOLUME.shift(1)).values.astype(int)
    chng = (data[source] - data[source].shift(1))/data[source].shift(1)
    
    nvi = (vo_id*chng).cumsum()
    nvi = get_avg(nvi, type='SMMA', period=period)
    
    nvi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'nvi':nvi})
    return nvi

## OBV ## https://www.tradingview.com/wiki/On_Balance_Volume_(OBV)
def get_obv(data):
    obv = ((data.CLOSE - data.CLOSE.shift(1)).values.astype(float) - 0.5)*2*data.VOLUME
    obv = obv.cumsum()
    obv = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'obv':obv})
    return obv

## SAR ## https://www.tradingview.com/wiki/Parabolic_SAR_(SAR)
def get_sar(data):
    return

## PO ## https://www.tradingview.com/wiki/Price_Oscillator_(PPO)
def get_po(data, source='CLOSE', period1=10, period=21):
    po1 = get_avg(data[source], type='EMA', period=period1)
    po2 = get_avg(data[source], type='EMA', period=period2)
    
    ppo = (po1 - po2) / po2
    
    ppo = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'ppo':ppo})
    return ppo

## PVT ## https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)
def get_pvt(data):
    raw_pvt = data.VOLUME * (data.CLOSE - data.CLOSE.shift(1))/data.CLOSE.shift(1) * data.VOLUME
    pvt = raw_pvt.cumsum()
    pvt = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'pvt':pvt,
                       'pvt_raw_e': raw_pvt})
    return pvt
    
## ROC ## https://www.tradingview.com/wiki/Rate_of_Change_(ROC)
def get_roc(data, source='CLOSE', period=14):
    roc = 100*(data[source] - data[source].shift(period))/data[source].shift(period)
    roc = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'roc':roc})
    return roc

## RSI ## https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)
def get_rsi(data, source='CLOSE', period=14):
    diff = (data[source] - data[source].shift(1)).values
    gain = (diff > 0).astype(int) * diff
    loss = (diff < 0).astype(int) * -diff
    
    gain_ma = get_avg(gain, type='SMA', period=period)
    loss_ma = get_avg(loss, type='SMA', period=period)
    
    rs = gain_ma/loss_ma
    
    rsi = 100 - (100 / (1 + rs))
    rsi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'rsi':rsi,
                       'rsi_diff_e': diff, 'rsi_gain_e': gain,
                       'rsi_loss_e': loss, 'rsi_gain_ma_e': gain_ma,
                       'rsi_loss_ma_e': loss_ma})
    return rsi

## SCTR ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:sctr
def get_sctr(data, source='CLOSE', period1ab=200, period1roc=125, period2ab=50, period2roc=20, period_rsi=14):
    pab200 = (data[source] - get_avg(data[source], type='SMA', period=period1ab))/get_avg(data[source], type='SMA', period=period1ab)
    roc125 = get_roc(data, source=source, period=period1roc).roc
    
    pab50 = (data[source] - get_avg(data[source], type='SMA', period=period2ab))/get_avg(data[source], type='SMA', period=period2ab)
    roc20 = get_roc(data, source=source, period=period2roc).roc
    
    rsi = get_rsi(data, source=source, period=periodrsi).rsi
    
    sctr = 0.3*pab200 + 0.3*roc125 + 0.15*pab50 + 0.15*roc20 + 0.1*rsi
    
    sctr = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'sctr':sctr})
    return sctr

## STOCH ## https://www.tradingview.com/wiki/Stochastic_(STOCH)
def get_stoch(data, period_k=14, smooth_k=3, smooth_d=3):
    pK_series = (data.CLOSE - data.LOW.rolling(period_k).min())/(data.HIGH.rolling(period_k).max() - data.LOW.rolling(period_k).min())
    pK = get_avg(pK_series, type='SMA', period=smooth_k)
    pD = get_avg(pK, type='SMA', period=smooth_d)
    
    stoch = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'stoch_pK':pK,
                       'stoch_pD': pD, 'stoch_pK_s_e': pK_series})
    return stoch

## STOCH RSI ## https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI)
def get_stochrsi(data, source='CLOSE', period_k=14, smooth_stoch=3, smooth_d=3, rsi_period=14):
    rsi = get_rsi(data, source='CLOSE', period=rsi_period)
    stochrsi = get_stoch(rsi.rsi, period_k=period_k, smooth_k=smooth_stoch, smooth_d=smooth_d)
    
    stochrsi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'stochrsi_pK':stochrsi.stoch_pK,
                       'stochrsi_pD': stochrsi.stoch_pD, 'stochrsi_pK_s_e': stochrsi.stoch_pK_s_e})
    return stochrsi

## TSI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:true_strength_index
def get_tsi(data, period1=13, period2=25):
    pc = data.CLOSE - data.CLOSE.shift(1)
    pc_fs = get_avg(pc, type='SMA', period=period1)
    pc_ss = get_avg(pc_fs, type='SMA', period=period2)
    
    apc = (data.CLOSE - data.CLOSE.shift(1)).abs()
    apc_fs = get_avg(apc, type='SMA', period=period1)
    apc_ss = get_avg(apc_fs, type='SMA', period=period2)
    
    tsi = 100 * pc_ss/apc_ss
    tsi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'tsi':tsi, 'tsi_pc_ss_e':pc_ss})
    return tsi
    
## TRIX ## https://www.tradingview.com/wiki/TRIX
def get_trix(data, period=18):
    ssmma = get_avg(data.CLOSE, type='SMMA', period=period)
    dsmma = get_avg(ssmma, type='SMMA', period=period)
    tsmma = get_avg(dsmma, type='SMMA', period=period)
    
    pdiff = (tsmma - tsmma.shift(1))/tsmma.shift(1)
    
    trix = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'trix':trix, 
                         'trix_ssmma_e':ssmma,
                         'trix_dsmma_e':dsmma,
                         'trix_tsmma_e':tsmma})
    return trix

## UI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index
def get_ui(data, period=14):
    pd = (data.CLOSE - data.CLOSE.rolling(period).max())/data.CLOSE.rolling(period).max()*100
    sa = pd.rolling(14).std()
    sa = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'sa':sa, 'sa_pd':pd})
    return sa

## UO ## https://www.tradingview.com/wiki/Ultimate_Oscillator_(UO)
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

## VI ## https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
def get_vi(data, period=14):
    pVM = (data.HIGH - data.LOW.shift(1)).abs().rolling(period).sum()
    nVM = (data.LOW - data.HIGH.shift(1)).abs().rolling(period).sum()
    
    tr = get_atr(data, average='SMMA', period=period).tr.rolling(period).sum()
    
    pVI = pVM/tr
    nVI = nVM/tr
    
    vi = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'vi_pVI':pVI, 'vi_nVI':nVI,
                      'vi_pVM_e':pVM, 'vi_nVM_e':nVM})
    return vi

## VWAP ## https://www.tradingview.com/wiki/Volume_Weighted_Average_Price_(VWAP)
def get_vwap(data):
    tp = (data.HIGH + data.LOW + data.CLOSE)/3
    tp_v = tp*data.VOLUME
    
    vwap = tp_v.cumsum()/tp.cumsum()
    vwap = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'vwap':vwap, 'vwap_tp_e':tp,
                       'vwap_tp_v_e':tp_v})
    return vwap

## WR ## https://www.tradingview.com/wiki/Williams_%25R_(%25R)
def ger_wr(data, period=30):
    hh = data.HIGH.rolling(period).max()
    ll = data.LOW.rolling(period).min()
    
    wr = (hh - data.CLOSE)/(hh - ll)
    wr = pd.DataFrame({'OPEN_TIME':data.OPEN_TIME, 'wr':wr, 'wr_hh_e':hh, 'wr_ll_e':ll})
    
    return wr
