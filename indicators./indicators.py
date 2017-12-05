# Stochastic - display the location of the close compared to the high/low range over a user defined number of periods

## Overbought/Oversold
# Overbought - when SO crosses upper threshold, say 80
# Oversold - when SO crosses lower threshold, say 20

## Divergence
# Bullish - when price records a lower low, but Stochastic records a higher low
# Bearish - when price records a higher high, but Stochastic records a lower high

## Bull/Bear Setups
# Bullish - price records a lower high, but Stochastic records a higher high
# 

def get_stoch(chart_timeseries_data, period=14, k_period=3):
    k = 100 * (
        chart_timeseries_data.close - 
        chart_timeseries_data.low.rolling(period).apply(np.min)
    )/(
        chart_timeseries_data.high.rolling(period).apply(np.max) - 
        chart_timeseries_data.low.rolling(period).apply(np.min)
    )
    d = pd.Series(k).emw(span=k_period).mean()
    
    return k, d

# TRIX - comprised of the rate of change of a triple exponentially smoothed moving average

def get_trix(chart_timeseries_data, period=18):
    single_ema = chart_timeseries_data.close.emw(span=period).mean()
    double_ema = pd.Series(single_ema).emw(span=period).mean()
    triple_ema = pd.Series(double_ema).emw(span=period).mean()
    
    trix = (triple_ema - triple_ema.shift(1))/triple_ema.shift(1) * 100
    
    return trix
    
# Ultimate Oscillator - measure momentum across 3 varying timeframes
# - tries to reduce false divergence signals

## Divergences
# Bullish - price makes a lower low while UO makes a higher low
#         - low of the uo below 30
#         - uo breaks above the high of divergence
# Bearish - price makes a higher high while UO makes a lower high
#         - high of the uo above 70
#         - uo falls below the low of divergence

def get_uo(chart_timeseries_data, period1=7, period2=14, period3=28):
    bp = chart_timeseries_data.close - np.minimum(chart_timeseries_data.low, chart_timeseries_data.close.shift(1))
    tr = np.maximum(chart_timeseries_data.high, chart_timeseries_data.close.shift(1)) - np.minimum(chart_timeseries_data.low, chart_timeseries_data.close.shift(1))
    
    avg1 = bp.rolling(period1).sum()/tr.rolling(period1).sum()
    avg2 = bp.rolling(period2).sum()/tr.rolling(period2).sum()
    avg3 = bp.rolling(period3).sum()/tr.rolling(period3).sum()
    
    uo = 100 * (
        float(period1)/(period1+period2+period3)*avg1 + 
        float(period2)/(period1+period2+period3)*avg2 + 
        float(period3)/(period1+period2+period3)*avg3
    )
    
    return uo

# Volume Weighted Average Price - average price weighted by volume
# - Keep in mind however, that much like a moving average, VWAP can also experience lag. 

## Trend IDentification
# Bullish Trend is characterized by prices trading above the VWAP.
# Bearish Trend is characterized by prices trading below the VWAP.
# Sideways Market is characterized by prices trading above and below the VWAP.

def get_vwap(chart_timeseries_data):
    tp = (chart_timeseries_data.high + chart_timeseries_data.low + chart_timeseries_data.close)/3
    vwap = np.cumsum(tp * chart_timeseries_data.quoteVolume) / np.sumsum(chart_timeseries_data.quoteVolume)
    
    return vwap

# Williams %R - comparison between the current close and the highest high for a user defined look back period
# Oscillates between 0 an -100

## Overbought/Oversold
# Overbought - When WR is in 0 to -20
# Oversold - When WR is in -80 to -100

## Momentum Failures
# Momentum failure occur when %R readings reach overbought or oversold conditions 
# for an extended period of time. Upon leaving overbought/oversold territory, 
# %R makes a move back towards the overbought/oversold levels but fails to 
# re-enter the territory.

def get_wr(chart_timeseries_data, period=20):
    highest_high = chart_timeseries_data.high.rolling(period).apply(np.max)
    lowest_low   = chart_timeseries_data.low.rolling(period).apply(np.min)
    
    wr = (highest_high - chart_timeseries_data.close) / (highest_high - lowest_low) * -100
    
    return wr
