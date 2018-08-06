import numpy as np
import pandas as pd
import collections as col

df = pd.read_csv('ETHBTC.txt', index_col=0)
df.OPEN_TIME = (df.OPEN_TIME/1000).astype(int)
df.OPEN_TIME = df.OPEN_TIME.astype(str)
df.head()

def ohlc_feats(data):
    o = data.OPEN.values
    h = data.HIGH.values
    l = data.LOW.values
    c = data.CLOSE.values
    
    ohlc_feats = pd.DataFrame({})
    
    ohlc_feats['OPEN_TIME'] = data.OPEN_TIME.values
    
    ohlc_feats['OHLC'] = h - o + h - l + c - l
    ohlc_feats['OLHC'] = o - l + h - l + h - c
    
    ohlc_feats['O_EQUAL_C'] = o == c
    ohlc_feats['O_EQUAL_L'] = o == l
    ohlc_feats['O_EQUAL_H'] = o == h
    ohlc_feats['C_EQUAL_H'] = c == h
    ohlc_feats['C_EQUAL_L'] = c == l
    ohlc_feats['L_EQUAL_H'] = l == h
    
    ohlc_feats['O_GREATER_C'] = o > c
    
    ohlc_feats['O_C_MEAN'] = (o + c)/2
    ohlc_feats['L_H_MEAN'] = (l + h)/2
    
    ohlc_feats['O_OC_MEAN_FRAC'] = o / ohlc_feats['O_C_MEAN']
    ohlc_feats['L_OC_MEAN_FRAC'] = l / ohlc_feats['O_C_MEAN']
    ohlc_feats['H_OC_MEAN_FRAC'] = h / ohlc_feats['O_C_MEAN']
    ohlc_feats['C_OC_MEAN_FRAC'] = c / ohlc_feats['O_C_MEAN']
    
    ohlc_feats['O_LH_MEAN_FRAC'] = o / ohlc_feats['L_H_MEAN']
    ohlc_feats['L_LH_MEAN_FRAC'] = l / ohlc_feats['L_H_MEAN']
    ohlc_feats['H_LH_MEAN_FRAC'] = h / ohlc_feats['L_H_MEAN']
    ohlc_feats['C_LH_MEAN_FRAC'] = c / ohlc_feats['L_H_MEAN']
    
    ohlc_feats['O_GREATER_LH_MEAN'] = o > ohlc_feats['L_H_MEAN']
    ohlc_feats['C_GREATER_LH_MEAN'] = c > ohlc_feats['L_H_MEAN']
    
    ohlc_feats['O_C_MEAN__L_H_MEAN__DIFF']     = ohlc_feats['O_C_MEAN'] - ohlc_feats['L_H_MEAN']
    ohlc_feats['O_C_MEAN__L_H_MEAN__DIFF_ABS'] = np.abs(ohlc_feats['O_C_MEAN'] - ohlc_feats['L_H_MEAN'])
    ohlc_feats['O_C_MEAN__L_H_MEAN__GREATER']  = ohlc_feats['O_C_MEAN'] > ohlc_feats['L_H_MEAN']
    
    ohlc_feats['O_L_DIFF'] = o - l
    ohlc_feats['O_H_DIFF'] = o - h
    ohlc_feats['C_L_DIFF'] = c - l
    ohlc_feats['C_H_DIFF'] = c - h
    ohlc_feats['O_C_DIFF'] = o - c
    ohlc_feats['L_H_DIFF'] = l - h
    
    ohlc_feats['O_L_DIFF_ABS'] = np.abs(o - l)
    ohlc_feats['O_H_DIFF_ABS'] = np.abs(o - h)
    ohlc_feats['C_L_DIFF_ABS'] = np.abs(c - l)
    ohlc_feats['C_H_DIFF_ABS'] = np.abs(c - h)
    ohlc_feats['O_C_DIFF_ABS'] = np.abs(o - c)
    ohlc_feats['L_H_DIFF_ABS'] = np.abs(l - h)
    
    ohlc_feats['O_C_MEAN_PERCENTILE']     = (ohlc_feats['O_C_MEAN'] - l)/(h - l)
    ohlc_feats['O_PERCENTILE']            = (o - l)/(h - l)
    ohlc_feats['C_PERCENTILE']            = (c - l)/(h - l)
    ohlc_feats['O_C_PERCENTILE_DIFF']     = ohlc_feats['O_PERCENTILE'] - ohlc_feats['C_PERCENTILE']
    ohlc_feats['O_C_PERCENTILE_DIFF_ABS'] = np.abs(ohlc_feats['O_PERCENTILE'] - ohlc_feats['C_PERCENTILE'])
    
    return ohlc_feats
    
def ohlc_cross_feats(data):
    o = data.OPEN.values
    h = data.HIGH.values
    l = data.LOW.values
    c = data.CLOSE.values
    
    ohlc_feats = pd.DataFrame({})
    
    ohlc_feats['OPEN_TIME'] = data.OPEN_TIME.values[1:]
    
    ohlc_feats['O_O1_EQUAL'] = o[1:] == o[:-1]
    ohlc_feats['H_H1_EQUAL'] = h[1:] == h[:-1]
    ohlc_feats['L_L1_EQUAL'] = l[1:] == l[:-1]
    ohlc_feats['C_C1_EQUAL'] = c[1:] == c[:-1]
    
    ohlc_feats['O_O1_GREATER'] = o[1:] > o[:-1]
    ohlc_feats['O_H1_GREATER'] = o[1:] > h[:-1]
    ohlc_feats['O_L1_GREATER'] = o[1:] > l[:-1]
    ohlc_feats['O_C1_GREATER'] = o[1:] > c[:-1]
    
    ohlc_feats['H_O1_GREATER'] = h[1:] > o[:-1]
    ohlc_feats['H_H1_GREATER'] = h[1:] > h[:-1]
    ohlc_feats['H_L1_GREATER'] = h[1:] > l[:-1]
    ohlc_feats['H_C1_GREATER'] = h[1:] > c[:-1]
    
    ohlc_feats['L_O1_GREATER'] = l[1:] > o[:-1]
    ohlc_feats['L_H1_GREATER'] = l[1:] > h[:-1]
    ohlc_feats['L_L1_GREATER'] = l[1:] > l[:-1]
    ohlc_feats['L_C1_GREATER'] = l[1:] > c[:-1]
    
    ohlc_feats['C_O1_GREATER'] = c[1:] > o[:-1]
    ohlc_feats['C_H1_GREATER'] = c[1:] > h[:-1]
    ohlc_feats['C_L1_GREATER'] = c[1:] > l[:-1]
    ohlc_feats['C_C1_GREATER'] = c[1:] > c[:-1]
    
    ohlc_feats['O_O1_ABS_PERC_DIFF'] = (o[1:] - o[:-1]) / o[:-1]
    ohlc_feats['O_H1_ABS_PERC_DIFF'] = (o[1:] - h[:-1]) / h[:-1]
    ohlc_feats['O_L1_ABS_PERC_DIFF'] = (o[1:] - l[:-1]) / l[:-1]
    ohlc_feats['O_C1_ABS_PERC_DIFF'] = (o[1:] - c[:-1]) / c[:-1]
    
    ohlc_feats['H_O1_ABS_PERC_DIFF'] = (h[1:] - o[:-1]) / o[:-1]
    ohlc_feats['H_H1_ABS_PERC_DIFF'] = (h[1:] - h[:-1]) / h[:-1]
    ohlc_feats['H_L1_ABS_PERC_DIFF'] = (h[1:] - l[:-1]) / l[:-1]
    ohlc_feats['H_C1_ABS_PERC_DIFF'] = (h[1:] - c[:-1]) / c[:-1]
    
    ohlc_feats['L_O1_ABS_PERC_DIFF'] = (l[1:] - o[:-1]) / o[:-1]
    ohlc_feats['L_H1_ABS_PERC_DIFF'] = (l[1:] - h[:-1]) / h[:-1]
    ohlc_feats['L_L1_ABS_PERC_DIFF'] = (l[1:] - l[:-1]) / l[:-1]
    ohlc_feats['L_C1_ABS_PERC_DIFF'] = (l[1:] - c[:-1]) / c[:-1]
    
    ohlc_feats['C_O1_ABS_PERC_DIFF'] = (c[1:] - o[:-1]) / o[:-1]
    ohlc_feats['C_H1_ABS_PERC_DIFF'] = (c[1:] - h[:-1]) / h[:-1]
    ohlc_feats['C_L1_ABS_PERC_DIFF'] = (c[1:] - l[:-1]) / l[:-1]
    ohlc_feats['C_C1_ABS_PERC_DIFF'] = (c[1:] - c[:-1]) / c[:-1]
    
    ohlc_feats['OC_ABS_PERC_GREATER'] = np.abs(o[1:] - c[1:]) > np.abs(o[:-1] - c[:-1])
    ohlc_feats['LH_ABS_PERC_GREATER'] = np.abs(l[1:] - h[1:]) > np.abs(l[:-1] - h[:-1])
    
    ohlc_feats['LH_ENGULF'] = (l[1:] < l[:-1]) & (h[1:] > h[:-1])
    ohlc_feats['LH_ENGULFED'] = (l[1:] > l[:-1]) & (h[1:] < h[:-1])
    
    dummy_feats = ohlc_feats.iloc[0:0]
    
    dummy_feats = ohlc_feats.iloc[0:0]
    dummy_feats.loc[0, :] = [np.nan for i in range(len(dummy_feats.columns))]
    ohlc_feats = dummy_feats.append(ohlc_feats, ignore_index=True)
    
    return ohlc_feats

ohlc_feat = ohlc_feats(df)
ohlc_cross_feat = ohlc_cross_feats(df)

ohlc_feat.head()
ohlc_cross_feat.head()

# all_feats = pd.concat([ohlc_feat, ohlc_cross_feat], axis=1)
all_feats = pd.merge(ohlc_feat, ohlc_cross_feat, how='inner', on='OPEN_TIME')
all_feats.shape
all_feats.head()

Y = all_feats.loc[:, 'O_GREATER_C'].values[1:]
X = all_feats.values[:-1]

X.shape
Y.shape

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, criterion='gini', n_jobs=-1, max_features="auto", random_state=0, verbose=1)
model = model.fit(X[:300000, 1:], Y[:300000])

Y_pred = model.predict(X[300000:, 1:])
(Y[300000:] == Y_pred).sum()

df_ = df.loc[:, ['OPEN_TIME', 'OPEN', 'CLOSE']]
df_.head()

signal = pd.DataFrame({'OPEN_TIME': X[300000:, 0].astype(int).astype(str), 'ACTION': [-1 if x else 1 for x in Y_pred]})
signal.head()

def buy_sell_signal_returns(data, signal):
    # data contains (atleast): OPEN_TIME, OPEN, CLOSE
    # singal : OPEN_TIME, ACTION {1 buy, 0 hold, -1 sell}
    
    data_signal_df = pd.merge(df_, signal, on='OPEN_TIME')
    data_signal_df.head()
    data_signal_df.shape

    returns = (data_signal_df.ACTION*(data_signal_df.CLOSE - data_signal_df.OPEN)).sum()
    return returns
    
window_size = 5

X_ = np.zeros((X.shape[0] - window_size + 1, X.shape[1]*window_size))
Y_ = np.zeros((X.shape[0] - window_size + 1, ))

for i in range(window_size):
    X_[:, i*X.shape[1]:(i+1)*X.shape[1]] = X[i:X.shape[0]-window_size+i+1]

Y_ = Y[window_size-1:]

model_ = RandomForestClassifier(n_estimators=1, criterion='gini', n_jobs=-1, max_features=50, random_state=0, verbose=1)
model_ = model_.fit(X_[:300000], Y_[:300000])
