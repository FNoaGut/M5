# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:18:18 2020

@author: PACO
"""


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.chdir('C:/Users/laguila/Google Drive/ARC_KAGGLE/m5-datos')
files = []
#for dirname, _, filenames in os.walk('/kaggle/input'):
for dirname, _, filenames in os.walk('C:/Users/laguila/Google Drive/ARC_KAGGLE/m5-datos'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

pd.options.display.max_columns = 50

h = 28 
max_lags = 14
tr_last = 1913
fday = datetime(2016,4, 25) 
fday

def create_dt(is_train = True, nrows = None, first_day = 1200):
    #prices
    prices = pd.read_csv("sell_prices.csv", dtype = PRICE_DTYPES)
    #calendar        
    cal = pd.read_csv("calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    
    #validation
    dt = pd.read_csv("sales_train_validation.csv", nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    dt.drop(["weekday", "wm_yr_wk", "year", "month"], axis=1, inplace = True)

    dt['store_id'] = dt['store_id'].map(lambda x: x[-1:])
    dt['dept_id'] = dt['dept_id'].map(lambda x: x[-1:])
    dt['item_id'] = dt['item_id'].map(lambda x: x[-3:]).astype('int16')
    
    columns = [ 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in columns: 
        dt[col] = dt[col].cat.codes.astype("int16")
        dt[col] -= dt[col].min()
    
    #
    
    dept_id = pd.get_dummies(dt.dept_id)
    idds = dept_id.columns
    dept_id.columns = [f"dept_{idd}" for idd in idds ]
    #dt=pd.concat([dt, dept_id], axis=1)
    
    store_id = pd.get_dummies(dt.store_id)
    idds = store_id.columns
    store_id.columns = [f"store_{idd}" for idd in idds ]
    dt=pd.concat([dt, store_id, dept_id, pd.get_dummies(dt.cat_id), pd.get_dummies(dt.state_id)], axis=1)
    
    dt.drop(["cat_id", "state_id", "dept_id", "store_id", 'd'], axis=1, inplace = True)
    
    features = {
        "wday": "weekday",
        "week": "weekofyear",
        "quarter": "quarter",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

    #dt.drop(["date"], axis=1, inplace = True)
    
    floats = ["snap_CA", "snap_TX", "snap_WI"]
    for el in floats:
        dt[el] = dt[el].astype("int16")
        
    return dt
    


def create_fea(dt):
    lags = [2, 7]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [2, 7]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean().round(2))

    

    
            
            

FIRST_DAY = 1800 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk

df = create_dt(is_train=True, first_day= FIRST_DAY)

create_fea(df)
df.dropna(inplace = True)


cat_feats = ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "sales", "date"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X = df[train_cols]
y = df["sales"]

np.random.seed(767)

test_inds = np.random.choice(X.index.values, round(0.25*X.shape[0]), replace = False)
train_inds = np.setdiff1d(X.index.values, test_inds)

X_train = X.loc[train_inds,]
X_test = X.loc[test_inds,]
y_train = y.loc[train_inds]
y_test = y.loc[test_inds]


del df, X, y, test_inds,train_inds ; gc.collect()

#%% LGBM

from time import time
t = time()
params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.073,
        "sub_row" : 0.73,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,
        'num_iterations' : 1150,
        'num_leaves': 124,
        "min_data_in_leaf": 100,
        "n_jobs" : -1
}
lgb_ = lgb.LGBMRegressor()
lgb_.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
print(t-time())

#%% XGBoost
import xgboost as xgb
params2 = {
        "n_estimators" : 50,
        #"max_depth" :,
        "learning_rate" : 0.1,
        "verbosity" : 1,
        "booster" : "gblinear",
        "n_jobs" : -1,
        #"min_child_weight " :,
        'reg_alpha ': 0.01,
        'reg_lambda ' : 0,
        'random_state': 124
}
t = time()
xgb_ = xgb.XGBRegressor()
xgb_.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
print(time()-t)
evals_result = xgb.evals_result()

#%%Voting regressor
xgb_ = xgb.XGBRegressor()
lgb_ = lgb.LGBMRegressor()
from sklearn.ensemble import VotingRegressor
vot = VotingRegressor([('xgb', xgb_), ('lgb', lgb_)])
vot.fit(X_train, y_train)

#%% Stacked model

estimators = [('xgb', xgb.XGBRegressor()),
              ('lgb', lgb.LGBMRegressor())]

from sklearn.ensemble import GradientBoostingRegressor #Para pegar todos juntos
from sklearn.ensemble import StackingRegressor

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(random_state=42))

reg.fit(X_train, y_train)


#%% Ranomized search

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

light = lgb.LGBMRegressor(num_iterations=15, objective = "poisson", silent = False, seed = 10)
from scipy.stats import uniform
params = {
         "boosting_type" : ['gbdt', 'rf'],
        #"objective" : ["poisson"],
        "learning_rate" : uniform(loc=0.05, scale=0.5),
        # 'num_leaves': [100, 120, 140],
         'min_child_samples ': [10, 20, 30],
        # "n_estimators" : [100, 120, 140]
        # 'reg_alpha' : uniform(loc=0.05, scale=1),
        # 'reg_lambda ' : uniform(loc=0.05, scale=1),
}
clf = RandomizedSearchCV(estimator = light, param_distributions = params, n_iter = 10, n_jobs=-1, cv = 2, verbose = 1)
print("Training...")
search3 = clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=5)

search3.best_params_

#%% Ver las distribuciones de cada feature y transformaciones




#%%Preprocesamiento

pipelines
quitar variables poca importancia
quitar correlacionadas
lda/PCA/...
elastic net
stochastic gradient descent
svm



permutation importance and correlated features
validation and learning curve

#%% GUARDAR MODELOS

from joblib import dump, load
dump(reg, 'stack.joblib') 

#%% PREDICCION
te = create_dt(False)
cols = [f"F{i}" for i in range(1,29)]

for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        create_fea(tst)
        tst = tst.loc[tst.date == day , train_cols]
        te.loc[te.date == day, "sales"] = search3.predict(tst) # magic multiplier by kyakovlev


#%% MAGIC MULTIPLIER KYAKOVLEV
alphas = [1.025, 1.023, 1.0175]
alphas = [1]
weights = [1/len(alphas)]*len(alphas)
sub = 0.


for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

   te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h),
#                                                                           "id"].str.replace("validation$", "evaluation")
   te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
   te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
   te_sub.fillna(0., inplace = True)
   te_sub.sort_values("id", inplace = True)
   te_sub.reset_index(drop=True, inplace = True)
   #te_sub.to_csv(f"submission_{icount}.csv",index=False)
   if icount == 0 :
       sub = te_sub
       sub[cols] *= weight*alpha
   else:
       sub[cols] += te_sub[cols]*weight*alpha
   print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission.csv",index=False)

sub.head(10)
sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape


