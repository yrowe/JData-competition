# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:52:24 2017

@author: 91243
"""

import pandas as pd
#import os
import preprocess as pp
import xgboost as xgb
#import pickle
from sklearn.model_selection import train_test_split

def xgb_train(train_end = '2016-04-11'):

    #train_start = '2016-03-10'
    
    print train_end
    
    user_item_pair,train_set,label = pp.get_offline_train_set(train_end)
    
    X_train, X_test, y_train, y_test = train_test_split(train_set.values, label.values, test_size=0.2, random_state=0)
    
    dtrain=xgb.DMatrix(X_train, label = y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    
    param = {'learning_rate' : 0.1, 
             'n_estimators': 1000, 
             'max_depth': 4, 
             'min_child_weight': 9, 
             'gamma': 0, 
             'subsample': 0.7, 
             'colsample_bytree': 0.6,
             'scale_pos_weight': 1, 
             'eta': 0.05, 
             'silent': 1, 
             'objective': 'binary:logistic',
             'seed':0,
             'nthread':8,
             'eval_metric':'logloss'
            }
    
    evallist = [(dtrain,'train'),(dtest,'test')]
    plst = param.items()
    num_round = 300
    
    model = xgb.train(plst, dtrain, num_round, evallist,early_stopping_rounds = 50)
    
    del user_item_pair,train_set,label,dtrain,dtest,X_train, X_test, y_train, y_test
    
    ans_user_item_pair,ans_train_set = pp.get_test_set()
    
    trainning_set = xgb.DMatrix(ans_train_set.values)
    
    y = model.predict(trainning_set,ntree_limit=model.best_iteration)
    
    ans_user_item_pair['prob'] = y
    
    pred = ans_user_item_pair.sort(columns = 'prob',ascending=False)[:1500]
    unique_user_id = pred.groupby('user_id',as_index = False)[['prob']].max()
    pred = pd.merge(unique_user_id,pred,on = ['user_id','prob'],how = 'left')
    del pred['prob']
    
    pred['user_id'] = pred['user_id'].astype(int)
    '''
    if not os.path.exists('./model/xgb_train_end.pkl'):
        with open('./model/xgb_train_{}.pkl'.format(train_end),'wb') as f:
            pickle.dump(model,f)
    '''
    
    return ans_user_item_pair,pred

if __name__ == '__main__':
    ans_user_item_pair,pred = xgb_train()