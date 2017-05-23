# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:05:39 2017

@author: 91243
"""
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
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
    
    ans_user_item_pair.to_csv('./ensemble_set/ans_{}.csv'.format(train_end),index=False)
    #pred = ans_user_item_pair.sort(columns = 'prob',ascending=False)[:1500]
    #unique_user_id = pred.groupby('user_id',as_index = False)[['prob']].max()
    #pred = pd.merge(unique_user_id,pred,on = ['user_id','prob'],how = 'left')
    #del pred['prob']
    
    #pred['user_id'] = pred['user_id'].astype(int)
    '''
    if not os.path.exists('./model/xgb_train_end.pkl'):
        with open('./model/xgb_train_{}.pkl'.format(train_end),'wb') as f:
            pickle.dump(model,f)
    '''
    
    return

if __name__ == '__main__':
    #ans_user_item_pair,pred = xgb_train()
    end = '2016-04-11'
    
    generate = False
    if generate == True:
        for i in range(7):
            xgb_train(end)
            end = datetime.strptime(end,'%Y-%m-%d') - timedelta(days = 1)
            end = end.strftime('%Y-%m-%d')
    """
    df = pd.DataFrame()
    for name in os.listdir('./ensemble_set/'):
        file_dir = './ensemble_set/'+name
        if df.empty:
            df = pd.read_csv(file_dir)
        else:
            tmp = pd.read_csv(file_dir)
            df = pd.merge(df,tmp,on = ['user_id','sku_id'])
    """
    
    df = pd.DataFrame()
    for name in os.listdir('./ensemble_set/'):
        file_dir = './ensemble_set/'+name
        ans = pd.read_csv(file_dir)
        buyed = ans.sort(columns='prob',ascending=False)[:1500]
        buyed['buyed'] = 1
        ans = pd.merge(ans,buyed,how='left',on = ['user_id','sku_id','prob'])
        ans = ans.fillna(0)
        ans = ans[['user_id','sku_id','prob','buyed']]
        
        if df.empty:
            df = ans
        else:
            df = pd.merge(df,ans,on = ['user_id','sku_id'])
        
        buy = df.iloc[:,[3,5,7,9,11,13,15]]
        prob = df.iloc[:,[2,4,6,8,10,12,14]]

    df['sum_pred'] = buy.apply(lambda x:x.sum(),axis = 1)
    df['sum_prob'] = prob.apply(lambda x:x.sum(),axis = 1)
        
    df = df[['user_id','sku_id','sum_pred','sum_prob']]
    ans = df[df.sum_pred >= 4]

    unique_user_id = ans.groupby('user_id',as_index = False)[['sum_prob']].max()
    ans = pd.merge(unique_user_id,ans,on = ['user_id','sum_prob'],how = 'left')
    
    ans = ans[['user_id','sku_id']]
    ans['user_id'] = ans['user_id'].astype(int)