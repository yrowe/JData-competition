# -*- coding: utf-8 -*-
"""
Created on Mon May 08 12:03:04 2017

@author: 91243
"""

#构建用户模型，预测哪些用户会进行购买操作
#userData,近7天各种行为加权和，近1-7天各行为和，各种转换率,7天内有过交互行为的商品个数，
#从结果分析 需不需要过滤异常用户

import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
import preprocess as pp
import math
import xgboost as xgb

def get_user_model_userData(end):
    path = './generated_data/user_model_userData_{}.csv'.format(end)
    if os.path.exists(path):
        userData = pd.read_csv(path)
    else:
        def map_age(age):
            if age == '-1':
                return 0
            if age == u'15岁以下':
                return 1
            if age == u'16-25岁':
                return 2
            if age == u'26-35岁':
                return 3
            if age == u'36-45岁':
                return 4
            if age == u'46-55岁':
                return 5
            if age == u'56岁以上':
                return 6
            return -1
            
        def tranform_user_regtime(df):
            if (df >= 0) & (df < 10):
                df = 0
            elif (df >= 10) & (df < 30):
                df = 1
            elif (df >= 30) & (df < 60):
                df = 2
            elif (df >= 60) & (df < 120):
                df = 3
            elif (df >= 120) & (df < 360):
                df = 4
            elif (df >= 360):
                df = 5
            else:
                df = -1
            return df
            
            
        userData = pd.read_csv('./data/JData_User.csv',encoding = 'gbk')
        userData.age = userData.age.apply(map_age)
        
        userData['user_reg_tm'] = pd.to_datetime(userData['user_reg_tm'])
        userData = userData[userData['user_reg_tm'] <= end]

        END_DATE = datetime.strptime(end, "%Y-%m-%d")
        
        userData['user_reg_tm'] = (END_DATE - userData['user_reg_tm']).map(lambda x: x.days)
        userData['user_reg_tm'] = userData['user_reg_tm'].map(tranform_user_regtime)
        reg_df = pd.get_dummies(userData['user_reg_tm'], prefix="reg_time")
        
        age_df = pd.get_dummies(userData["age"], prefix="age")
        sex_df = pd.get_dummies(userData["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(userData["user_lv_cd"], prefix="user_lv_cd")
        userData = pd.concat([userData['user_id'], age_df, sex_df, user_lv_df,reg_df], axis=1)
        userData.to_csv(path,index = False)
        
    return userData

def get_user_interactive_num(start,end):
    path = './generated_data/user_interactive_num_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = pp.get_action(start,end)
        action['inter_num'] = 1
        action = action[['user_id','inter_num']]
        action = action.groupby('user_id',as_index = False).sum()
        action.to_csv(path,index = False)
        
    return action
    
def get_user_action_with_weight(start,end):
    path = './generated_data/user_action_with_weight_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = pp.get_action(start,end)
        df = pd.get_dummies(action['type'],prefix = 'action')
        action = pd.concat([action,df],axis = 1)
        action['weight'] = action['time'].map(lambda x: datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        
        action['weight'] = action['weight'].map(lambda x: math.exp(-x.days))
    
        action['action_1'] = action['action_1'] * action['weight']
        action['action_2'] = action['action_2'] * action['weight']
        action['action_3'] = action['action_3'] * action['weight']
        action['action_4'] = action['action_4'] * action['weight']
        action['action_5'] = action['action_5'] * action['weight']
        action['action_6'] = action['action_6'] * action['weight']

        action = action.drop(['time','model_id','type','weight','sku_id','cate','brand'],axis = 1)
        action = action.groupby(['user_id'],as_index = False).sum()
        action.to_csv(path,index = False)
        
    return action
    
def get_user_label(start):
    end = datetime.strptime(start,'%Y-%m-%d') + timedelta(days = 5)
    end = end.strftime('%Y-%m-%d')
    path = './generated_data/user_label_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
        
    else:
        action = pp.get_action(start,end)
        action = action[action.type == 4]
        action = action.groupby(['user_id'],as_index = False).first()
        action['label'] = 1
        action = action[['user_id','label']]
        action.to_csv(path,index = False)
        
    return action
    
def get_user_action_feat(start,end):
    path = './generated_data/user_action_feat_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = pp.get_action(start,end)
        action = action[['user_id','type']]
        df = pd.get_dummies(action['type'],prefix  = 'user_action_{}_{}'.format(start,end))
        action = pd.concat([action, df], axis=1)
        action = action.groupby(['user_id'],as_index = False).sum()
        del action['type']
        action.to_csv(path,index = False)
        
    return action
    
        
def get_user_model(end_date = '2016-04-16'):
    start = (datetime.strptime(end_date,'%Y-%m-%d')-timedelta(days = 7)).strftime('%Y-%m-%d')
    start_very_beginning = '2016-02-01'
    path = './generated_data/user_model_{}_{}.csv'.format(start,end_date)
    
    data = get_user_model_userData(end_date)
    
    #最近7天行为习惯
    user_ratio = pp.get_user_action_ratio(start,end_date)
    data = pd.merge(data,user_ratio,on = 'user_id',how = 'right')
    
    #历史行为习惯
    user_ratio2 = pp.get_user_action_ratio(start_very_beginning,end_date)
    data = pd.merge(data,user_ratio2,on = 'user_id')
    
    #7天内有过交互商品数量
    inter_num = get_user_interactive_num(start,end_date)
    data = pd.merge(data,inter_num,on = 'user_id')
    
    #7天各行为（包括删购行为）的时间衰减加权和
    user_action_with_weight = get_user_action_with_weight(start,end_date)
    data = pd.merge(data,user_action_with_weight,on = 'user_id')
    
    #1-7天用户6种行为累加和
    for i in (1, 2, 3, 4, 5, 6, 7):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        data = pd.merge(data,get_user_action_feat(start_days,end_date),how = 'left',on = ['user_id'])
    
    
    #是否购买
    label = get_user_label(end_date)
    data = pd.merge(data,label,on = 'user_id',how = 'left')
    data = data.fillna(0)
    
    data.to_csv(path,index = False)
    
    #向外输出user_id,features,label
    user = data[['user_id']]
    label = data[['label']]
    data = data.drop(['user_id','label'],axis = 1)
    
    
    return user,data,label
    
def get_user_test_set(end_date):
    start = (datetime.strptime(end_date,'%Y-%m-%d')-timedelta(days = 7)).strftime('%Y-%m-%d')
    start_very_beginning = '2016-02-01'
    path = './generated_data/user_model_test_{}_{}.csv'.format(start,end_date)
    
    data = get_user_model_userData(end_date)
    
    #最近7天行为习惯
    user_ratio = pp.get_user_action_ratio(start,end_date)
    data = pd.merge(data,user_ratio,on = 'user_id',how = 'right')
    
    #历史行为习惯
    user_ratio2 = pp.get_user_action_ratio(start_very_beginning,end_date)
    data = pd.merge(data,user_ratio2,on = 'user_id')
    
    #7天内有过交互商品数量
    inter_num = get_user_interactive_num(start,end_date)
    data = pd.merge(data,inter_num,on = 'user_id')
    
    #7天各行为（包括删购行为）的时间 衰减加权和
    user_action_with_weight = get_user_action_with_weight(start,end_date)
    data = pd.merge(data,user_action_with_weight,on = 'user_id')
    
    #1-7天用户6种行为累加和
    for i in (1, 2, 3, 4, 5, 6, 7):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        data = pd.merge(data,get_user_action_feat(start_days,end_date),how = 'left',on = ['user_id'])
    
    data.to_csv(path,index = False)
    
    #向外输出user_id,features,label
    user = data[['user_id']]
    data = data.drop(['user_id'],axis = 1)
    
    return user,data
    
def user_model_train(train_end,test_end):
    user,train_set,label = get_user_model(train_end)
    X_train, X_test, y_train, y_test = train_test_split(train_set.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label = y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    
    param = {'learning_rate' : 0.1, 
             'n_estimators': 1000, 
             'max_depth': 3, 
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
    
    del user,train_set,label,dtrain,dtest,X_train, X_test, y_train, y_test
    
    ans_user,ans_train_set = get_user_test_set(test_end)
    trainning_set = xgb.DMatrix(ans_train_set.values)
    
    y = model.predict(trainning_set,ntree_limit=model.best_iteration)
    ans_user['prob'] = y
    
    pred = ans_user.sort(columns = 'prob',ascending=False)[:1500]
    unique_user_id = pred.groupby('user_id',as_index = False)[['prob']].max()
    pred = pd.merge(unique_user_id,pred,on = ['user_id','prob'],how = 'left')
    del pred['prob']
    pred['user_id'] = pred['user_id'].astype(int)
    
    return ans_user,pred
    
def count_score(pred,label_start):
    actual1 = pp.get_label(label_start)

    num_pred = pred.index.size
    
    
    for i in range(10):
        actual = actual1.sample(frac = 0.5)
    
        num_actual = actual.index.size
        tp1 = pd.merge(actual,pred,on = ['user_id']).index.size
        p1 = tp1*1.0/num_pred
        r1 = tp1*1.0/num_actual
    
        
        f11 = (6*p1*r1)/(p1+5*r1)
        print 'No.{}'.format(i)
        print 'num_pred',num_pred
        print 'num_actual',num_actual
        print '正确命中用户数个数',tp1
        print 'f11 = {}'.format(f11)
    
#data = get_user_model(end_date = '2016-04-03')

train_end = '2016-04-09'
test_end = '2016-04-16'

#train_end = '2016-04-02'
#test_end = '2016-04-09'
ans_user,pred = user_model_train(train_end,test_end)

#count_score(pred,test_end)