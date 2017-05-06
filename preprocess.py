
# coding: utf-8

# # 京东JData算法大赛-高潜用户购买意向预测
# ## 方法：特征工程+模型

# 简介:数据预处理，主要使用python+pandas库，将原始数据进行one-hot encoding处理，以用户-商品对为基本单元，完成对原始的行为数据的合并，以及各个统计特征的采集。

# ## 目录：
# [1.商品，用户，评论数据预处理](#preprocess1)

# <a id='preprocess1'></a>
# ### 商品，用户，评论数据预处理
# 使用独热编码的技巧，合适的表达了并没有大小联系的特征

# In[16]:

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import os
import math


# In[2]:

def get_userData():
    path = './generated_data/userData.csv'
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
        userData = pd.read_csv('./data/JData_User.csv',encoding = 'gbk')
        userData.age = userData.age.apply(map_age)
        
        age_df = pd.get_dummies(userData["age"], prefix="age")
        sex_df = pd.get_dummies(userData["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(userData["user_lv_cd"], prefix="user_lv_cd")
        userData = pd.concat([userData['user_id'], age_df, sex_df, user_lv_df], axis=1)
        userData.to_csv(path,index = False)
        
    return userData



# In[5]:

def get_productData():
    path = './generated_data/productData.csv'
    if os.path.exists(path):
        productData = pd.read_csv(path)
    else:
        productData = pd.read_csv('./data/JData_Product.csv')
        attr1_df = pd.get_dummies(productData["a1"], prefix="a1")
        attr2_df = pd.get_dummies(productData["a2"], prefix="a2")
        attr3_df = pd.get_dummies(productData["a3"], prefix="a3")
        productData = pd.concat([productData[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        productData.to_csv(path,index = False)
        
    return productData




# In[7]:

def get_commentsData():
    path = './generated_data/commentsData.csv'
    if os.path.exists(path):
        commentsData = pd.read_csv(path)
        
    else:
        df = pd.read_csv('./data/JData_Comment.csv')
        #df = df[df.dt < end]
        comments = df.groupby('sku_id',as_index = False).last() 
        dummies_index = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments,dummies_index],axis = 1)
        commentsData = comments.drop(['dt','comment_num'],axis = 1)
        commentsData.to_csv(path,index = False)
        
    return commentsData




# 获得给定时间范围内用户对某商品的各种行为（浏览，加购物车，删购，下单，关注，点击行为）的统计求和特征

# In[13]:

def get_all_action():
    path = './generated_data/all_action.csv'
    if os.path.exists(path):
        all_action = pd.read_csv(path)
        
    else:
        action1 = pd.read_csv('./data/JData_Action_201602.csv')
        action2 = pd.read_csv('./data/JData_Action_201603.csv')
        action3 = pd.read_csv('./data/JData_Action_201604.csv')
        all_action = pd.concat([action1,action2,action3])
        #all_action = all_action.drop_duplicates()
        all_action.to_csv(path,index = False)
        
    return all_action

def get_action(start,end):
    path = './generated_data/action_{}_{}.csv'.format(start,end)
    if os.path.exists(path): 
         action = pd.read_csv(path)
    else:
        action = get_all_action()
        action = action[(action.time > start) & (action.time < end)]
        action.to_csv(path,index = False)
        
    return action

def get_action_feat(start,end):
    path = './generated_data/action_feat_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = get_action(start,end)
        action = action[['user_id','sku_id','type']]
        df = pd.get_dummies(action['type'],prefix  = 'action_{}_{}'.format(start,end))
        action = pd.concat([action, df], axis=1)
        action = action.groupby(['user_id','sku_id'],as_index = False).sum()
        del action['type']
        action.to_csv(path,index = False)
        
    return action


# 加上时间权重的个行为求和，时间衰退函数为exp(-x)，其中x为行为发生的时间与预测日期的差值，单位为天

# In[15]:

def get_action_weight_feat(start,end):
    path = './generated_data/action_weight_feat_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
        
    else:
        action = get_action(start,end)
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

        action = action.drop(['time','model_id','type','weight'],axis = 1)
        action = action.groupby(['user_id','sku_id','cate','brand'],as_index = False).sum()
        action.to_csv(path,index = False)
        
    return action


# 用户的各种行为与最终购买的转化率,此处设计一个平滑函数，避免除0操作

# In[38]:

def get_user_action_ratio(start,end):
    path = './generated_data/user_action_ratio_{}_{}.csv'.format(start,end)

    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = get_action(start,end)
        df = pd.get_dummies(action['type'],prefix = 'action')
        action = pd.concat([action['user_id'],df],axis = 1)
        action = action.groupby('user_id',as_index = False).sum()
        action['user_action_1_ratio'] = action['action_4'] / action['action_1']
        action['user_action_2_ratio'] = action['action_4'] / action['action_2']
        action['user_action_3_2_ratio'] = action['action_3'] / action['action_2']
        action['user_action_5_ratio'] = action['action_4'] / action['action_5']
        action['user_action_6_ratio'] = action['action_4'] / action['action_6']
        
        action = action.replace(np.inf,1)
        action = action.fillna(0)
        
        action['user_buy_browse_ratio'] =  (np.log(1 + action['action_4']) - np.log(1 + action['action_1'])).map(lambda x: '%.2f' % x)
        action['user_buy_addcart_ratio'] = (np.log(1 + action['action_4']) - np.log(1 + action['action_2'])).map(lambda x: '%.2f' % x)
        action['user_buy_follow_ratio'] = (np.log(1 + action['action_4']) - np.log(1 + action['action_5'])).map(lambda x: '%.2f' % x)
        action['user_buy_click_ratio'] = (np.log(1 + action['action_4']) - np.log(1 + action['action_6'])).map(lambda x: '%.2f' % x)
        action['user_delcart_addcart_ratio'] = (np.log(1 + action['action_3']) - np.log(1 + action['action_2'])).map(lambda x: '%.2f' % x)
        
        action = action.drop(['action_1','action_2','action_3','action_4','action_5','action_6'],axis = 1)
        action.to_csv(path,index = False)
        
    return action


# In[39]:

action1 = get_user_action_ratio(start = '2016-04-10',end = '2016-04-16')
action1[:5]


# 商品的各种行为与最终的购买转化率，同样的，也使用上述平滑函数

# In[42]:

def get_product_action_ratio(start,end):
    path = './generated_data/product_action_ratio_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = get_action(start,end)
        df = pd.get_dummies(action['type'],prefix = 'action')
        action = pd.concat([action['sku_id'],df],axis = 1)
        action = action.groupby('sku_id',as_index = False).sum()
        action['product_action_1_ratio'] = action['action_4']/action['action_1']
        action['product_action_2_ratio'] = action['action_4']/action['action_2']
        action['product_action_3_ratio'] = action['action_4']/action['action_3']
        action['product_action_5_ratio'] = action['action_4']/action['action_5']
        action['product_action_6_ratio'] = action['action_4']/action['action_6']
        
        action = action.replace(np.inf,1)
        action = action.fillna(0)
        
        action['product_buy_browse_ratio'] =  (np.log(1 + action['action_4']) - np.log(1 + action['action_1'])).map(lambda x: '%.2f' % x)
        action['product_buy_addcart_ratio'] = (np.log(1 + action['action_4']) - np.log(1 + action['action_2'])).map(lambda x: '%.2f' % x)
        action['product_buy_follow_ratio'] = (np.log(1 + action['action_4']) - np.log(1 + action['action_5'])).map(lambda x: '%.2f' % x)
        action['product_buy_click_ratio'] = (np.log(1 + action['action_4']) - np.log(1 + action['action_6'])).map(lambda x: '%.2f' % x)
        action['product_delcart_addcart_ratio'] = (np.log(1 + action['action_3']) - np.log(1 + action['action_2'])).map(lambda x: '%.2f' % x)
        
        action = action.drop(['action_1','action_2','action_3','action_4','action_5','action_6'],axis = 1)
        action.to_csv(path,index = False)
        
    return  action


# In[44]:

action1 = get_product_action_ratio(start = '2016-04-10',end = '2016-04-16')
action1[:5]


# 商品热度特征，规定时间内，与商品产生交互的用户的数量特征

# In[65]:

def get_product_extra_data(start,end):
    path = './generated_data/product_extra_data_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        action = get_action(start,end)
        df = pd.get_dummies(action['type'],prefix = 'action')
        action = pd.concat([action['sku_id'],df],axis = 1)
        action['cnt'] = 1
        action = action.groupby(['sku_id'],as_index = False).sum()
        action['trend_{}_{}'.format(start,end)] = action['action_1']*0.1 + action['action_2']*0.6 + action['action_4']*1 + action['action_5']*0.3 + action['action_6']*0.1
        action = action[['sku_id','cnt','trend_{}_{}'.format(start,end)]]
        action.to_csv(path,index = False)
    return action




# In[67]:

def get_user_rank_for_item(start,end):
    path = './generated_data/user_rank_for_item_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        col = ['user_id','sku_id','type_1','type_2','type_3','type_4','type_5','type_6']
        df = get_action_weight_feat(start,end)
        df = df.drop(['cate','brand'],axis = 1)
        df.columns = col
        df['sort_type_1'] = df['type_1'].groupby(df['user_id']).rank(method = 'min',ascending = False)
        df['sort_type_2'] = df['type_2'].groupby(df['user_id']).rank(method = 'min',ascending = False)
        df['sort_type_3'] = df['type_3'].groupby(df['user_id']).rank(method = 'min',ascending = False)
        df['sort_type_4'] = df['type_4'].groupby(df['user_id']).rank(method = 'min',ascending = False)
        df['sort_type_5'] = df['type_5'].groupby(df['user_id']).rank(method = 'min',ascending = False)
        df['sort_type_6'] = df['type_6'].groupby(df['user_id']).rank(method = 'min',ascending = False)
        
        date = '_{}_{}'.format(start,end)
        col_finnal = [i+date for i in ['sort_1','sort_2','sort_3','sort_4','sort_5','sort_6']]
        col_finnal = ['user_id','sku_id'] + col_finnal        

        df = df[['user_id','sku_id','sort_type_1','sort_type_2',
                 'sort_type_3','sort_type_4','sort_type_5','sort_type_6']]
        df.columns = col_finnal
        df.to_csv(path,index = False)
        
    return df

def get_item_rank_for_user(start,end):
    path = './generated_data/item_rank_for_user_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        col = ['user_id','sku_id','type_1','type_2','type_3','type_4','type_5','type_6']
        df = get_action_weight_feat(start,end)
        df = df.drop(['cate','brand'],axis = 1)
        df.columns = col
        
        df['sort_type_1'] = df['type_1'].groupby(df['sku_id']).rank(method = 'min',ascending = False)
        df['sort_type_2'] = df['type_2'].groupby(df['sku_id']).rank(method = 'min',ascending = False)
        df['sort_type_3'] = df['type_3'].groupby(df['sku_id']).rank(method = 'min',ascending = False)
        df['sort_type_4'] = df['type_4'].groupby(df['sku_id']).rank(method = 'min',ascending = False)
        df['sort_type_5'] = df['type_5'].groupby(df['sku_id']).rank(method = 'min',ascending = False)
        df['sort_type_6'] = df['type_6'].groupby(df['sku_id']).rank(method = 'min',ascending = False)
        
        date = '_{}_{}'.format(start,end)
        col_finnal = [i+date for i in ['isort_1','isort_2','isort_3','isort_4','isort_5','isort_6']]
        col_finnal = ['user_id','sku_id'] + col_finnal 

        df = df[['user_id','sku_id','sort_type_1','sort_type_2',
                 'sort_type_3','sort_type_4','sort_type_5','sort_type_6']]
        df.columns = col_finnal
        df.to_csv(path,index = False)
    return df


# In[68]:

def get_label(start):
    end = datetime.strptime(start,'%Y-%m-%d') + timedelta(days = 5)
    end = end.strftime('%Y-%m-%d')
    path = './generated_data/label_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
        
    else:
        action = get_action(start,end)
        action = action[action.type == 4]
        action = action.groupby(['user_id','sku_id'],as_index = False).first()
        action['label'] = 1
        action = action[['user_id','sku_id','label']]
        action.to_csv(path,index = False)
        
    return action


# In[69]:

def get_offline_train_set(end):
    start = (datetime.strptime(end,'%Y-%m-%d')-timedelta(days = 7)).strftime('%Y-%m-%d')
    path = './generated_data/train_set_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    else:
        start_days = "2016-02-01"
        label_start = end
        
        userData = get_userData()
        productData = get_productData()
        product_extra_data = get_product_extra_data(start,end)
        commentsData = get_commentsData()
        user_action_ratio = get_user_action_ratio(start_days,end)
        product_action_ratio = get_product_action_ratio(start_days,end)
        
        label = get_label(label_start)
        action = get_action_weight_feat(start,end)
        
        for i in (1, 2, 3, 4, 5, 6, 7):
            start_days = datetime.strptime(end, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            action = pd.merge(action,get_user_rank_for_item(start_days,end),how = 'left',on = ['user_id','sku_id'])
            
            action = pd.merge(action,get_item_rank_for_user(start_days,end),how = 'left',on = ['user_id','sku_id'])
        
        for i in (1, 2, 3, 4, 5, 6, 7):
            start_days = datetime.strptime(end, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            action = pd.merge(action,get_action_feat(start_days,end),how = 'left',on = ['user_id','sku_id'])
            
        action = pd.merge(action,userData,how = 'left',on = 'user_id')
        action = pd.merge(action,productData,how = 'left',on = 'sku_id')
        action = pd.merge(action,commentsData,how = 'left',on = 'sku_id')
        action = pd.merge(action,user_action_ratio,how = 'left',on = 'user_id')
        action = pd.merge(action,product_action_ratio,how = 'left',on = 'sku_id')
        action = pd.merge(action,product_extra_data,how = 'left',on = 'sku_id')
        
        action = pd.merge(action,label,how = 'left',on = ['user_id','sku_id'])
        action = action.fillna(0)
        action = action[action.cate_x == 8]
        
        action.to_csv(path,index = False)
        
    user_item_pair = action[['user_id','sku_id']]
    label = action['label']
    
    action = action.drop(['user_id','sku_id','label'],axis = 1)

    return user_item_pair,action,label


# In[70]:

def get_test_set(end = '2016-04-16'):
    start = (datetime.strptime(end,'%Y-%m-%d')-timedelta(days = 7)).strftime('%Y-%m-%d')
    path = './generated_data/test_set_{}_{}.csv'.format(start,end)
    path2 = './generated_data/train_set_{}_{}.csv'.format(start,end)
    if os.path.exists(path):
        action = pd.read_csv(path)
    elif os.path.exists(path2):
        action = pd.read_csv(path2)
        action = action.drop(['label'],axis = 1)
        action = action[action.cate_x == 8]
        action.to_csv(path,index = False)
    else:
        start_days = "2016-02-01"
        
        #start = (datetime.strptime(end,'%Y-%m-%d') - timedelta(days = 30)).strftime('%Y-%m-%d')
        
        userData = get_userData()
        productData = get_productData()
        product_extra_data = get_product_extra_data(start,end)
        commentsData = get_commentsData()
        user_action_ratio = get_user_action_ratio(start_days,end)
        product_action_ratio = get_product_action_ratio(start_days,end)
        
        action = get_action_weight_feat(start,end)
        
        for i in (1, 2, 3, 4, 5, 6, 7):
            start_days = datetime.strptime(end, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            action = pd.merge(action,get_user_rank_for_item(start_days,end),how = 'left',on = ['user_id','sku_id'])
            
            action = pd.merge(action,get_item_rank_for_user(start_days,end),how = 'left',on = ['user_id','sku_id'])
        
        for i in (1, 2, 3, 4, 5, 6, 7):
            start_days = datetime.strptime(end, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            action = pd.merge(action,get_action_feat(start_days,end),how = 'left',on = ['user_id','sku_id'])
            
        action = pd.merge(action,userData,how = 'left',on = 'user_id')
        action = pd.merge(action,productData,how = 'left',on = 'sku_id')
        action = pd.merge(action,commentsData,how = 'left',on = 'sku_id')
        action = pd.merge(action,user_action_ratio,how = 'left',on = 'user_id')
        action = pd.merge(action,product_action_ratio,how = 'left',on = 'sku_id')
        action = pd.merge(action,product_extra_data,how = 'left',on = 'sku_id')

        action = action.fillna(0)
        action = action[action.cate_x == 8]
        
        action.to_csv(path,index = False)
        
    user_item_pair = action[['user_id','sku_id']]
    action = action.drop(['user_id','sku_id'],axis = 1)
    
    return user_item_pair,action




