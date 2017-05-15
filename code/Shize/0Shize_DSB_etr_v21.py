import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

import xgboost as xgb

from sklearn import cross_validation, linear_model
from sklearn.metrics import roc_auc_score  #mean_squared_error
from math import sqrt, pi
from scipy.optimize import minimize
from sklearn import neighbors

from sklearn import svm
#from multilayer_perceptron  import MultilayerPerceptronClassifier #Regressor
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss


import sys

class flushfile():
    def __init__(self, f):
        self.f = f
    def __getattr__(self,name): 
        return object.__getattribute__(self.f, name)
    def write(self, x):
        self.f.write(x)
        self.f.flush()
    def flush(self):
        self.f.flush()
        
        
#from random import random
np.random.seed(8)
print("Start loading data")


df1 = pd.read_csv('feats_keras8_0313_stage1_1595_0411.csv')
df1.columns.values[0]='id' #add "id" colname
df1.to_csv('feats_keras8_0313_stage1_1595_0411.csv', index=False)
df1 = pd.read_csv('feats_keras8_0313_stage1_1595_0411.csv')


#add stage1 public LB solution labels to corresponding records
cv_index=pd.read_csv('cv_index_1952_all.csv') 
cv_index=cv_index[['id','cancer']]
df1 = df1.drop(['cancer'], axis = 1)
df1 = df1.merge(cv_index, how='inner', on='id')


#stage2 data
df1_stage2 = pd.read_csv('feats_keras8_0313_stage2_506_0411.csv')
df1_stage2.columns.values[0]='id' #add "id" colname
df1_stage2.to_csv('feats_keras8_0313_stage2_506_0411.csv', index=False)
df1_stage2 = pd.read_csv('feats_keras8_0313_stage2_506_0411.csv')
#make sure stage2 'cancer' column to also in the last column, be consistent with stage1 data format
label_2=df1_stage2['cancer']
df1_stage2 = df1_stage2.drop(['cancer'], axis = 1)
df1_stage2['cancer']=label_2


#concat stage1 and stage2 data
df1=pd.concat([df1, df1_stage2])
print(df1.shape)

#
df1.columns.values[0]='id' #"id"
df1.to_csv('feats_keras8_0313_stage12.csv', index=False)
df1 = pd.read_csv('feats_keras8_0313_stage12.csv')
print(df1.iloc[0:3,0:5])

df1.sort(['id'], ascending=[True], inplace=True)

#################################################################################


###############################################################################
df = pd.read_csv('feats_keras_0311_stage1_1595_0411.csv')
df.columns.values[0]='id' #add "id" colname
df.to_csv('feats_keras_0311_stage1_1595_0411.csv', index=False)
df = pd.read_csv('feats_keras_0311_stage1_1595_0411.csv')

#add stage1 public LB solution labels
cv_index=pd.read_csv('cv_index_1952_all.csv') 
cv_index=cv_index[['id','cancer']]
df = df.drop(['cancer'], axis = 1)
df = df.merge(cv_index, how='inner', on='id')

#stage2 data
df_stage2 = pd.read_csv('feats_keras_0311_stage2_506_0411.csv')
df_stage2.columns.values[0]='id' #add "id" colname
df_stage2.to_csv('feats_keras_0311_stage2_506_0411.csv', index=False)
df_stage2 = pd.read_csv('feats_keras_0311_stage2_506_0411.csv')
#make sure stage2 'cancer' column to also in the last column, be consistent with stage1 data 
label_2=df_stage2['cancer']
df_stage2 = df_stage2.drop(['cancer'], axis = 1)
df_stage2['cancer']=label_2


#concat stage1 and stage2 data
df=pd.concat([df, df_stage2])

df.columns.values[0]='id' #"id" colname
df.to_csv('feats_keras_0311_stage12.csv', index=False)
df = pd.read_csv('feats_keras_0311_stage12.csv')
print(df.iloc[0:3,0:5])

df.sort(['id'], ascending=[True], inplace=True)


feats6 = ["id", "emphy2_1", "zcenter_2_0.9999999_n0_0", "skewness_2_0.9999999","vol_2_0.9999999"] 
  
df = df[feats6]

######################################################################################


df = df1.merge(df, how='inner', on='id')
print(df.shape, df1.shape)  


#################################
df2 = pd.read_csv('feats_8_stage1_1595_0411.csv')
df2.columns.values[0]='id' #add "id" colname
df2.to_csv('feats_8_stage1_1595_0411.csv', index=False)
df2 = pd.read_csv('feats_8_stage1_1595_0411.csv')

#add stage1 public LB solution labels to corresponding records
cv_index=pd.read_csv('cv_index_1952_all.csv') 
cv_index=cv_index[['id','cancer']]
df2 = df2.drop(['cancer'], axis = 1)
df2 = df2.merge(cv_index, how='inner', on='id')

#stage2 data
df2_stage2 = pd.read_csv('feats_8_stage2_506_0411.csv')
df2_stage2.columns.values[0]='id' #add "id" colname
df2_stage2.to_csv('feats_8_stage2_506_0411.csv', index=False)
df2_stage2 = pd.read_csv('feats_8_stage2_506_0411.csv')
#make sure stage2 'cancer' column to also in the last column, be consistent with stage1 data format
label_2=df2_stage2['cancer']
df2_stage2 = df2_stage2.drop(['cancer'], axis = 1)
df2_stage2['cancer']=label_2


#concat stage1 and stage2 data
df2=pd.concat([df2, df2_stage2])
print(df2.shape)

#
df2.columns.values[0]='id' #"id"
df2.to_csv('feats_8_stage12.csv', index=False)
df2 = pd.read_csv('feats_8_stage12.csv')
print(df2.iloc[0:3,0:5])

df2.sort(['id'], ascending=[True], inplace=True)

df2 = df2.drop(['cancer'], axis = 1)

feat=['id', 'skewness_2_0.999999', 'variance_2_0.999999', 'zrel_2_0.999999_ns','vol_2_0.999999'] 

df2=df2[feat]



df = df.merge(df2, how='inner', on='id')
print(df.shape, df2.shape)  


#################################################################################

###
train_org = df[df.cancer>=0]  
test_org = df[df.cancer<0] 
print("train_org, test_org shape:")
print(train_org.shape, test_org.shape)  


#################################
print ('Start 5fold CV Split:')  

cv_index=pd.read_csv('cv_index_1952_all.csv') 
cv_index=cv_index[['id','cv_k']]


print(train_org.shape)  
train_org = train_org.merge(cv_index, how='inner', on='id')
print(train_org.shape)  

print("CV Split Done.")


#################################
submission = test_org[['id','cancer']]

label=train_org['cancer']
data_index=train_org[['id','cancer', 'cv_k']]

print("Done loading data.")


###############################################################################
###############################################################################

#################################
scores=[0]*5

for z in range(0,6):
    np.random.seed(8)
    
    # load training and test datasets
    train =train_org
    test=test_org
    
    # drop useless columns and create labels
    train = train.drop(['id','cancer','cv_k'], axis = 1)
    test = test.drop(['id','cancer'], axis = 1)
    print("Original feature set size: \n")
    print train.shape, test.shape 
    
    train_x, train_y = train[data_index.cv_k != z], label[data_index.cv_k != z]
    test_x, test_y = train[data_index.cv_k == z], label[data_index.cv_k == z]
    
    print train_x.shape, test_x.shape
    
    print("Done data preprocessing")
    
    #For LB version trained on full data
    if (z==0):
        train_x, train_y =train, label
        test_x, test_y = train[data_index.cv_k == 1], label[data_index.cv_k == 1]
    
  
    # convert data to numpy array   
    train = np.array(train)
    test = np.array(test)  

    # object array to float
    train = train.astype(float)
    test = test.astype(float)
    print train.shape, test.shape
    
    ##For CV
    train_x = np.array(train_x)
    test_x = np.array(test_x)
            
    # object array to float
    train_x = train_x.astype(float)
    test_x = test_x.astype(float)
    print train_x.shape, test_x.shape
    
          

    #Start etr modeling
    np.random.seed(88)
    
    clf2 = ensemble.ExtraTreesClassifier(n_jobs=30,criterion='gini', n_estimators=800, max_depth=20, min_samples_leaf=6 , random_state=8)  #6: 0.443315, #8:0.44368

    if(z==0):
        print('Start model training:')
        num_round=78 #           
        oldstderr = sys.stderr 
        sys.stderr = open('pred/Shize_DSB_etr_v21_LBfullTrain'+'.txt', 'w')
        sys.stderr = flushfile(sys.stderr) 
        clf2.fit(train_x, train_y)
        sys.stderr = oldstderr
        print('End model training:')
        
        print("Start predicting and saving fullTrain LB submissions as a csv file.")
        
        submission['cancer'] = clf2.predict_proba(test)[:,1] 
        
        submission.sort(['id'], ascending=[True], inplace=True)
        submission.to_csv('pred/Shize_DSB_etr_v21_test(FullTrain).csv', index=False)
        
        print("Done save submissions.")
        continue  #z=0: LB version trained on full data, don't need to do following cv prediction
    
    #####################################################
    
    print('Start model training:')
    oldstderr = sys.stderr 
    sys.stderr = open('pred/Shize_DSB_etr_v21_fold'+str(z)+'.txt', 'w')
    sys.stderr = flushfile(sys.stderr) 
    clf2.fit(train_x, train_y)
    sys.stderr = oldstderr
    print('End model training:')
    
    print("Start cv prediction")
    pred = clf2.predict_proba(test_x)[:,1]  
     
    print("Done cv prediction")
    
    print('Save cv predictions')
    cv_pred=data_index[['id']]
    cv_pred=cv_pred[data_index.cv_k == z]
    print(cv_pred.shape)
    
    cv_pred['cancer'] = pred
    
    print(cv_pred.shape)
    
    print("Done save cv predictions.")
    
    print('Shize_DSB_etr_v21 logloss CV score,fold '+str(z)+':')
    print(log_loss(np.array(test_y.astype(int)), np.array(pred.astype(float))))
    scores[z-1]=log_loss(np.array(test_y.astype(int)), np.array(pred.astype(float))) #log_loss(test_y, pred)
    print("check logloss function:")
    print(log_loss(test_y, pred))
    scores[z-1]=log_loss(test_y, pred)

    
    if(z==1):
        full_cv_pred=cv_pred
    else:
        frames=[full_cv_pred, cv_pred]
        full_cv_pred=pd.concat(frames)
    print("cv_pred shape:" )
    print(full_cv_pred.shape)
    
    if(z==5):
        full_cv_pred.sort(['id'], ascending=[True], inplace=True)
        full_cv_pred.to_csv('pred/Shize_DSB_etr_v21_cv_pred_full.csv', index=False)    
        
        print("Done save full cv predictions.")
        
    
    #continue

    print("Start LB prediction")
    if (z==1):
        pred_LB = clf2.predict_proba(test)[:,1]
    else:
        pred_LB = pred_LB+clf2.predict_proba(test)[:,1]
    print('Done LB prediction.')
    
    

    if(z==5):
        print("Start saving submissions as a csv file.")
        print(submission.shape)
        submission['cancer'] = pred_LB/5.0         
        
        submission.sort(['id'], ascending=[True], inplace=True)
        submission.to_csv('pred/Shize_DSB_etr_v21_test_5foldAverage.csv', index=False)

        print("Done save submissions.")
        
        print("5folds logloss cv scores detail:")
        print(scores)
        print("Average logloss:")
        print(1/5.0*(scores[0]+scores[1]+scores[2]+scores[3]+scores[4]))
        
        

        
##########