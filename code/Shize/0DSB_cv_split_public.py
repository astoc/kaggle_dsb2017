
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



####################################
np.random.seed(1952)

print("Start loading data")


print ('Start 5fold CV Split:')  
#cv_index=train_org[['id','cancer']]
cv_index=pd.read_csv('stage1_solution.csv') 

skf = StratifiedKFold(cv_index.cancer, n_folds=5, shuffle=True, random_state=1952) #1952) #8)
cv_index['cv_k'] = 0
for k, (_, valid_index) in enumerate(skf, 1):
    cv_index.iloc[valid_index, -1] = k

cv_index=cv_index[['id','cancer','cv_k']]
cv_index.to_csv('cv_index_1952_public.csv', index=False)


cv_index1=pd.read_csv('cv_index_1952.csv') 

cv_index_all = pd.concat([cv_index1, cv_index])
cv_index_all.to_csv('cv_index_1952_all.csv', index=False)

#cv_index=cv_index[['id','cv_k']]
print(cv_index_all.iloc[0:5,:])


print("Done")