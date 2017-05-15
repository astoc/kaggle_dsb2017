from __future__ import print_function # for python3 compatibility 
import numpy as np

import pandas as pd
from sklearn.metrics import log_loss

if __name__ == '__main__':
 

    test = pd.read_csv('Shize_DSB_etr_v21_test_5foldAverage.csv')        
    test.sort(['id'], ascending=[True], inplace=True)

    test2 = pd.read_csv('Shize_DSB_etr_v0_test_5foldAverage.csv')        
    test2.sort(['id'], ascending=[True], inplace=True)
    

    test3 = pd.read_csv('Shize_DSB_xgb_v10_xgb_test_5foldAverage.csv')        
    test3.sort(['id'], ascending=[True], inplace=True)  


    test4 = pd.read_csv('Shize_DSB_allFea_xgb_v1_test_5foldAverage.csv')        
    test4.sort(['id'], ascending=[True], inplace=True)    


  
  
    test5 = pd.read_csv('Shize_DSB_xgb_feat3_v5_xgb_test_5foldAverage.csv')        
    test5.sort(['id'], ascending=[True], inplace=True)  
    

    
    a1=0.3  
    a2=0.0  
    a3=0.1  
    a4=0.1  
    a5=0.5   
    
    
    
    B=a1*test['cancer'].astype(float)+a2*test2['cancer'].astype(float)+a3*test3['cancer'].astype(float)+a4*test4['cancer'].astype(float)+a5*test5['cancer'].astype(float)
    
    
    
    B=np.clip(B,0.03,0.97)
            
    B=np.array(B)
    print (B[0:4])
    
    sub=test
    sub['cancer']=B
    sub.to_csv('ensemble_solution_v2.csv', index=False)
    
