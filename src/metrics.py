import numpy as np 
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score)
import warnings
warnings.filterwarnings('ignore')


@property
def my_acc_auc(outputs, targets):
    """Computes the accuracy and auc score for multiple binary predictions"""
    y_true = targets.cpu().detach().numpy().squeeze()
    y_pred = outputs.cpu().detach().numpy().squeeze()
   
    y_pred = np.where(y_pred>=0,1,y_pred) # if larger than 0 then convert to 1 (True)
    y_pred  = np.where(y_pred<0,0,y_pred) # if smaller than 0 then convert to 0 (False)

    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall= recall_score(y_true, y_pred)
    
    return acc, f1, precision, recall 
