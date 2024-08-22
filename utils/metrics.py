import torch
import numpy as np
from time import time
import time
from sklearn.metrics import confusion_matrix
from pandas import DataFrame as pd_Dataframe


class SegmentationMetrics:
    """Store and compute accuracy metrics for evaluation of semantic segmentation 
        (Use only for a single inference pass)
    """
    
    def __init__(self,classes_dict, ignore_index=None):
        
        self.batch_metrics = []
        if ignore_index in (classes_dict.keys()):
            _ = classes_dict.pop(ignore_index)
           # print('Init segmentation metrics : ', classes_dict)
        self.categories = list(classes_dict.values())
        self.categories_id = list(classes_dict.keys())
        self.categories_id = [int(x) for x in self.categories_id]
        self.nb_class = len(self.categories_id)
        self.confusion_matrix = np.zeros(shape=[self.nb_class,self.nb_class])
        self.num_batch = 0
        self.ignore_index = ignore_index
        
    def zero_metrics(self):
        self.confusion_matrix = np.zeros(shape=[self.nb_class,self.nb_class])
        self.num_batch = 0
        
    def add(self, y_true, y_pred):
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Remove background pixel and predictions :        
        if self.ignore_index is not None  :
            background_mask = y_true!= self.ignore_index
            y_true = y_true[background_mask]
            y_pred = y_pred[background_mask]     
        
        cm = confusion_matrix( y_true, y_pred, labels = self.categories_id)          
    
        self.confusion_matrix  += cm 
          
        self.num_batch +=1
        
    
    def get_IoU(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) -TP
        FN = np.sum(self.confusion_matrix, axis=1) -TP
        
        IoU = np.divide(TP, TP + FP +FN  , 
                            out=np.zeros_like(TP), 
                            where=(TP+FP +FN )!=0)      

        return IoU
    
    def get_IoU_dict(self, val=False ):
        class_iou = self.get_IoU()
        classes = self.categories
        if val :
            classes = ['val_'+c for c in classes]
        else :
            classes = ['test_'+c for c in classes]
            
        class_iou = {c:iou for c,iou in zip ( classes,class_iou)  }
        return class_iou
        
        
    def get_overall_accuracy(self):
        TP = np.diag(self.confusion_matrix)
        global_accuracy = TP.sum()/ self.confusion_matrix.sum()
        return   global_accuracy
    
        
    def get_metrics(self):
        
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) -TP
        FN = np.sum(self.confusion_matrix, axis=1) -TP
        
        IoU = np.divide(TP, TP + FP +FN  , 
                            out=np.zeros_like(TP), 
                            where=(TP+FP +FN )!=0)      
        precision = np.divide(TP, TP + FP , 
                            out=np.zeros_like(TP), 
                            where=(TP+FP)!=0)
        recall = np.divide(TP, TP+FN , 
                            out=np.zeros_like(TP), 
                            where=(TP+FN)!=0)
        f1_score = np.divide( 2 *precision* recall, precision + recall , 
                            out=np.zeros_like(TP), 
                            where=(precision + recall )!=0)
        
        support = np.sum(self.confusion_matrix, axis=1)
        global_accuracy = TP.sum()/ self.confusion_matrix.sum()
        
        final_metrics = {
            'id': self.categories_id,
            'IoU' : np.round( IoU*100,2),
            'precision' :  np.round( precision*100,2),
            'recall' :  np.round( recall*100,2),
            'f1_score' :  np.round( f1_score*100,2),
            'support' :  np.round( support,2),
            'global_acc':  np.round( global_accuracy*100,2),
            'mIoU' :  np.round( IoU.mean()*100, 2),
            'timestamp' : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }  
        
        
        results =  pd_Dataframe(final_metrics, index=self.categories)
        
        print()
        print('Global accuracy:', results['global_acc'].iloc[0]  , '%'  )
        print('Mean IoU: ', results['mIoU'].iloc[0] ,'%' )
        
        print('\n\n',results.drop(columns=['global_acc','mIoU']))
        print('*'*80)
        
        return results
        
