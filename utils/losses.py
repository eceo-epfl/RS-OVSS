import torch.nn as nn
import torch
import torch.nn.functional as F
from yacs.config import CfgNode as CN  
from numpy.random import choice 
import numpy as np
import os

class CELoss(nn.Module):
    """Wrapper around the cross entropy loss"""
    def __init__(self,ignore_index = -99):
        super().__init__()
        
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none' )

    def forward(self, outputs, targets,return_correct =False):    
         
        loss =self.CELoss(outputs , targets )
        loss = loss.mean()
        
        if return_correct:
            # Calculate the number of correct predictions
            predicted_labels = torch.argmax(outputs, dim=1)
            correct = predicted_labels == targets
            correct = correct.sum()/correct.numel()
            
            return loss, correct.item()
        return  loss
    



class BcosLossWAugm(nn.Module):
    
    def __init__(self,contrastive_cfg,  n_class, device,ignore_index = -99 ) -> None:
        """
        Implementation of the L_{B-cos} with EDA augmentation.
        contrastive_cfg (dict) : values to defining the contrastive setting such as file path, dimension etc.
        n_class (int): number of classes
        device : 'cpu' or 'cuda:0'
        ignore_index (int) : Index of background class, default: -99 (no background class)
        """    
        super(BcosLossWAugm,self).__init__()
        print('Init  Bcos Loss with augmentations')
        self.cfg = contrastive_cfg
        self.tao = contrastive_cfg.TEMPERATURE 
        self.ignore_index = ignore_index
        self.num_queries = contrastive_cfg.NUM_QUERIES # number of vectors to samples per batch
        print('\tNum queries', self.num_queries)
        self.n_class=n_class[0]  
        self.device =device 
        self.augm_vectors = None
        
        
        #Load the prototypes for standard contrastive loss from files :
        embedding_fn = contrastive_cfg.EMBEDDING_PATH  
        self.prototypes = torch.load(embedding_fn ).to(self.device) 
        self.prototypes.requires_grad_(False)        
        print('\tLoad prototypes for contrative loss from :', embedding_fn.split('/')[-1]   )
        
        
        # Setup the text augmentation framework
        if contrastive_cfg.USE_AUGMENTATION :
            self.get_augmented_vectors_setup (self.cfg) 
        else :
            print('\tNo use of EDA text augmentation')
                

        # set up Bcos loss
        self.B =contrastive_cfg.BCOS_VALUE
        if self.B == -1 :            
            self.B = nn.Parameter(torch.tensor(1.1,requires_grad=True).to(self.device))
            self.B.requires_grad = True
            print('b-cos loss with learnable B parameters of initial value', self.B.item())

         
    def get_augmented_vectors_setup (self,cfg):
            """ 
            Build two dictionnaries that contains each the indices of the 
            augmented (pos or neg) vectors for each category
            Create 2 variables :
            augm_vectors :tensors with embeddings of augmented text per class size :(num_vectors,emb_dim,)
            positive_idx_per_cls : dict with classes as keys, and values are the index of 
                                    the vectors of a specific class in  augm_vectors
            """
            
            # Load augmented vectors from files :
            augm_vectors = torch.load(cfg.AUGMENTED_PATH )
            assert augm_vectors.shape[1] == cfg.DIMENSION, f"{augm_vectors.shape[1]} is not equal to {cfg.DIMENSION}" 
            self.augm_vectors = augm_vectors.requires_grad_(False).to(self.device)                   

            # Compute the number of augmented vector per category:
            nb_augm_vectors = self.augm_vectors.shape[0]
            n_class = self.n_class 
            nb_augm_vectors_per_cls = int(nb_augm_vectors / (n_class -1   ))   # -1 for background removal

            
            # Get the positive indices  indices for the augmented vectors for each class 
            self.positive_idx_per_cls = dict()
            
            # Loop over the classes and build the dictionnary : 
            for cls in range(n_class):
                
                if cls == self.ignore_index : # Background categories are ignored
                    self.positive_idx_per_cls[cls] = []
                    continue
                
                # Get min and max index per category :
                # we assume ignore cls is 0
                min_idx, max_idx = (cls -1 )*nb_augm_vectors_per_cls , cls *nb_augm_vectors_per_cls 
                self.positive_idx_per_cls[cls] = torch.arange(min_idx, max_idx)

            
            print('\tLoad  augmented vectors for contrative loss from :', cfg.AUGMENTED_PATH.split('/')[-1],'\tWith shape : ',self.augm_vectors.shape,)
            if self.cfg.NB_POS_VECTORS >0 : 
                print( '\tUse ', nb_augm_vectors_per_cls,'augmented vector per class. ',self.cfg.NB_POS_VECTORS  )                   
            if self.cfg.NB_NEG_VECTORS >0 : 
                print('\tUse ',len(self.positive_idx_per_cls[cls]),'vector for negative  augmentation',self.cfg.NB_NEG_VECTORS )

   
    def forward(self, y_pred,y_true ):  # forward method from the Bcos loss            
        
        
        # Random sampling of pixel where loss will be computed :  
        y_true = y_true.flatten() 
        random_indices = torch.randperm(n=y_true.size(0), requires_grad=False)[:self.num_queries]
        random_indices,_  = torch.sort(random_indices)
        sampled_y_true = y_true[random_indices]     
            
        # Get count of pixel per class :
        unique_cls, counts_cls = torch.unique(sampled_y_true , return_counts = True)  
        unique_cls, counts_cls = unique_cls.tolist(), counts_cls.tolist()
            
        # Flatten y_pred  : Bx emb_dim xWxH --> ( BxWxH) x emb_dim
        y_pred = y_pred['contrastive']   
        y_pred = y_pred.moveaxis(1,-1).flatten(end_dim=-2)
        sampled_y_pred = y_pred[random_indices,:]  
        
        losses = torch.tensor(0.0).to(self.device)    
            
        # Loop over the classes :
        for cls,  num_cls_pix in zip ( unique_cls, counts_cls) : 
            
            if cls == self.ignore_index : # ignore background pixel
                continue 
                    
            # Get positive and negatives representations from prototypes :
            # Bcos loss : similar to the network paramters W
            all_feat = self._get_all_features (cls,num_cls_pix)
                              
            # Get the anchor features = predicted values for the pixels of interest
            # Bcos loss : similar to the network paramters x
            cls_mask = (sampled_y_true == cls).flatten()
            anchor_feat = sampled_y_pred[cls_mask].unsqueeze(1)
            
            
            # Compute B-cos loss:
            # no need to normalized w since they were originally normalized
            cos = torch.cosine_similarity (anchor_feat, all_feat,dim=-1 )
            sign = torch.sign(cos)
            norm_features = torch.norm(anchor_feat) 
            logits = norm_features * torch.pow(torch.abs(cos),self.B)*sign
 
            targets = torch.zeros(num_cls_pix).long().to(self.device)  
            
            #Compute log softmax over the logits and target :
            losses += F.cross_entropy( logits / self.tao,  targets ) 

                    
        if self.augm_vectors is not  None  :
            return losses 
        else :             
            return  losses / len(unique_cls)
    

    @torch.no_grad()
    def _get_all_features (self,class_id,num_cls_pix) :
        """ return a tensor containing the 1 positive feature at index 0 and one
         negative features for each negatives class.
        Shape of the output : num_class x embedding size 
        """
    
        with torch.no_grad():
                            
            # Pick the positive vector for the given class :  
            # ! class id 1 has prototype in position 0 
            positive_feat = self.prototypes[class_id-1,:].unsqueeze(0) 

            # Get the negative from all prototypes except target class :
            neg_indices = list(range(self.n_class))
            neg_indices = neg_indices.remove(class_id)
            negative_feat = self.prototypes [ neg_indices,:].squeeze() 
        
            # Use augmented vectors:
            if self.augm_vectors is not  None  : 
                
                if self.cfg.NB_POS_VECTORS >0 :
                    # Get some random augmented version of the positive vector
                    pos_indices  = self.positive_idx_per_cls[class_id].numpy()
                    random_pos_idx = choice(a= pos_indices,size=self.cfg.NB_POS_VECTORS ) # usually  NB_POS_VECTORS =1
                    positive_feat = self.augm_vectors[random_pos_idx,:]  

                
                if self.cfg.NB_NEG_VECTORS >0 :
                    neg_indices= np.array([])
                    # Get 1 augmented negative index per class : 
                    for k in range(1,self.n_class):
                        if k == class_id : # 
                            continue
                        neg_idx = self.positive_idx_per_cls[k].numpy()
                        random_neg_idx = choice(a= neg_idx ,size=self.cfg.NB_NEG_VECTORS)                        
                        neg_indices  = np.append( neg_indices, random_neg_idx,axis=0)
                        
                    # get the negatives vectors values
                    negative_feat = self.augm_vectors[neg_indices,:]                   
            

            # Concatenate positive and negative features, with positive feature first :                    
            all_feat = torch.cat( (positive_feat,negative_feat), dim =0).unsqueeze(0)
            all_feat = all_feat.repeat(num_cls_pix, 1, 1)

            
        return all_feat       




class WrapperBcosLossWAugm(nn.Module):

    def __init__(self,n_class,device,embedding_path,temperature = 1.5,  bcos_value = 1,ignore_index=0,
                  use_augmentation =True,nb_pos_vectors=1,nb_neg_vectors =10,
                  augmented_path=None,embedding_size=None
                 ):
        """Simple Wrapper around the B_cos loss                
        """
        super(WrapperBcosLossWAugm,self).__init__()
        contrastive_cfg =  CN()
        contrastive_cfg.TEMPERATURE = temperature
        contrastive_cfg.USE_AUGMENTATION = use_augmentation
        contrastive_cfg.NUM_QUERIES = 1000
        contrastive_cfg.USE = True
        contrastive_cfg.EMBEDDING_PATH = embedding_path
        contrastive_cfg.BCOS_VALUE = bcos_value
        contrastive_cfg.NB_POS_VECTORS = nb_pos_vectors
        contrastive_cfg.NB_NEG_VECTORS = nb_neg_vectors
        contrastive_cfg.AUGMENTED_PATH = augmented_path
        contrastive_cfg.DIMENSION = embedding_size
        
        self.bcos_loss = BcosLossWAugm(contrastive_cfg, [n_class], device, ignore_index) 
        self.embedding_vectors = self.bcos_loss.prototypes.transpose(1,0).to(device)
        self.norm_vector = torch.linalg.norm(self.embedding_vectors ,dim=0)
       
        
    def forward(self, y_pred,y_true, return_correct = False ):   
        loss =  self.bcos_loss({'contrastive':y_pred}, y_true)
        
        if return_correct:
            with torch.no_grad():
                # Calculate the number of correct predictions
                # Compute cosine similarity as dot product and norms
                b,c,h,w = y_pred.shape
                data = y_pred.moveaxis(1,-1).reshape(-1,c)
                dot_product = torch.einsum('nc,ck->nk',
                                            data, 
                                            self.embedding_vectors )
                norm = torch.einsum( 'n,k->nk', 
                                    torch.linalg.norm(data,dim=-1) , 
                                    self.norm_vector )       
                predictions = torch.argmax (dot_product / norm , dim=1) 
        
                # Reshape and add 1 to taken into account the background class :
                predictions = predictions.reshape(b,h,w) +1 
                correct = predictions  == y_true
                correct = correct.sum()/correct.numel()
            return loss, correct.item()
        
        else :
            return loss

if __name__ == '__main__':
    import sys
    sys.path.append('/home/valerie/Projects/contrastive-segmentation/contrastive-lc/losses/')

    criterion = WrapperBcosLossWAugm(n_class=13,
                                    device='cuda:0',
                                    embedding_path ='/home/valerie/Projects/contrastive-segmentation/contrastive-lc/data/flair_labels/flair_descrip_clip.pt' ,
                                    temperature = 1, 
                                    bcos_value = 1,
                                    ignore_index=0,
                                    use_augmentation =True,
                                    nb_pos_vectors=1,
                                    nb_neg_vectors =10,
                                    augmented_path='/home/valerie/Projects/contrastive-segmentation/contrastive-lc/data/flair_labels/augm/flair_descrip_augmented_clip.pt',
                                    embedding_size=512,
                               )
    logits = torch.rand([4,512,256,256]).cuda()
    targets = torch.randint(13,[4,256,256]).cuda()
    
    loss = criterion(logits,targets, return_correct = False)
    print(loss)
    