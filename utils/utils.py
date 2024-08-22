import torch
import numpy as np
import csv
import os
from torch import nn
from transformers import SegformerForSemanticSegmentation
import json
from yacs.config import CfgNode as cfg 
from utils.SegformerModel import ContrastiveSegformer
from utils.DeepLabv3pModel import DeepLabv3plus, ContrastiveDeepLabv3plus
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def parse_args():
    parser = ArgumentParser(description='Train Segformer network')

    parser.add_argument("--cfg",
                        help="decide which cfg file to use",
                        required=False,
                        default='debug',
                        type=str)

    args = parser.parse_args()
    return args 


def load_config_from_json(config_file_path):
    
    # Check if the config file exists
    if os.path.isfile(config_file_path):
        # Load the JSON data from the file
        with open(config_file_path, "r") as config_file:
            config_dict = json.load(config_file)

        # Convert the dictionary to a YACS configuration object<
        cfg_obj = cfg.CfgNode(config_dict)
        
        # Now you can access the configuration parameters
        print('config sucessfullly loaded from ',config_file_path)
    else:
        print("Config file does not exist.")
        raise NameError
    return cfg_obj


def load_model (model_type='segformer' , n_class=13, from_file = '', freeze_encoder = False , embedding_size =768):
    """
    Loads and returns a semantic segmentation model based on the specified arguments.

    Args:
    - model_type (str): Type of model to load. Options include 'segformer', 
      'segformer_contrastive', 'deeplabv3plus', and 'deeplabv3plus_contrastive'.
    - n_class (int): Number of output classes for the segmentation task. Default is 13 (FLAIR dataset).
    - from_file (str): Path to the  pretrained model weights. 
      If provided, weights will be loaded into the model.
    - freeze_encoder (bool): If True, freezes the encoder layers of the model. 
      Only the decoder layers will remain trainable.
    - embedding_size (int): Size of the embedding vector for contrastive models. Default is 768 (SentenceBERT).

    Returns:
    - model (torch.nn.Module): The loaded and configured model.
    """
    
    if model_type == 'segformer' :
        print('Load Segformer model.')
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels = n_class)
        
        if os.path.exists(from_file) and not os.path.isdir(from_file) and from_file != '':
            weights = torch.load(from_file)
            mis_keys, un_keys = model.load_state_dict(weights, strict=True)
            assert len(mis_keys) == 0 and len(un_keys) == 0, "Missing or unexpected keys when loading pretrained weights"
            print( f'\tModel weights sucessfully loaded from {from_file}')
                    
        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.decode_head.classifier.parameters():
                param.requires_grad = True
            print( '\tModel encoder is frozen' )
        
    elif model_type == 'segformer_contrastive' :
        print('Load CONTRASTIVE Segformer model from scratch')
        model =  ContrastiveSegformer(embedding_size=embedding_size,
                                      from_pretrained=from_file,
                                      freeze_encoder=freeze_encoder)
        
    elif model_type == 'deeplabv3plus':
        model = DeepLabv3plus(num_classes=n_class)
        print('Load DeepLabV3plus model.')
        
        if os.path.exists(from_file) and not os.path.isdir(from_file) and from_file != '':
            weights = torch.load(from_file)
            mis_keys, un_keys = model.load_state_dict(weights['model_state_dict'], strict=True)
            assert len(mis_keys) == 0 and len(un_keys) == 0, "Missing or unexpected keys when loading pretrained weights"
            print( f'Model weights sucessfully loaded from {from_file}')     
        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            print( 'Model encoder is frozen')       
        
    
    elif model_type == 'deeplabv3plus_contrastive':
        print('Load CONTRASTIVE DeepLabV3plus model.')
        model = ContrastiveDeepLabv3plus(num_classes=n_class, embedding_size=embedding_size, from_file=from_file)    
        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False                
            for param in model.contrastive_classifier.parameters():
                param.requires_grad = True
            print( 'Model encoder is frozen')

    else : 
        raise NotImplementedError# f'{model_type} model does not exist.'
    return model

def optimizer_factory(optimizer_type: str, model: nn.Module, lr,) -> torch.optim.Optimizer:
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(),lr=lr )
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(),lr=lr )
    else:
        raise NotImplementedError()


def scheduler_factory(    scheduler_type, optimizer, max_epochs) :
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer)
    elif scheduler_type == "stepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size= max_epochs//3 ,gamma=0.3)
    elif scheduler_type == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=160000,power=2)

    else:
        raise NotImplementedError()
    

def save_module(model: nn.Module, path: str, mode: str = "best") -> None:
    torch.save(model.state_dict(), f"{path}_{mode}.pth")
   
    
def load_config_from_json(config_file_path):
    # Check if the config file exists
    if os.path.isfile(config_file_path):
        # Load the JSON data from the file
        with open(config_file_path, "r") as config_file:
            config_dict = json.load(config_file)

        # Convert the dictionary to a YACS configuration object<
        cfg_obj = cfg(config_dict)
        
        # Now you can access the configuration parameters
       # print(cfg_obj)
        print('config sucessfullly loaded from ',config_file_path)
    else:
        print("Config file does not exist.")
        raise NameError
    return cfg_obj


class myEmbeddings(object):
    
    def __init__(self, embedding_path,device ) :
        """
        Class for handling word embeddings and their transformations.

        Args:
        - embedding_path (str): Path to the pre-trained embedding tensors.
        - device ('cpu' or 'cuda:0'): Device on which to load the embeddings.

        Attributes:
        - embedding (torch.nn.Embedding): Embedding layer initialized with  tensors from embedding path.
        - embedding_vectors (torch.Tensor): Transposed and device-moved embedding vectors.
        - norm_vector (torch.Tensor): Norms of embedding vectors.

        Methods:
        - to_word_vectors(data): Converts input indices to word vectors.
        - get_pred_from_vw(data): Transforms word embeddings back to targets (indices).
        """

        vectors = torch.load(embedding_path, map_location=device)
        
        self.embedding = torch.nn.Embedding.from_pretrained(vectors).to(device)
        
        self.embedding_vectors = vectors.transpose(1,0).to(device)
        
        self.norm_vector = torch.linalg.norm(self.embedding_vectors ,dim=0)
                    
    def to_word_vectors (self,data) :
        """
        Converts input indices to word vectors.

        Args:
        - data (torch.Tensor): Input tensor of indices.

        Returns:
        - wv (torch.Tensor): Word vectors corresponding to input indices.
        """
        data=data-1
        data[data<=0] = 0
        wv = self.embedding( data)
        wv = wv.moveaxis(-1,1)
        
        return wv

    def get_pred_from_vw (self,data):
        """
        Transforms word embeddings back to targets.

        Args:
        - data (torch.Tensor or dict): Input tensor or dictionary with 'contrastive' key.

        Returns:
        - predictions (torch.Tensor): Predicted targets based on cosine similarity.
        """
        
        if type(data).__name__ == 'dict': 
            data=data['contrastive']
        
        
        if len(data.shape) ==4 :     
            b,c,h,w = data.shape
            #Reorder the channel to have embedding dimension as the last dimension
            data = data.moveaxis(1,-1).reshape(-1,c)
        else : 
            c,h,w = data.shape
            b=1
            #Reorder the channel to have embedding dimension as the last dimension
            data = data.moveaxis(0,-1).reshape(-1,c)
        
        
        # Compute cosine similarity as dot product and norms
        dot_product = torch.einsum('nc,ck->nk',
                                    data, 
                                    self.embedding_vectors )
        norm = torch.einsum( 'n,k->nk', 
                             torch.linalg.norm(data,dim=-1) , 
                             self.norm_vector )
        
        
        predictions = torch.argmax (dot_product / norm , dim=1) 
        
        # Reshape and add 1 to taken into account the background class (index 0) :
        predictions = predictions.reshape(b,h,w) +1 
        
        return predictions
    

