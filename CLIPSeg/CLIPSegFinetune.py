
"""
Training script for CLIPSeg
"""
import numpy as np
import torch
import random
import pandas as pd
############# set all random seeds #########################################################
seed = 2438  
torch.backends.cudnn.benchmark = False # True improve perf, False improve reproductibility
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed) 
random.seed(seed) 
np.random.seed(seed)
##########################################################################################
import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from argparse import ArgumentParser
from torch.cuda.amp import autocast
from CLIPSegDataset import CLIPSegFLAIRDataset
from CLIPSegTLMDataset import tlmDataset
from metrics import SegmentationMetrics
from transformers import  CLIPSegForImageSegmentation, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLAIR_CLASSES_dict  = {
        0: "Others",        1: "building",      2: "pervious surface",      3: "impervious surface",    4: "bare soil",
        5: "water",         6: "coniferous",            7: "deciduous",             8: "brushwood",
        9: "vineyard",      10: "herbaceous vegetation",11: "agricultural land",    12: "plowed land",
        }
TLM_CLASSES_dict = {   
        0:'Others',    1:'agricultural area',2:'building',    3:'bush forest',
        4:'forest',    5:'glacier',6:'lake',    7:'wetlands',
        8:'vineyards',    9:'railways',10:'river',    11:'road',
        12:'rocks',    13:'open forest',14:'rocks with grass',   
        }
FLAIR_CLASSES_list = list(FLAIR_CLASSES_dict.values())
TLM_classes = list(TLM_CLASSES_dict.values())

SPLIT_PATH =  './CLIPSeg/data/tlm_split_100m/soleil_100m.csv'
RGB_DIR = '/data/valerie/swisstopo/SI_2020_50cm_100m/'
LABEL_DIR ='/data/valerie/contrastive-lc/tlm_14cls_100m/'


def train_epoch_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    ) -> float:  
    """Script to train a model for one epoch.

    Args:
        model (nn.Module): Semantic segmentation model.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler.
        dataloader (DataLoader): Input DataLoader.
        criterion (nn.Module): Loss function.
        epoch (int): Integer representing the current epoch.
    Returns:
        float: Return loss value.
    """
    
    global device
        
    progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} - Training Progress")
    model.train()
    total_loss = 0
    dataloader.dataset.sample_id_list()
    total_correct =0
    
    scaler = torch.cuda.amp.GradScaler()
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    autocast_fn = autocast
    
    for i, batch in enumerate(dataloader):
        
        optimizer.zero_grad()        
        images, labels ,lbl_idx = batch
        cond = ['an image of a '+ FLAIR_CLASSES_dict[x.item()] for x in lbl_idx]
        inputs = processor(text=cond, padding=True, return_tensors  = 'pt')
        with autocast_fn():
            outputs = model(pixel_values = images.to(device), 
                              input_ids =inputs['input_ids'].to(device),
                              attention_mask = inputs['attention_mask'].to(device),
                              )
        
            loss = criterion(outputs['logits'].squeeze(), labels.squeeze().to(device)) 
        total_loss += loss.item()
        pred = (outputs['logits'].sigmoid() >0.5).long().cpu()
        correct = (pred == labels).sum()/len(labels.flatten())

	    # Scale Gradients
        scaler.scale(loss).backward()
	    # Update Optimizer
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"Train Loss": loss.item()})
        total_correct += correct
        total_loss += loss.item()
           

    return total_loss / (i + 1), total_correct / (i + 1)


@torch.no_grad()
def val_epoch_step(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, epoch: int) -> float:
    
    """Script to validate a model for one epoch.
        Args:
            model (nn.Module): Semantic segmentation model.
            dataloader (DataLoader): Input DataLoader.
            criterion (nn.Module): Loss function.
            epoch (int): Integer representing the current epoch.

        Returns:
            float: Return loss value.
    """
    global device
    progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} - Validation Progress")
    model.eval()
    total_loss = 0
    total_correct =0
    
    autocast_fn = autocast
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    for i, batch in enumerate(dataloader):

        images, labels ,lbl_idx = batch
        cond = ['an image of a '+ FLAIR_CLASSES_dict[x.item()] for x in lbl_idx]
        inputs = processor(text=cond, padding=True, return_tensors  = 'pt')
        with autocast_fn():
            outputs = model(pixel_values = images.to(device), 
                              input_ids =inputs['input_ids'].to(device),
                              attention_mask = inputs['attention_mask'].to(device),
                              )
        
            loss = criterion(outputs['logits'].squeeze(), labels.to(device).squeeze()) 
        total_loss += loss.item()
        pred = (outputs['logits'].sigmoid() >0.5).long().cpu()
        correct = (pred == labels).sum()/len(labels.flatten())

        progress_bar.update(1)
        progress_bar.set_postfix({"Val Loss": loss.item()})
        total_correct += correct
        total_loss += loss.item()
       

    return total_loss / (i + 1), total_correct / (i + 1)


@torch.no_grad()
def test_epoch_model(model, dataloader, flair_classes_dict, return_cm = False) :
    global device
    model.eval()
    metrics = SegmentationMetrics(classes_dict=flair_classes_dict,ignore_index=0)
    progress_bar = tqdm(total=len(dataloader), desc="Testing Progress")
    n_classes = 13# len(FLAIR_CLASSES_dict)
    
    autocast_fn = autocast
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    for batch in dataloader:
        
        images,labels,_ = batch
        images = images.repeat(n_classes,1,1,1)
        cond = ['an image of a '+ x for x in FLAIR_CLASSES_list ] 
        inputs = processor(text=cond, padding=True, return_tensors  = 'pt')
        with autocast_fn():
            outputs = model(pixel_values = images.to(device), 
                              input_ids =inputs['input_ids'].to(device),
                              attention_mask = inputs['attention_mask'].to(device),
                              )
        
    
        
        prediction = torch.argmax(outputs['logits'],dim=0).detach().cpu()
        metrics.add(prediction.squeeze(), labels.squeeze())
        
        
        
    metric_scores = metrics.get_metrics()
    if return_cm :
        cm =metrics.confusion_matrix
        return metric_scores, cm
    return  metric_scores


@torch.no_grad()
def test_on_tlm(model, output_dir,):
    
    # setup TLM Dataset from testing : 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    EXP_DIR = output_dir +'/transfer_to_TLM/'
    
    if not os.path.exists (EXP_DIR ):
        os.mkdir(EXP_DIR )
    
    # Set up dataset  
    test_dataset = tlmDataset( SPLIT_PATH =SPLIT_PATH, rgb_dir= RGB_DIR, label_dir=LABEL_DIR)
    test_dataloader = DataLoader(test_dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    num_workers=1)
    # init test loop : 
    metrics = SegmentationMetrics(classes_dict=TLM_CLASSES_dict,ignore_index=0)
    autocast_fn = autocast
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    n_classes = len(TLM_CLASSES_dict)
    
    for batch in tqdm ( test_dataloader ):
        
        images = batch['pixel_values']
        labels = batch['labels']
        images = images.repeat(n_classes,1,1,1)
        cond = ['an image of a '+ x for x in TLM_classes ]
        
        inputs = processor(text=cond, padding=True, return_tensors  = 'pt')
        
        
        outputs = model(pixel_values = images.to(device), 
                              input_ids =inputs['input_ids'].to(device),
                              attention_mask = inputs['attention_mask'].to(device),
                              )       
        logits = outputs['logits'].detach().cpu()
        prediction = torch.argmax(logits,dim=0)

        metrics.add(prediction.squeeze(), labels.squeeze())
        
    metric_scores = metrics.get_metrics()
    metric_scores.to_csv( EXP_DIR + 'TLM_results.csv' )
    return  metric_scores

        



def save_weights(model, base_path, weight_file='weights.pth'):

    weights_path = os.path.join(base_path, weight_file)

    weight_dict = model.state_dict()

    weight_dict = {n: weight_dict[n] for n, p in model.named_parameters() if p.requires_grad}

    torch.save(weight_dict, weights_path)
    print(f'Saved weights to {weights_path}')


def parse_args():
    parser = ArgumentParser(description='Train CLIPSeg network')


    parser.add_argument("--lr",
                        help="decide which lr to use",
                        default=1e-3,
                        type=float)
    parser.add_argument("--split",
                        help="decide which split to usefrom 'dev',or 'base'",
                        default='dev',
                        type=str)

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = parse_args()  
    
    lr =args.lr
    batch_size = 1
    max_epoch = 100

    exp_name = f'CLIPSeg_ft_{args.split}_20pct_neg_samples'
    
        
    # exp output dir :
    output_dir = 'output/'+exp_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Get dataset
    dataset = CLIPSegFLAIRDataset(phase='train',split=args.split)
    dataset.sample_id_list()
    dataset_val = CLIPSegFLAIRDataset(phase='val',split=args.split)
    dataset_test = CLIPSegFLAIRDataset(phase='test',split=args.split)
    dataloaders = { 'train' :DataLoader(dataset, batch_size=batch_size, num_workers=16),
                'val': DataLoader(dataset_val, batch_size=batch_size, num_workers=16,drop_last=False),
                'test': DataLoader(dataset_test, batch_size=1, num_workers=16,drop_last=False),
                    }

    # training setup :
    max_iterations = len(dataset)*max_epoch //batch_size
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-4)
    criterion = torch.nn.functional.binary_cross_entropy_with_logits


    ################
    # Training Loop
    ################
    if True : # train_model : 
        print('-'*40,'START TRAINING','-'*40)
        progress_bar_epoch = tqdm(total=max_iterations, desc="Epoch Progress")
        val_loss = 0
        best_val_loss = np.inf
        best_val_oacc = 0
        
        # Model finetuning :
        for epoch in range(max_epoch):
            
            # Training phase :
            train_loss, _ = train_epoch_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=dataloaders['train'],
                criterion=criterion,
                epoch=epoch,
            )

            # Validation phase :
            val_loss,val_oacc = val_epoch_step(model=model, 
                                                dataloader=dataloaders['val'], 
                                                criterion=criterion, 
                                                epoch=epoch)
                
            # Save model if best :
            save_weights(model=model, base_path=output_dir, weight_file='last_weights.pt')
            if val_loss < best_val_loss or val_oacc> best_val_oacc:
                best_val_loss = val_loss
                save_weights(model=model, base_path=output_dir, weight_file='best_weights.pt')


            progress_bar_epoch.update(1)
            progress_bar_epoch.set_postfix({"Total Val Loss": val_loss})
        
        print('-'*40,'END TRAINING','-'*40)   
    ################
    # Testing
    ################
    print('-'*40,'START model testing','-'*40)
    weights = torch.load(output_dir + '/best_weights.pt')
    model.load_state_dict(weights, strict=True)
    results,cm = test_epoch_model(model, dataloaders['test'], flair_classes_dict=FLAIR_CLASSES_dict,return_cm=True,)
    pd.DataFrame(results).to_csv(output_dir+'/best_results.csv')      

    print('*'*80)
    print('-'*40,'START model transfer to TLM','-'*40)
    test_on_tlm(model=model,output_dir=output_dir,)
    
    
    print('-'*40,'END TESTING','-'*40)  