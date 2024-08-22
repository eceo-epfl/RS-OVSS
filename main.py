import os
import numpy as np
import random
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


import utils.utils as u
from utils.dataset import FLAIRDataset
from utils.TLMdataset import tlmDataset
from utils.losses import CELoss
from utils.metrics import SegmentationMetrics
from config.default import get_cfg_defaults
from utils.losses import WrapperBcosLossWAugm
from utils.utils import parse_args


### set all random seeds 
seed = 2438  
torch.backends.cudnn.benchmark = False # True improve perf, False improve reproductibility
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed) 
random.seed(seed) 
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    ) -> float:  
    """Function to train a model for one epoch.

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
    
    for i, batch in enumerate(dataloader):
        
        optimizer.zero_grad()

        images, mask = batch["pixel_values"].to(device), batch["labels"].to(device)
        outputs = model(images)
        logits = outputs["logits"]
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=mask.shape[1:], mode="bilinear", align_corners=False
        )
        loss,correct = criterion(upsampled_logits, mask, return_correct=True)
        loss.backward()
        optimizer.step()       

        progress_bar.update(1)
        progress_bar.set_postfix({"Train Loss": loss.item()})
        total_correct += correct
        total_loss += loss.item()
        
    scheduler.step()    

    return total_loss / (i + 1), total_correct / len(dataloader)


@torch.no_grad()
def val_epoch_step(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, epoch: int) -> float:
    
    """Function to validate a model for one epoch.
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

    for _, batch in enumerate(dataloader):
        images, mask = batch["pixel_values"].to(device), batch["labels"].to(device)
        outputs = model(images)
        logits = outputs["logits"]
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=mask.shape[1:], mode="bilinear", align_corners=False
        )
        loss,correct = criterion(upsampled_logits, mask, return_correct=True)

        progress_bar.update(1)
        progress_bar.set_postfix({"Val Loss": loss.item()})
        total_loss += loss.item()
        total_correct += correct
       

    return total_loss / len(dataloader), total_correct / len(dataloader)


@torch.no_grad()
def test_model(model, dataloader, classes_dict, embeddings, return_cm = False) :
    """Function to evaluate the model on the test set.

    Args:
        model (nn.Module): Semantic segmentation model.
        dataloader (DataLoader): Input DataLoader.
        classes_dict(dict): Key-values pairs with indices and classes names.
        embeddings (Object) : MyEmbeddings object used to retrieved class index from vectors.
    Returns:
        results(dict): dictionnary with results metrics.
    """
    metrics = SegmentationMetrics(classes_dict=classes_dict,ignore_index=0)
    progress_bar = tqdm(total=len(dataloader), desc="Testing Progress")

    for batch in dataloader:
        images, mask = batch["pixel_values"].to(device), batch["labels"].to(device)
        outputs = model(images)
        logits = outputs["logits"]
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=mask.shape[1:], mode="bilinear", align_corners=False
        )
        if embeddings is not None :
            pred_seg = embeddings.get_pred_from_vw(upsampled_logits)
        else : 
            pred_seg = torch.argmax(upsampled_logits, dim=1)
        metrics.add(y_true= mask.detach().cpu(),y_pred=pred_seg.detach().cpu())
        progress_bar.update(1)

    results =    metrics.get_metrics()
    if return_cm :
        cm =metrics.confusion_matrix
        return results, cm
    return results     


@torch.no_grad()
def test_model_on_tlm(model, dataloader, classes_dict, embeddings):
    """Function to evaluate the model on the TLM dataset.

    Args:
        model (nn.Module): Semantic segmentation model.
        dataloader (DataLoader): Input DataLoader.
        classes_dict(dict): Key-values pairs with indices and classes names.
        embeddings (Object) : MyEmbeddings object used to retrieved class index from vectors.
    Returns:
        results(dict): dictionnary with results metrics.
    """
    metrics = SegmentationMetrics(classes_dict=classes_dict,ignore_index=0)
    progress_bar = tqdm(total=len(dataloader), desc="Testing Progress")

    for batch in dataloader:
        images, mask = batch["pixel_values"].to(device), batch["labels"].to(device)
        # oversample input TLM image to be more similar to to FLAIr resolution (50cm-->30cm):
        images = torch.nn.functional.interpolate(images,size=[350,350])
        outputs = model(images)
        logits = outputs["logits"]
        logits = torch.nn.functional.interpolate(
            logits, size=mask.shape[1:], mode="bilinear", align_corners=False
        )
        if embeddings is not None :
            pred_seg = embeddings.get_pred_from_vw(logits)
        else : 
            pred_seg = torch.argmax(logits, dim=1)
            
        pred_seg = torch.nn.functional.interpolate(  pred_seg.unsqueeze(0).float(), size = [200,200],  mode='nearest').long()
        metrics.add(y_true= mask.detach().cpu(),y_pred=pred_seg.detach().cpu())
        progress_bar.update(1)

    results =    metrics.get_metrics()

    return results  

@torch.no_grad()
def run_test_on_TLM(cfg):
    """
    Function to initialise the model evaluation on the TLM dataset

    Args:
        cfg (cfgNode): configuration object
    """
    print('-'*40,'START model transfer to TLM','-'*40)
    # Setup Dataset and dataloader :
    test_dataset = tlmDataset(split_path=cfg.TLM.TEST_SPLIT_PATH,
                              rgb_dir=cfg.TLM.RGB_DIR,
                              label_dir=cfg.TLM.LABEL_DIR,
                              patch_size=cfg.TLM.TEST_PATCH_SIZE,
                              debug=cfg.TLM.DEBUG,
                              )
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=cfg.test_batch_size, 
                                 shuffle=False, 
                                 num_workers=cfg.train.num_workers)
    embeddings =None
    if cfg.train.contrastive != [] and 'contrastive' in cfg.MODEL_TYPE:
        embeddings = u.myEmbeddings( embedding_path= cfg.TLM.EMBEDDING_PATH, device=device )
        # embedings has 14 vectors (no embeddings for the background class)
    print('\tTest batch size : ',cfg.test_batch_size,  )
    
    # Test model with best model weights 
    model = u.load_model(model_type = cfg.MODEL_TYPE, n_class = cfg.FLAIR.N_CLASS , 
                         embedding_size = cfg.train.contrastive[0]['embedding_size'] if 'contrastive' in cfg.MODEL_TYPE else 0
                         )
    weights = torch.load(cfg.output_dir + '/model_best.pth')
    model.load_state_dict(weights, strict=True)
    print('loaded weights from ',cfg.output_dir + '/model_best.pth')
    model.eval()
    model.to(device)
    print('*'*40,'Start test on TLM','*'*40,)
    best_results = test_model_on_tlm(model,
                                     dataloader=test_dataloader, 
                                     classes_dict=cfg.TLM.CLASSES[0], 
                                     embeddings=embeddings)
    best_results.to_csv( cfg.output_dir + '/TLM_results.csv' )
    print('*'*40,'End test on TLM','*'*40,)
    

@torch.no_grad()
def run_test (cfg) :
    """Function to initialise the model evaluation on the test set
    Args:
        cfg (cfgNode): configuration object
    """
    print('-'*40,'Start model testing','-'*40)
    test_dataset = FLAIRDataset(config=cfg, phase='test')
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=cfg.test_batch_size, 
                                 shuffle=False, 
                                 num_workers=cfg.train.num_workers)
    embeddings =None
    if cfg.train.contrastive != [] and 'contrastive' in cfg.MODEL_TYPE:
        embeddings = u.myEmbeddings( embedding_path= cfg.train.contrastive[0]['embedding_path'], device=device )
    print('\tTest batch size : ',cfg.test_batch_size,  )
        
    ### Test model with best model weights 
    model = u.load_model(model_type = cfg.MODEL_TYPE, n_class = cfg.FLAIR.N_CLASS , 
                         embedding_size = cfg.train.contrastive[0]['embedding_size'] if 'contrastive' in cfg.MODEL_TYPE else 0
                         )
    weights = torch.load(cfg.output_dir + '/model_best.pth')
    model.load_state_dict(weights, strict=True)
    print('loaded weights from ',cfg.output_dir + '/model_best.pth')
    model.eval()
    model.to(device)
    print('*'*40,'BEST MODEL RESULTS','*'*40,)
    best_results = test_model(model,dataloader=test_dataloader, 
                              classes_dict=cfg.FLAIR.CLASSES[0], 
                              embeddings=embeddings)
    best_results.to_csv( cfg.output_dir + '/model_best_results.csv' )
    
 
    return 


def launch_training(cfg) -> None:
    """
    Launch training models script for our TACOSS models.
    Args:
        cfg (DictConfig): Configuration object from yacs.
    """
    # General experiment setup :
    global device    
    nbr_epochs = cfg.MAX_EPOCHS 
    date = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_dir = cfg.RESULTS_PATH +'/' +cfg.NAME #+date
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg.output_dir = output_dir
    
    # save config to json file :
    with open(  output_dir+"/cfg.json", "w") as file:
        json.dump(dict(cfg), file,indent =4)

    
    # Dataset Set-up
    train_dataset = FLAIRDataset(config=cfg, phase='train')    
    val_dataset   = FLAIRDataset(config=cfg, phase='val')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers  )
    train_dataloader.dataset.sample_id_list()
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    
    # Model & Optimizer & Loss Set-up   
    model = u.load_model (model_type = cfg.MODEL_TYPE,
                          n_class =cfg.FLAIR.N_CLASS , 
                          from_file = cfg.DIR_PATH +'output/' + cfg.MODEL_FROM_FILE,
                          freeze_encoder = cfg.FREEZE_ENCODER,
                          embedding_size = cfg.train.contrastive[0]['embedding_size'] if 'contrastive' in cfg.MODEL_TYPE else 0
                          )
    model.to(device)
    optimizer = u.optimizer_factory(cfg.train.optimizer_type, model,lr=cfg.train.lr)
    scheduler = u.scheduler_factory(cfg.train.scheduler_type, optimizer, max_epochs = nbr_epochs   )
    
    if 'contrastive' not in cfg.MODEL_TYPE :
        criterion = CELoss(ignore_index=0)
    else : 
        contrastive_cfg = cfg.train.contrastive[0]
        contrastive_cfg['device']=device
        contrastive_cfg['n_class'] = cfg.FLAIR.N_CLASS   
        contrastive_cfg['augmented_path'] = cfg.DIR_PATH + contrastive_cfg['augmented_path'] 
        contrastive_cfg['embedding_path'] = cfg.DIR_PATH + contrastive_cfg['embedding_path'] 
        contrastive_cfg['device']
        criterion = WrapperBcosLossWAugm(**contrastive_cfg
                                    )
    

    ################
    # Training Loop
    ################
    print('-'*40,'START TRAINING','-'*40)
    progress_bar_epoch = tqdm(total=nbr_epochs, desc="Epoch Progress", ncols =30)
    val_loss = 0
    best_val_loss = np.inf
    best_val_oacc = 0
    
    for epoch in range(nbr_epochs):
        
        train_dataloader.dataset.sample_id_list()
        
        # Training phase :
        train_loss, _ = train_epoch_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
            criterion=criterion,
            epoch=epoch,
        )


        # Validation phase :
        val_loss,val_oacc = val_epoch_step(model=model, 
                                                dataloader=val_dataloader, 
                                                criterion=criterion, 
                                                epoch=epoch)

            
        # Save model if best oacc :
        if  val_oacc> best_val_oacc  :
            best_val_oacc = val_oacc
            u.save_module(model=model, path=Path (output_dir , "model"), mode="best_oacc")

        # Save model if best loss :
        if  val_loss < best_val_loss :
            best_val_loss = val_loss
            u.save_module(model=model, path=Path (output_dir , "model"), mode="best")
            u.save_module(model=optimizer, path=Path(output_dir ,"optim"), mode="best")

        progress_bar_epoch.update(1)
        progress_bar_epoch.set_postfix({"Total Val Loss": val_loss})

        u.save_module(model=model, path=Path (output_dir,"model"), mode="last")
        u.save_module(model=optimizer, path=Path(output_dir , "optim"), mode="last")
        
        
    print('-'*40,'END TRAINING',args.cfg,'-'*40) 




if __name__ == '__main__':
    args = parse_args()    
    
    cfg = get_cfg_defaults()  
    cfg.merge_from_file(cfg.DIR_PATH + f'/config/{args.cfg}.yaml')

            
    print('-'*40,'Launch experiment',args.cfg, '-'*40)
    launch_training(cfg)
    run_test(cfg)
    run_test_on_TLM(cfg)
    print('\n\n','*'*40,' SUCCESSFUL RUN FOR',args.cfg,'*'*40,'\n\n',) 
