## From the FLAIR Repository  : https://github.com/IGNF/FLAIR-1-AI-Challenge/blob/main/py_module/dataset.py
import os
import numpy as np
from tifffile import tifffile
from random import sample
import csv
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FLAIRDataset(Dataset):

    def __init__(self, config, phase, reduced_to_13=True):
        
        self.data_dir = config.FLAIR.DATA_DIR
        self.means = [0.4394,0.4554,0.4257,0.,]
        self.stds = [0.1310,0.1215,0.1135,1.]
        self.phase = phase
        self.patch_size = config.FLAIR.TRAIN_PATCH_SIZE if phase =='train' else config.FLAIR.TEST_PATCH_SIZE 
        self.num_classes = config.FLAIR.N_CLASS
        self.reduced_to_13 = reduced_to_13
        split_path = config.FLAIR.SPLIT_PATH +  '/' + config.FLAIR.SPLIT  +'/'+phase +'.csv'      

        # Read list of tile id from file:                        
        self.id_list = []
        with open(split_path, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:  
                self.id_list.append( row) 
                
        self.all_id_list = self.id_list
        

        # Print Dataset Description : 
        print(  'Get FLAIR Dataset from', '/'.join(split_path.split('/')[-3:]))
        print( '\tWith',len(self.id_list), 'samples',  
                'from ',split_path.split('/')[-2] ,'split')
        print(  '\tPatch  size : ', self.patch_size )  
        print('\tBatch size :',config.train.batch_size)
      


        # Define COLOR TRANSFORM 
        if self.phase == 'train': 
            # Apply augmentations in train phase :
            self.augm = T.Compose([  
                        T.Normalize( self.means, self.stds),
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomCrop(self.patch_size ),
                    ])     
            self.color_jitter = T.ColorJitter(
                        brightness= 0.4, 
                        contrast= 0.4, 
                        saturation = 0.4, 
                        hue= 0.1,
                    )         
        else:
            self.augm = T.Compose([   
                        T.Normalize( self.means, self.stds),
                        T.RandomCrop(self.patch_size ),
                    ]) 
        
    def __len__(self):
        return len(self.id_list)
    
    def sample_id_list(self):
       
        if len(self.all_id_list) > 5000 and self.phase == 'train':
            self.id_list = sample(self.all_id_list,5000)
            print('random sampling of', len(self.id_list),
                  'samples instead of', len(self.all_id_list))

    
    
    def read_img(self, raster_file: str) -> np.ndarray:
        array =  tifffile.imread(raster_file)
        array = np.moveaxis(array, -1,0)
        array = array[:3,:,:]
        array = torch.from_numpy(array/255).float()
        if self.phase == 'train': 
            array = self.color_jitter(array)
            
        return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        array =  tifffile.imread(raster_file)
        if self.reduced_to_13 :
            #due to imbalanced class frequencies, the number of classes has been reduced to 13 (remapping >12 to background 0)
            array  [array >12] =0
        return torch.from_numpy (array).float().unsqueeze(0)

    def __getitem__(self, index):

        img_id,msk_id = self.id_list[index]
        rgb = self.read_img(self.data_dir + img_id )
        mask =  self.read_msk(self.data_dir + msk_id)
        sample = torch.cat((rgb,mask),axis=0)       
        rgb,tlm =[],[]
 
        sample = self.augm(sample)
        encoded_inputs = {
                        "pixel_values": sample[:3,:,:], 
                        "labels": sample[3,:,:].long(), 
                        'id': img_id
                        }
        return encoded_inputs
    


    