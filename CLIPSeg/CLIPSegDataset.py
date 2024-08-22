# flair dataset for longformer 
import os
import numpy as np
from tifffile import tifffile
from random import sample
import csv
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Define source directory  path :
all_src_path = [
        '/home/valerie/Projects/contrastive-segmentation/contrastive-lc/',
                ]
for path in all_src_path:
    if os.path.isdir(path):
        src_path = path
if src_path is None :
        raise Exception('Failed to find src path')
    
all_data_path = [
        '/data/valerie/flair/',
                ]
for path in all_data_path:
    if os.path.isdir(path):
        data_path = path
if data_path is None :
        raise Exception('Failed to find data path')


FLAIR_CLASSES  = {
        0:'Others',
        1: "building",      2: "pervious surface",      3: "impervious surface",    4: "bare soil",
        5: "water",         6: "coniferous",            7: "deciduous",             8: "brushwood",
        9: "vineyard",      10: "herbaceous vegetation",11: "agricultural land",    12: "plowed land",
        }
all_classes = list(range(13))

class CLIPSegFLAIRDataset(Dataset):

    def __init__(self,   phase,split ):
        
        self.data_dir = data_path
        self.means = [0.4394,0.4554,0.4257,0.,]
        self.stds = [0.1310,0.1215,0.1135,1.]
        self.phase = phase
        self.patch_size = 512
        self.num_classes = 13
        split_path =src_path +'/data/flair_split/'
        split_path = split_path +  '/' + split  +'/'+phase +'.csv'
        if phase =='test' and 'dev' in split :
            print('Warning : testing on dev split. To have baseline results use base split !')
            
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
        array  [array >12] =0 #Sanity check
        return torch.from_numpy (array).float().unsqueeze(0)

    def __getitem__(self, index):

        img_id,msk_id = self.id_list[index]
        rgb = self.read_img(self.data_dir + img_id )
        mask =  self.read_msk(self.data_dir + msk_id)
        sample = torch.cat((rgb,mask),axis=0)      
        sample = self.augm(sample)
        inputs = {"pixel_values": sample[:3,:,:], "labels": sample[3,:,:].long(), 'id': img_id}
        
        if self.phase == 'test' :
            return inputs['pixel_values'],inputs['labels'],img_id
            
            
        elif self.phase in ['train','val']:
            
            unique_label = inputs['labels'].unique()
            
            if torch.rand(1).item()<0.2 : 
                # randomly pick one of the label not in the images with 20% chance :
                absent_label = [ k for k in  all_classes if k not in unique_label.tolist()   ]
                r = torch.randint(1,len(absent_label),(1,)) if len(absent_label) >1 else 0
                class_id = torch.tensor(absent_label[r]).item()
                label = inputs['labels'] == class_id 
                
            else :    
                # randomly pick one of the label present in the images :          
                r = torch.randint(1,len(unique_label),(1,)) if len(unique_label) >1 else 0
                class_id  = unique_label[r].item()
                label = inputs['labels'] == class_id
                
            return (inputs ['pixel_values'], label.float(), class_id)

    
    