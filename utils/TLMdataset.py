import csv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

class tlmDataset(Dataset) : 
    
    def __init__(self,split_path,rgb_dir,label_dir, patch_size=200, debug=False) -> None:
        
        super().__init__()
        self.rgb_dir = rgb_dir 
        self.label_dir = label_dir
        test_split_path = split_path
        means = np.array ( [0.3902, 0.4307, 0.3547, 0.])#config.TLM.MEANS)
        stds  = np.array ([0.1306, 0.1316, 0.1221, 1.] )#config.TLM.STDS )

        
        # Read list of tile id from file:                        
        self.id_list = []
        with open(test_split_path, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:  
                self.id_list.append( row[0]) 
        if debug :
            self.id_list=self.id_list[:50]
        
        # Define data augmentation    
        self.augm = T.Compose([   
                        T.Normalize( means, stds),
                        T.RandomCrop( patch_size ),
                    ]) 
                                                           
        # Print Dataset Description : 
        print(  'Get TLM Dataset from', self.rgb_dir)
        print( '\tWith',len(self.id_list), 'samples',  
                'from ',test_split_path.split('/')[-2] ,'split')
        print(  '\tPatch  size : ', patch_size )      
        if debug :
            print('\t Debuggging TLM dataset')
        
    def __len__(self):
        return len(self.id_list)    
       
        
    def __getitem__(self, index) :
        id = self.id_list[index]
        rgb_path = self.rgb_dir +  id +'_rgb.tif'
        tlm_path = self.label_dir + id +'_tlm.tif'

        rgb = Image.open(rgb_path)
        tlm = Image.open(tlm_path)
        rgb = np.array(rgb)
        tlm = np.array(tlm)
        try :
            # update tlm labels : maps number tp new classes
            tlm[tlm==7] =0 # remove foot path
            tlm[tlm==8] = 0 # remove parks areas
            tlm[tlm==15]=7 # move swamp from 15 to 7
            tlm[tlm==16]=8 # move vineyard from 16 to 8
        except OSError :
            print('OSERROR. cannot read tile ', id,)
            raise OSError
        
        tlm = np.expand_dims(tlm, -1)       
        rgb = torch.from_numpy(rgb/255).float()
        tlm = torch.from_numpy (tlm).float()
        
        if tlm.size(0) != rgb.size(0) :
            tlm = torch.nn.functional.interpolate(tlm.squeeze().unsqueeze(0).unsqueeze(0), size = rgb.shape[.2:]).squeeze().squeeze().unsqueeze(-1)
            
        sample = torch.cat((rgb,tlm),axis=-1)
        sample =torch.transpose(sample, 0,-1)
        rgb,tlm =[],[]
 
        # apply augmentation on both input image and labels
        sample = self.augm(sample)  
        
        # return rgb image, label, and tile id:
        return  {"pixel_values" : sample[:3,:,:].float(),
                 "labels" : sample[3,:,:].long(),
                 'id':id
                 } 

    
    
    
    