from yacs.config import CfgNode as CN 
import os

# Define source directory  and data path :
all_src_path = [
        '/home/valerie/Projects/public-contrastive-segmentation/',
                ]
for path in all_src_path:
    if os.path.isdir(path):
        src_path = path
if src_path is None :        
        raise Exception('Failed to find src path')

# Define data path here : 
all_data_path = [
        '/data/valerie/flair/',
                ]
for path in all_data_path:
    if os.path.isdir(path):
        data_path = path
if data_path is None :
        raise Exception('Failed to find data path')

# ----- GENERAL -----
_C = CN()
_C.NAME = ''
_C.DIR_PATH = src_path 
_C.RESULTS_PATH = src_path +'output/'
_C.WANDB_LOG = False
_C.WANDB_PROJECT = 'TACOSS'
_C.MAX_EPOCHS  = 10
_C.MODEL_TYPE = 'contrastive_segformer'
_C.MODEL_FROM_FILE = ''
_C.FREEZE_ENCODER =  False # if True, Freeze encoder
_C.seed = 1234
_C.test_batch_size = 8
_C.output_dir = src_path +'output/'


# ----- FLAIR DATASET-----
_C.FLAIR= CN()
_C.FLAIR.N_CLASS = 13
_C.FLAIR.SPLIT = ''
_C.FLAIR.DATA_DIR = data_path
_C.FLAIR.SPLIT_PATH =  src_path + 'data/flair_split'
_C.FLAIR.TEST_SPLIT_PATH = src_path + 'data/flair_split/base/test.csv'
_C.FLAIR.PLOT_SPLIT_PATH = src_path + '/data/flair_split/base/plot.csv'
_C.FLAIR.TRAIN_PATCH_SIZE  = 512
_C.FLAIR.TEST_PATCH_SIZE  = 512
_C.FLAIR.CLASSES  =[ {
        0:'Others',
        1: "building",      2: "pervious surface",      3: "impervious surface",    4: "bare soil",
        5: "water",         6: "coniferous",            7: "deciduous",             8: "brushwood",
        9: "vineyard",      10: "herbaceous vegetation",11: "agricultural land",    12: "plowed land",
        }]

# ----- TLM DATASET-----
_C.TLM= CN()
_C.TLM.N_CLASS = 14
_C.TLM.RGB_DIR = '/data/valerie/swisstopo/SI_2020_50cm_100m/'
_C.TLM.LABEL_DIR = '/data/valerie/contrastive-lc/tlm_14cls_100m/'
_C.TLM.TEST_SPLIT_PATH = src_path + 'data/tlm_split/soleil_100m.csv'
_C.TLM.TEST_PATCH_SIZE  = 200
_C.TLM.EMBEDDING_PATH = src_path +'data/tlm_labels/tlm_labels_sbert.pt'
_C.TLM.CLASSES  =[{ 
        0:'Others',    1:'agricultural area',2:'building',    3:'bush forest',
        4:'forest',    5:'glacier',6:'lake',    7:'wetlands',
        8:'vineyards',    9:'railways',10:'river',    11:'road',
        12:'rocks',    13:'open forest',14:'rocks with grass',   
        }]
_C.TLM.DEBUG = False




# ----- TRAINING ----
_C.train = CN()
_C.train.batch_size = 10
_C.train.num_workers = 16
_C.train.optimizer_type = 'adamw'
_C.train.scheduler_type = 'polynomial'
_C.train.lr = 6e-5
_C.train.val_epoch_freq = 1
_C.train.contrastive = []

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



