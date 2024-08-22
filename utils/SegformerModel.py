from transformers import SegformerForSemanticSegmentation, logging
import torch.nn as nn
import torch
import os
from collections import OrderedDict
logging.set_verbosity(logging.CRITICAL )

def my_init_weights(modules):
    """Initialize the weights"""
    for module in modules:

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class ContrastiveSegformer (nn.Module):
    def __init__(self, embedding_size,from_pretrained = '', freeze_encoder = False):
        super(ContrastiveSegformer, self).__init__()
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",num_labels = 13)

        if os.path.exists(from_pretrained) :
            weights = torch.load(from_pretrained)            
            mis_keys, un_keys = model.load_state_dict(weights, strict=False)
            assert len(mis_keys) == 0 and len(un_keys) == 0, "Missing or unexpected keys when loading pretrained weights"
            print ( '\tEncoder Segformer weights sucessfully loaded from', from_pretrained)

        elif from_pretrained != '':
            raise Exception('failed to load weights from checkpoints', from_pretrained)
        
        model.decode_head.classifier=   nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, embedding_size, 1),
                        ) 
        my_init_weights(modules = model.decode_head.classifier)
        
        if freeze_encoder:
            
            for param in model.parameters():
                param.requires_grad = False
            for param in model.decode_head.parameters():
                param.requires_grad = True
                
            print ( '\tEncoder frozen for contrastive Segformer. Decoder is trained')

                            
        self.model = model


                
    def forward(self, x):
        
        return  self.model(x)
    
