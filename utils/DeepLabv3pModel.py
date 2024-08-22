import torch.nn as nn
import torch
import torch.nn.functional as F
import os

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models import resnet

#######################################################################
# Deeplabv3plus implementation from https://github.com/VainF/DeepLabV3Plus-Pytorch.git
#######################################################################

class DeepLabv3plus(nn.Module):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone or ResNet-101.
    Args:
        num_classes (int): number of classes.
        embedding_size (int): size of embeddings.
        backbone_name (str): name of the backbone: resnet50 or resnet101.
    """
    def __init__(self,num_classes=21,  backbone_name='resnet50'):
        super(DeepLabv3plus, self).__init__()
        
        # Build backbone .
        backbone = resnet.__dict__[backbone_name](
                weights='DEFAULT',
                replace_stride_with_dilation=[False, True, True]
                )
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # Build classifier
        inplanes = 2048
        low_level_planes = 256
        self.asppPlus= DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
           
        self._init_weight()

    def forward(self, input):
        input_shape = input.shape[-2:]
        features = self.backbone(input)
        features = self.asppPlus(features)
        result = dict()
        
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["logits"] = x
        
        return result

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
        

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, ):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] ) # Reduce the number of channels
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        
        return  torch.cat( [ low_level_feature, output_feature ], dim=1 ) 
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ContrastiveDeepLabv3plus(DeepLabv3plus):
    """Constructs a contrastive DeepLabV3 model with a ResNet-50 backbone or ResNet-101 with a contrastive head
    Args:
        num_classes (int): number of classes.
        embedding_size (int): size of embeddings.
        backbone_name (str): name of the backbone: resnet50 or resnet101.
    """
    def __init__(self,num_classes=21, embedding_size =None, backbone_name='resnet50', from_file=None):
        super(ContrastiveDeepLabv3plus, self).__init__(num_classes, backbone_name)
        self.embedding_size =embedding_size
        
        if os.path.exists(from_file) and not os.path.isdir(from_file) and from_file is not None :
            weights = torch.load(from_file)
            mis_keys, un_keys = self.load_state_dict(weights['model_state_dict'], strict=True)
            assert len(mis_keys) == 0 and len(un_keys) == 0, "Missing or unexpected keys when loading pretrained encoder weights from file"
            print( f'Model weights sucessfully loaded from {from_file}')
        
        self.contrastive_classifier =nn.Sequential(
                nn.Conv2d(304, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, embedding_size, 1),
                ) 
            
        self._init_weight()


    def forward(self, input):
        input_shape = input .shape[-2:]
        features = self.backbone(input )
        features = self.asppPlus(features)
        result = dict()
        
        x = self.contrastive_classifier (features)
        x = F.interpolate(x, size = input_shape, mode = "bilinear", align_corners = False)
        result["logits"] = x
    
        return result

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

             






