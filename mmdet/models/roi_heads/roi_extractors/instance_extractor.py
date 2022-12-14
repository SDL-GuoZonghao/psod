import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor
import torch.nn as nn
from models.utils import trunc_normal_
from models.vision_transformer import Mlp

@ROI_EXTRACTORS.register_module()
class AttnInstanceExtractor(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels=1024,
                 out_channels=256,
                 value_norm=True,
                 filter_thr=0,
                 extraction_method='matrix',
                ):
        super(AttnInstanceExtractor, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.value_norm = value_norm
        self.filter_thr = filter_thr
        self.extraction_method = extraction_method
        
        self.linear = nn.Linear(in_channels, in_channels, bias=True)
        self.proj = nn.Linear(in_channels, in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=hidden_channels, 
                       act_layer=nn.GELU)
        self.norm = nn.LayerNorm(in_channels)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def init_weights(self):
        self.apply(self._init_weights)
            
    def forward(self, feats, multiple_cams):
        '''
        input:
        feats -> (batch_size, num_patches, in_channels)  
        multiple_cams -> list -> [num_gt, num_scale, num_patch] * batch_size
        
        output:
        instance_feature -> (num_gt, num_scale, out_channels)
        '''
        instance_feats = []
        num_imgs = feats.size(0)
        # 1. obtain multiple scale attention maps
        # [num_gt, scale, num_patches] x batch_size
        # 2. use 'extraction method' to obtain instance feature
        for i in range(num_imgs):
            multiple_cams_per_img = multiple_cams[i] # (num_gt, scale, num_patches)
            # filter noise
            if self.filter_thr > 0:
                min_value, _ = multiple_cams_per_img.min(-1)
                max_value, _ = multiple_cams_per_img.max(-1)
                temp_cams = (multiple_cams_per_img - min_value.unsqueeze(-1)) / (max_value - min_value).unsqueeze(-1)
                multiple_cams_per_img[temp_cams <= self.filter_thr] = 0
                
            # make sure sum is 1
            if self.value_norm:
                multiple_cams_per_img /= multiple_cams_per_img.sum(-1).unsqueeze(-1)
            
            num_gt = multiple_cams_per_img.size(0)
            # use 'extraction method'
            if self.extraction_method == 'matrix':
                # (num_gt, num_patches, in_channels)
                feats_per_img = self.linear(feats[i]).unsqueeze(0).repeat(num_gt, 1, 1) 
                # (num_gt, scale, num_patch) 
#                     x (num_gt, num_patch, in_channels) 
#                     -> (num_gt, scale, in_channels)    
                instance_feats_per_img = multiple_cams_per_img @ feats_per_img 
                instance_feats_per_img = self.proj(instance_feats_per_img)
        
                instance_feats_per_img = instance_feats_per_img + self.mlp(self.norm(instance_feats_per_img))
                
            instance_feats.append(instance_feats_per_img)
        return torch.cat(instance_feats)
