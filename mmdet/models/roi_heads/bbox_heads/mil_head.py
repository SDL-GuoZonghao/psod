import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from models.utils import trunc_normal_

from einops import rearrange

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

from ...losses import weight_reduce_loss

def get_bbox_from_cam(cam, point, cam_thr=0.2, area_ratio=0.5, 
                      img_size=None, box_method='expand'):
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    img_h, img_w = img_size
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)
    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        estimated_bbox = []
        areas = list(map(cv2.contourArea, contours))
        area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
        for idx in area_idx:
            if areas[idx] >= areas[area_idx[0]] * area_ratio:
                c = contours[idx]
                x, y, w, h = cv2.boundingRect(c)
                estimated_bbox.append([x, y, x + w, y + h])
    else:
        estimated_bbox = [[0, 0, 1, 1]]
        
    estimated_bbox = np.array(estimated_bbox)
    
    if box_method == 'expand':
        proposal_xmin = np.min(estimated_bbox[:, 0])
        proposal_ymin = np.min(estimated_bbox[:, 1])
        proposal_xmax = np.max(estimated_bbox[:, 2])
        proposal_ymax = np.max(estimated_bbox[:, 3])
        xc, yc = point

        if np.abs(xc - proposal_xmin) > np.abs(xc - proposal_xmax):
            gt_xmin = proposal_xmin
            gt_xmax = xc * 2 -  gt_xmin
            gt_xmax = gt_xmax if gt_xmax < img_w else float(img_w)
        else:
            gt_xmax = proposal_xmax
            gt_xmin = xc * 2 -  gt_xmax
            gt_xmin = gt_xmin if gt_xmin > 0 else 0.0

        if np.abs(yc - proposal_ymin) > np.abs(yc - proposal_ymax):
            gt_ymin = proposal_ymin
            gt_ymax = yc * 2 -  gt_ymin
            gt_ymax = gt_ymax if gt_ymax < img_h else float(img_h)
        else:
            gt_ymax = proposal_ymax
            gt_ymin = yc * 2 -  gt_ymax
            gt_ymin = gt_ymin if gt_ymin > 0 else 0.0

        estimated_bbox = np.array([[gt_xmin, gt_ymin, gt_xmax, gt_ymax]])
        return estimated_bbox    
    
@HEADS.register_module()
class MILHead(nn.Module):

    def __init__(self,
                in_channels=256,
                hidden_channels=1024,
                pooling_type='roi',
                roi_size=7,
                num_classes=20,
                topk_merge=1,
                loss_mil=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=True, # BCE loss 
                    reduction='mean',
                    loss_weight=1.0),
                ):
        super(MILHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.topk_merge = topk_merge
        self.num_classes = num_classes
        self.loss_mil = build_loss(loss_mil)
        
        self.pooling_type = pooling_type
        self.roi_size = roi_size
        
        if pooling_type == 'attn':
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        elif pooling_type == 'roi':
            self.fc1 = nn.Linear(in_channels * roi_size ** 2, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, hidden_channels)
            
        self.score1 = nn.Linear(hidden_channels, num_classes)
        self.score2 = nn.Linear(hidden_channels, num_classes)
        
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
    
    def forward(self, x, num_scale=None):
        if self.pooling_type == 'attn':
            # x -> (num_gt, num_scale, c)
            x = F.relu(self.fc1(x), inplace=True)
            x = F.relu(self.fc2(x), inplace=True)
        elif self.pooling_type == 'roi':
            x = x.reshape(-1, num_scale, self.in_channels * self.roi_size ** 2)
            x = F.relu(self.fc1(x), inplace=True)
            x = F.relu(self.fc2(x), inplace=True)
        # dual-stream 
        score1 = self.score1(x).softmax(-1) # cls branch
        score2 = self.score2(x).softmax(-2) # scale branch
        bag_score = score1 * score2
        return bag_score

    @force_fp32(apply_to=('bag_score'))
    def loss(self,
             bag_score,
             gt_labels,
             **kwargs):
        losses = dict()
        gt_labels = torch.cat(gt_labels).reshape(-1)
        losses['loss_mil'] = self.loss_mil(bag_score.sum(-2), gt_labels)
        return losses
    
    def pre_get_bboxes(self, multiple_cams, gt_points, 
                       patch_size=None, test_cfg=None):
        num_imgs = len(gt_points)
        split_lengths = [len(p) for p in gt_points]
        gt_points = torch.cat(gt_points)
        patch_h, patch_w = patch_size
        num_scale = multiple_cams.size(1)
        
        # 每个attn maps 变成 ori 尺度大小，再进行生成框操作
        multiple_cams = multiple_cams.reshape(-1, num_scale, patch_h, patch_w)
        matched_cams = F.interpolate(multiple_cams, 
                                     (patch_h * 16, patch_w * 16), 
                                     mode='bilinear') # num_gt, num_scale, H, W
        pseudo_proposals = []
        for cam_per_gt, point in zip(matched_cams.detach().cpu().numpy(), gt_points.detach().cpu().numpy()):
            for cam in cam_per_gt:
                box = get_bbox_from_cam(cam, 
                                        point, 
                                        cam_thr=test_cfg['cam_thr'], 
                                        area_ratio=test_cfg['area_ratio'], 
                                        img_size=(patch_h * 16, patch_w * 16), 
                                        box_method=test_cfg['box_method'])
                pseudo_proposals.append(torch.as_tensor(box, 
                                                        dtype=gt_points.dtype, 
                                                        device=gt_points.device))
        pseudo_proposals = torch.cat(pseudo_proposals).reshape(-1, num_scale, 4)
        pseudo_proposals = list(torch.split(pseudo_proposals, split_lengths, dim=0))
        return pseudo_proposals
    
    def post_get_bboxes(self, pseudo_proposals, bag_score, gt_labels):
        num_imgs = len(gt_labels)
        split_lengths = [len(proposals) for proposals in pseudo_proposals]
        
        gt_labels = torch.cat(gt_labels)
        pseudo_proposals = torch.cat(pseudo_proposals)
        num_scale = bag_score.size(-2)
        # 
        index = gt_labels.reshape(-1, 1, 1).repeat(1, num_scale, 1)
        bag_score = torch.gather(bag_score, dim=-1, index=index)[..., 0]
        _, pseudo_index = bag_score.topk(self.topk_merge)
        pseudo_index = pseudo_index.reshape(-1, self.topk_merge, 1).repeat(1, 1, 4)
        pseudo_gt_bboxes = torch.gather(pseudo_proposals,
                                        dim=1,
                                        index=pseudo_index).reshape(-1, 4)
        
        pseudo_gt_bboxes = list(torch.split(pseudo_gt_bboxes, split_lengths, dim=0))
        pseudo_gt_labels = list(torch.split(gt_labels, split_lengths, dim=0))
        return pseudo_gt_labels, pseudo_gt_bboxes
    

    def get_bboxes(self, multiple_cams, bag_score, 
                        gt_points, gt_labels, patch_size=None, 
                        test_cfg=None):
        
        split_lengths = [len(p) for p in gt_points]
        patch_h, patch_w = patch_size
        # 选择bag实际标签的score分数
        pseudo_gt_labels = gt_labels
        
        gt_points = torch.cat(gt_points)
        gt_labels = torch.cat(gt_labels)   
        num_scale = bag_score.size(-2)
        # 
        index = gt_labels.reshape(-1, 1, 1).repeat(1, num_scale, 1)
        bag_score = torch.gather(bag_score, dim=-1, index=index)[..., 0]
        _, pseudo_index = bag_score.topk(self.topk_merge)
        # 获得响应尺度响应最高的层标号
        # pseudo_index (num_gt, topk)
        # multiple_cams (num_gt, scale, h*w)
        pseudo_index = pseudo_index.reshape(
            pseudo_index.size(0), self.topk_merge, 1).repeat(1, 1, multiple_cams.size(-1))
        matched_cams = torch.gather(multiple_cams, dim=1, index=pseudo_index)[:, 0, :]
        matched_cams = matched_cams.reshape(-1, patch_h, patch_w)
        # 变成原图大小并进行后处理
        matched_cams = F.interpolate(matched_cams.unsqueeze(1), 
                                     (patch_h * 16, patch_w * 16), 
                                     mode='bilinear').squeeze(1)
        pseudo_gt_bboxes = []
        for cam, point in zip(matched_cams.detach().cpu().numpy(), gt_points.detach().cpu().numpy()):
            box = get_bbox_from_cam(cam, 
                                    point, 
                                    cam_thr=test_cfg['cam_thr'], 
                                    area_ratio=test_cfg['area_ratio'], 
                                    img_size=(patch_h * 16, patch_w * 16), 
                                    box_method=test_cfg['box_method'])
            pseudo_gt_bboxes.append(torch.as_tensor(box, 
                                                    dtype=gt_points.dtype, 
                                                    device=gt_points.device))
        pseudo_gt_bboxes = torch.cat(pseudo_gt_bboxes)
        pseudo_gt_bboxes = list(torch.split(pseudo_gt_bboxes, split_lengths, dim=0))
        return pseudo_gt_labels, pseudo_gt_bboxes
        