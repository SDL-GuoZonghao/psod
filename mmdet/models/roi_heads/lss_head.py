import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

@HEADS.register_module()
class LatentScaleSelectionHead(nn.Module):
    def __init__(self,
                 instance_extractor=None,
                 point_head=None,
                 mil_head=None,
                 train_cfg=None,
                 test_cfg=None,
                ):
        super(LatentScaleSelectionHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 设置pooling的pre_process
        self.pooling = self.train_cfg.get('pooling', None)
        if self.pooling is None:
            assert False, 'should identificate "pooling" in train_cfg'
        self.pooling_type = self.pooling.type
        self.scale = self.pooling.get('scale')
        self.scale_method = self.pooling.get('scale_method')
        
        if mil_head is not None:
            self.init_mil_head(instance_extractor, mil_head)
        if point_head is not None:
            self.init_point_head(point_head)
            
        self.init_assigner_sampler()

    @property
    def with_mil(self):
        return hasattr(self, 'mil_head') and self.mil_head is not None
    
    @property
    def with_point(self):
        return hasattr(self, 'point_head') and self.point_head is not None
    
    def init_mil_head(self, instance_extractor, mil_head):
        self.instance_extractor = build_roi_extractor(instance_extractor)
        self.mil_head = build_head(mil_head)
        
    def init_point_head(self, point_head):
        self.point_head = build_head(point_head)
    
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.point_assigner = None
        self.point_sampler = None
        if self.train_cfg:
            self.point_assigner = build_assigner(self.train_cfg.assigner)
            self.point_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_weights(self):
        if self.with_mil:
            self.instance_extractor.init_weights()
            self.mil_head.init_weights()
        if self.with_point:
            self.point_head.init_weights()
            
    def generate_multi_scale_cams(self, attns_maps, point_results, align_label=True):
        '''
        input:
            attns_maps -> (blocks, batch_size, N, N)
            #只取需要的层数
            -> attns_maps -> (scale, batch_size, N, N)
            #得到多尺度的attns
            -> joint_attentions -> (scale, batch_size, N, N)
            -> joint_attentions -> (batch_size, scale, N, N)
            #取出match的object的attention maps,并resize成maps
            -> point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
        output:
            point_attentions -> (batch_size, scale, num_gt, patch_h * patch_w)
        '''
        # 1. multiple matrics
        if self.scale_method == 'average':
            assert False, 'no implemention'
            # (1, batch_size, patch_h, patch_w) -> (batch_size, 1, patch_h, patch_w)
            joint_attentions = attns_maps.mean(0).unsqueeze(0).permute(1, 0, 2, 3)
        elif self.scale_method == 'multiple':
            attns_maps = attns_maps[-self.scale:]
            residual_att = torch.eye(attns_maps.size(2), dtype=attns_maps.dtype, device=attns_maps.device)
            aug_att_mat = attns_maps + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)
            joint_attentions = torch.zeros(aug_att_mat.size(), dtype=aug_att_mat.dtype, device=aug_att_mat.device)
            joint_attentions[-1] = aug_att_mat[-1]
            for i in range(2, len(attns_maps) + 1):
                joint_attentions[-i] = torch.matmul(joint_attentions[-(i - 1)], aug_att_mat[-i])
            # (scale, batch_size, N, N) -> (batch_size, scale, N, N)
            joint_attentions = joint_attentions.permute(1, 0, 2, 3)
            
        # 2. get point-wise attention maps
        pos_inds = point_results['pos_inds']        
        num_imgs = len(pos_inds)
        num_points = point_results['cls_score'].size(1)
        
        multiple_cams = []
        for i in range(num_imgs):
            pos_inds_per_img = pos_inds[i]
            point_attn_maps_per_img = joint_attentions[i, :, -num_points:, 1:-num_points]
            matched_point_attn_maps_per_img = point_attn_maps_per_img[:, pos_inds_per_img] # (scale, num_gt, num_patches)
            matched_point_attn_maps_per_img = matched_point_attn_maps_per_img.permute(1, 0, 2) # (num_gt, scale, num_patches)
#             if self.attn_norm:
#                 matched_point_attn_maps_per_img = matched_point_attn_maps_per_img / matched_point_attn_maps_per_img.sum(-1).unsqueeze(-1)
            multiple_cams.append(matched_point_attn_maps_per_img)
        multiple_cams = torch.cat(multiple_cams)
        
        # 3. align oders of "gt_labels" and "gt_points" to that of "pos_inds"
        gt_labels = []
        gt_points = []
        labels = point_results['point_targets'][0].reshape(-1, num_points)
        points = point_results['point_targets'][2].reshape(-1, num_points, 2)
        
        for i in range(num_imgs):
            pos_inds_per_img = pos_inds[i]
            labels_per_img = labels[i]
            points_per_img = points[i]
            
            gt_labels.append(labels_per_img[pos_inds_per_img])
            gt_points.append(points_per_img[pos_inds_per_img])        
        
        return multiple_cams, gt_labels, gt_points

    def forward_train(self,
                      x,
                      vit_feat,
                      point_tokens,
                      attns,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_points,
                      imgs_whwh=None,
                      gt_bboxes_ignore=None):
        losses = dict()
        patch_h, patch_w = x[2].size(-2), x[2].size(-1)
        if self.with_point:
            point_results = self._point_forward_train(point_tokens, gt_points, 
                                                      gt_labels, img_metas, 
                                                      imgs_whwh=imgs_whwh)
            losses.update(point_results['loss_point'])
        if self.with_mil:
            mil_results = self._mil_forward_train(vit_feat, attns, gt_points,
                                                  gt_labels, point_results, 
                                                  patch_size=(patch_h, patch_w),
                                                  test_cfg=self.test_cfg)
            losses.update(mil_results['loss_mil'])
        return mil_results['pseudo_gt_labels'], mil_results['pseudo_gt_bboxes'], losses
    
    def _point_forward_train(self, x, gt_points, gt_labels, img_metas, imgs_whwh=None):
        # x -> point tokens -> (batch_size, point_num, c)
        num_imgs = len(img_metas)
        num_proposals = x.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        # inference in mlp_head
        point_results = self._point_forward(x)
        cls_score = point_results['cls_score'].detach()
        point_pred = point_results['point_pred'].detach()
        # get assign and sample results
        point_assign_results = []
        for i in range(num_imgs):
            assign_result = self.point_assigner.assign(
                point_pred[i], cls_score[i], gt_points[i],
                gt_labels[i], img_metas[i])
            point_sampling_result = self.point_sampler.sample(
                assign_result, point_pred[i], gt_points[i]
            )
            point_assign_results.append(point_sampling_result)  
        # matched for generating box
        point_results.update(pos_inds=[sample_results.pos_inds 
                                       for sample_results in point_assign_results]) # pos_inds
        # get point targets
        point_targets = self.point_head.get_targets(point_assign_results, gt_points,
                                                  gt_labels, self.train_cfg, True)
        point_results.update(point_targets=point_targets)
        # point loss
        loss_point = self.point_head.loss(point_results['cls_score'],
                                          point_results['point_pred'],
                                          *point_targets, imgs_whwh=imgs_whwh)
        point_results.update(loss_point=loss_point)
        
        return point_results

    def _point_forward(self, x):
        cls_score, point_pred = self.point_head(x)
        point_results = dict(
            cls_score=cls_score, point_pred=point_pred, point_tokens=x)
        return point_results
    
    def _mil_forward_train(self, vit_feat, attns, gt_points, 
                           gt_labels, point_results, patch_size=None,
                           test_cfg=None):
        mil_results = self._mil_forward(vit_feat, attns, point_results, patch_size,
                                        test_cfg=test_cfg)
        loss_mil = self.mil_head.loss(mil_results['bag_score'], 
                                      mil_results['pseudo_gt_labels'])
        mil_results.update(loss_mil=loss_mil)
        return mil_results
    
    def _mil_forward(self, x, attns, point_results, patch_size=None,
                     test_cfg=None):
        # (num_gt1 + num_gt2, scale, patch_h * patch_w)
        multiple_cams, gt_labels, gt_points = self.generate_multi_scale_cams(attns, point_results)
        
        if self.pooling_type == 'roi':
            num_imgs = len(gt_points)
            pseudo_proposals = self.mil_head.pre_get_bboxes(multiple_cams,
                                                            gt_points,
                                                            patch_size=patch_size,
                                                            test_cfg=test_cfg)
            # x --> (bs, n, c)
            x = x.permute(0, 2, 1).reshape(num_imgs, -1, *patch_size).contiguous()
            rois = bbox2roi([proposals.reshape(-1, 4) for proposals in pseudo_proposals])
            
            instance_feats = self.instance_extractor(
                [x][:self.instance_extractor.num_inputs], rois)
            
            bag_score = self.mil_head(instance_feats, num_scale=self.scale)
            pseudo_gt_labels, pseudo_gt_bboxes = self.mil_head.post_get_bboxes(pseudo_proposals, 
                                                                               bag_score,
                                                                               gt_labels)
            mil_results = dict(bag_score=bag_score, 
                               multiple_cams=multiple_cams,
                               pseudo_proposals=pseudo_proposals,
                               pseudo_gt_labels=pseudo_gt_labels,
                               pseudo_gt_bboxes=pseudo_gt_bboxes)
            return mil_results
        
        elif self.pooling_type == 'attn':
            
            pos_inds = point_results['pos_inds']
            split_lengths = [len(inds) for inds in pos_inds]
            multiple_cams = list(torch.split(multiple_cams, split_lengths, dim=0))
            
            instance_feats = self.instance_extractor(x, multiple_cams)
            bag_score = self.mil_head(instance_feats)
            
            multiple_cams = torch.cat(multiple_cams)
            pseudo_gt_labels, pseudo_gt_bboxes = self.mil_head.get_bboxes(multiple_cams, 
                                                                          bag_score, 
                                                                          gt_points, 
                                                                          gt_labels,
                                                                          patch_size=patch_size,
                                                                          test_cfg=test_cfg)
            mil_results = dict(bag_score=bag_score,
                               multiple_cams=multiple_cams,
                               pseudo_gt_labels=pseudo_gt_labels,
                               pseudo_gt_bboxes=pseudo_gt_bboxes)
        
            return mil_results