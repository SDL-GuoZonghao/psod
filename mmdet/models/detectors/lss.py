from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..builder import build_backbone, build_head, build_neck
import torch.nn as nn
import torch


@DETECTORS.register_module()
class LSS(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 lss_head,
                 train_cfg,
                 test_cfg,
                 roi_skip_fpn=False,
                 neck=None,
                 init_cfg=None,
                 *args, **kwargs):
        super(LSS, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            *args, **kwargs)
        
        if lss_head is not None:
            lss_train_cfg = train_cfg.lss if train_cfg is not None else None
            lss_head.update(train_cfg=lss_train_cfg)
            lss_head.update(test_cfg=test_cfg.lss)
            self.lss_head = build_head(lss_head)
            
        self.roi_skip_fpn = roi_skip_fpn
        
    @property
    def with_lss_head(self):
        return hasattr(self, 'lss_head') and self.lss_head is not None
            
    def get_roi_feat(self, x, vit_feat):
        B, _, H, W = x[2].shape
        x = [
            vit_feat.transpose(1, 2).reshape(B, -1, H, W).contiguous()
        ]
        return x
    
    def extract_feat(self, img):
        x = self.backbone(img)
        x = list(x)
        x[0] = self.neck(x[0])
        return x
            
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        
        x = self.extract_feat(img)
        if len(x) == 4:
            x, vit_feat, point_tokens, attns = x
            # gt_points
            gt_points = [torch.cat([
                bboxes[:, 0::2].mean(-1).unsqueeze(-1), 
                bboxes[:, 1::2].mean(-1).unsqueeze(-1)
            ], dim=-1) for bboxes in gt_bboxes]
            # point settings
            imgs_whwh = []
            for meta in img_metas:
                h, w, _ = meta['img_shape']
                imgs_whwh.append(x[0].new_tensor([[w, h]]))
            imgs_whwh = torch.cat(imgs_whwh, dim=0)
            imgs_whwh = imgs_whwh[:, None, :]
            
            losses = dict()
            # point training / pseudo gt generation
            pseudo_gt_labels, pseudo_gt_bboxes, \
                point_losses = self.lss_head.forward_train(x,
                                                        vit_feat,
                                                        point_tokens,
                                                        attns,
                                                        img_metas,
                                                        gt_bboxes,
                                                        gt_labels,
                                                        gt_points,
                                                        imgs_whwh=imgs_whwh)      
            losses.update(point_losses)
            # rpn setting
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                pseudo_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            # rcnn setting
            if self.roi_skip_fpn: # imted
                roi_losses = self.roi_head.forward_train(self.get_roi_feat(x, vit_feat), img_metas, proposal_list,
                                                         pseudo_gt_bboxes, pseudo_gt_labels,
                                                         gt_bboxes_ignore, gt_masks,
                                                         img=img, **kwargs)
#             else: # faster rcnn
#                 roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
#                                                          pseudo_gt_bboxes, pseudo_gt_labels,
#                                                          gt_bboxes_ignore, gt_masks,
#                                                          **kwargs)
            losses.update(roi_losses)
            return losses
        else:
            assert False, 'no implemention'
            
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)
        if len(x) == 4:
            x, vit_feat, point_tokens, attns = x
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            
            if self.roi_skip_fpn: # imted
                return self.roi_head.simple_test(
                    self.get_roi_feat(x, vit_feat), proposal_list, img_metas, rescale=rescale)
            else:
                return self.roi_head.simple_test(
                    x, proposal_list, img_metas, rescale=rescale)