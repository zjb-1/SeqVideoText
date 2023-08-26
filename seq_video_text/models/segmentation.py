# ------------------------------------------------------------------------
# SeqFormer Sequence Segmentation.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import copy

from util.boxes import Boxes
import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
from .roi_align import ROIAlign
from .rec_stage import REC_STAGE
from .poolers import ROIPooler


class SeqFormer(nn.Module):
    def __init__(self, args, detr, rel_coord=True, freeze_detr=False):
        super().__init__()
        self.detr = detr
        self.rel_coord = rel_coord
        self.with_rec = args.rec
        self.with_obj = args.obj

        # if freeze_detr:
        #     for p in self.parameters():
        #         p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead

        self.in_channels = hidden_dim // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.max_insts_num = 100
        self.mask_out_stride = 4

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

        self.mask_head = MaskHeadSmallConv(hidden_dim, None, hidden_dim)
        # Build recognition heads
        if args.rec:
            self.roi_mask_size = args.roi_mask_size  # 28
            self.rec_batch_size = args.rec_batch_size  # 128
            self.box_pooler_rec = self._init_box_pooler_rec(detr.backbone.strides, detr.num_feature_levels)

            self.rec_stage = REC_STAGE(args, args.hidden_dim, detr.num_classes, (args.roi_mask_size, args.roi_mask_size),
                                       args.rec_batch_size, args.dim_feedforward, args.nheads, args.dropout)
            self.rec_cnn = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True),
            )

    def _init_box_pooler_rec(self, backbone_strides, num_feature_levels):
        strides = backbone_strides.copy()
        if len(strides) < num_feature_levels:
            ext = [strides[-1] * pow(2, i) for i in range(1, num_feature_levels - len(strides) + 1)]
            strides.extend(ext)

        pooler_resolution = (self.roi_mask_size, self.roi_mask_size)  # (28, 28)
        pooler_scales = tuple(1.0 / s for s in strides)
        sampling_ratio = 2
        pooler_type = "ROIAlignV2"

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, samples: NestedTensor, gt_targets, criterion, train=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        features, pos = self.detr.backbone(samples)
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []
        if self.with_rec:
            self.prepare_roi_mask_targets(gt_targets)  # add gt_roi_mask for rec

        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose()
            src_proj_l = self.detr.input_proj[l](src)  # src_proj_l: [nf*N, C, Hi, Wi] ,  reduce channel

            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))
            src_proj_l = src_proj_l.reshape(n // self.detr.num_frames, self.detr.num_frames, c, h, w)

            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n // self.detr.num_frames, self.detr.num_frames, h, w)

            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l + 1].shape
            pos_l = pos[l + 1].reshape(np // self.detr.num_frames, self.detr.num_frames, cp, hp, wp)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.detr.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)  # 最后一层，down-sample, 1/64 of input, too little
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask  # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))
                src = src.reshape(n // self.detr.num_frames, self.detr.num_frames, c, h, w)
                mask = mask.reshape(n // self.detr.num_frames, self.detr.num_frames, h, w)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np // self.detr.num_frames, self.detr.num_frames, cp, hp, wp)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        query_embeds = None
        query_embeds = self.detr.query_embed.weight

        # srcs_f: [ [bz, nf, C, H, W] * num_feature_levels ]
        hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.detr.transformer(
            srcs, masks, poses, query_embeds)
        valid_ratios = valid_ratios[:, 0]  #
        # memory: [bz,n_f, \sigma(HiWi), C]
        # hs: [num_encoders, bs, num_querries, C]  # used to class & mask
        # hs_box: [num_encoders, bs, nf, num_querries, C]  # used to bbox, has time dimension

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_objs = []
        outputs_masks = []
        indices_list = []

        enc_lay_num = hs.shape[0]
        for lvl in range(enc_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs_box[lvl])
            if self.with_obj:
                outputs_obj = self.detr.obj_embed[lvl](hs_box[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            if self.with_obj:
                outputs_objs.append(outputs_obj)
                outputs_layer['pred_objs'] = outputs_obj

            dynamic_mask_head_params = self.controller(hs[lvl])  # [bs, num_quries, num_params]
            # for training & log evaluation loss
            indices = criterion.matcher(outputs_layer, gt_targets, self.detr.num_frames, valid_ratios)
            indices_list.append(indices)

            reference_points, mask_head_params, num_insts = [], [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                num_insts.append(len(pred_i))
                mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                # This is the image size after data augmentation (so as the gt boxes & masks)

                orig_h, orig_w = gt_targets[i]['size']
                scale_f = torch.stack([orig_w, orig_h], dim=0)

                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[..., :2]
                ref_cur_f = ref_cur_f * scale_f[None, None, :]

                reference_points.append(ref_cur_f[:, pred_i].unsqueeze(0))

            # reference_points: [1, nf,  \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            reference_points = torch.cat(reference_points, dim=2)
            mask_head_params = torch.cat(mask_head_params, dim=1)

            # mask prediction
            outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes,
                                                         reference_points, mask_head_params, num_insts)
            outputs_masks.append(outputs_layer['pred_masks'])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_obj = torch.stack(outputs_objs) if self.with_obj else None
        outputs_mask = outputs_masks

        # bs, outputs_mask = len(outputs_masks[0]), []
        # outputs_masks: dec_num x bs x [1, num_insts, 1, h, w]

        # outputs['pred_samples'] = inter_samples[-1]
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1]
        outputs['pred_objs'] = outputs_obj[-1] if self.with_obj else None
        # rec stage
        if self.with_rec:
            rec_gt_masks, rec_idx, rec_map, target_rec = self.extra_rec_feat(indices_list[-1], outputs, gt_targets,
                                                                                 memory, spatial_shapes)
            # rec_map : [\sum{box}, d_model, roi_mask_size, roi_mask_size]

            rec_map = self.rec_cnn(rec_map)
            rec_proposal_features = hs_box[-1].clone()  # [bs, nf, num_queries, C]

            rec_result = self.rec_stage(rec_map, rec_proposal_features, rec_gt_masks, rec_idx, target_rec)
            outputs['pred_rec'] = rec_result

        if self.detr.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask, outputs_obj)

        # # Retrieve the matching between the outputs of the last layer and the targets
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # indices = criterion.matcher(outputs_without_aux, gt_targets)
        loss_dict = criterion(outputs, gt_targets, indices_list, valid_ratios)

        if not train:
            outputs['reference_points'] = inter_references[-2, :, :, :, :2]

            dynamic_mask_head_params = self.controller(hs[-1])  # [bs, num_quries, num_params]
            bs, num_queries, _ = dynamic_mask_head_params.shape
            num_insts = [num_queries for i in range(bs)]

            reference_points = []
            for i, gt_target in enumerate(gt_targets):
                orig_h, orig_w = gt_target['size']
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                ref_cur_f = outputs['reference_points'][i] * scale_f[None, None, :]
                reference_points.append(ref_cur_f.unsqueeze(0))
            # import pdb;pdb.set_trace()
            # reference_points: [1, N * num_queries, 2]
            # mask_head_params: [1, N * num_queries, num_params]

            reference_points = torch.cat(reference_points, dim=2)
            mask_head_params = dynamic_mask_head_params.reshape(1, -1, dynamic_mask_head_params.shape[-1])

            # mask prediction
            outputs = self.forward_mask_head_train(outputs, memory, spatial_shapes,
                                                   reference_points, mask_head_params, num_insts)

            # outputs['pred_masks']: [bs, num_queries, num_frames, H/4, W/4]

            outputs['pred_masks'] = torch.cat(outputs['pred_masks'], dim=0)
            outputs['pred_boxes'] = outputs['pred_boxes'][:, 0]
            outputs['reference_points'] = outputs['reference_points'][:, 0]
            # import pdb;pdb.set_trace()

        return outputs, loss_dict

    def inference(self, samples: NestedTensor, orig_w, orig_h):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.detr.backbone(samples)
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose()
            src_proj_l = self.detr.input_proj[l](src)  # src_proj_l: [nf*N, C, Hi, Wi]

            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))
            src_proj_l = src_proj_l.reshape(n // self.detr.num_frames, self.detr.num_frames, c, h, w)

            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n // self.detr.num_frames, self.detr.num_frames, h, w)

            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l + 1].shape
            pos_l = pos[l + 1].reshape(np // self.detr.num_frames, self.detr.num_frames, cp, hp, wp)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.detr.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask  # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))
                src = src.reshape(n // self.detr.num_frames, self.detr.num_frames, c, h, w)
                mask = mask.reshape(n // self.detr.num_frames, self.detr.num_frames, h, w)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np // self.detr.num_frames, self.detr.num_frames, cp, hp, wp)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        query_embeds = None
        query_embeds = self.detr.query_embed.weight

        hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, enc_outputs_coord_unact = self.detr.transformer(
            srcs, masks, poses, query_embeds)
        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_objs = []
        outputs_masks = []
        indices_list = []

        enc_lay_num = hs.shape[0]
        for lvl in range(enc_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs_box[lvl])
            outputs_obj = self.detr.obj_embed[lvl](hs_box[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objs.append(outputs_obj)
            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord, 'pred_objs': outputs_obj}

            mask_head_params = self.controller(hs[lvl])  # [bs, num_quries, num_params]

            # reference_points: [1, \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            orig_w = torch.tensor(orig_w).to(reference)
            orig_h = torch.tensor(orig_h).to(reference)
            scale_f = torch.stack([orig_w, orig_h], dim=0)
            reference_points = reference[..., :2].sigmoid() * scale_f[None, None, None, :]
            mask_head_params = mask_head_params
            num_insts = [300]
            # mask prediction
            outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes,
                                                         reference_points, mask_head_params, num_insts)
            outputs_masks.append(outputs_layer['pred_masks'])

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_obj = torch.stack(outputs_objs)
        outputs_mask = outputs_masks
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1]
        outputs['pred_objs'] = outputs_obj[-1]

        if self.with_rec:
            rec_masks, rec_map = self.extra_rec_feat(None, outputs, None, memory, spatial_shapes,
                                                    orig_w, orig_h)
            rec_map = self.rec_cnn(rec_map)
            rec_proposal_features = hs_box[-1].clone()  # [bs, nf, num_querries, C]
            rec_result = self.rec_stage(rec_map, rec_proposal_features, rec_masks)
            rec_result = torch.tensor(rec_result).reshape(*rec_proposal_features.shape[:3], -1)
            outputs['pred_rec'] = rec_result  # [bs, nf, num_query, word_max_len]

        outputs['pred_masks'] = outputs['pred_masks'][0]

        return outputs

    def get_roi_masks(self, boxes, masks):
        device = boxes.device
        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5
        rois = rois.to(device=device)

        output = (
            ROIAlign((self.roi_mask_size, self.roi_mask_size), 1.0, 0, aligned=True)
            .forward(masks[:, None, :, :], rois)
            .squeeze(1)
        )
        del rois
        return output

    def prepare_roi_mask_targets(self, gt_targets):
        for target_per_image in gt_targets:
            gt_masks = target_per_image['masks'].to(dtype=torch.float32)
            gt_boxes = target_per_image['boxes_xyxy']

            output = self.get_roi_masks(gt_boxes, gt_masks)
            output = output >= 0.5

            target_per_image['gt_roi_masks'] = output.float()

    def extra_rec_feat(self, indices, outputs, gt_targets, enc_memory, spatial_shapes, img_w=-1, img_h=-1):
        gt_masks = list()
        gt_boxes = list()
        proposal_boxes_pred = list()
        masks_pred = list()
        enc_memory = enc_memory.clone()
        outputs = outputs.copy()
        bboxes = outputs['pred_boxes']  # [bs, nf, query_num, 4], relative value
        masks = outputs['pred_masks']  # [[1, num_insts, nf, h, w] * bs]
        bs, nf, nr_boxes = bboxes.shape[:3]

        if gt_targets:  # train
            gt_targets = gt_targets.copy()

            indices_with_time = []
            for ind in indices:
                indices_with_time.extend([ind for _ in range(nf)])
            idx = _get_src_permutation_idx(indices_with_time)

            target_rec = []
            for t, (_, i) in zip(gt_targets, indices):
                r_num, r_len = t['rec'].shape
                t_rec = t['rec'].reshape(r_num // nf, nf, r_len)
                t_rec = t_rec[i].permute(1, 0, 2).reshape(-1, r_len)
                target_rec.append(t_rec)
            target_rec = torch.cat(target_rec, dim=0)
            target_rec = target_rec.repeat(2, 1)

        for b in range(bs):
            if gt_targets:  # train
                temp_boxes = gt_targets[b]['boxes_xyxy'].reshape(-1, nf, 4)
                temp_roi_masks = gt_targets[b]['gt_roi_masks']
                temp_roi_masks = gt_targets[b]['gt_roi_masks'].reshape(-1, nf, *temp_roi_masks.shape[-2:])

                img_wh = gt_targets[b]['size'].flip(0)
                img_whwh = img_wh.repeat(2).float()
            else:  # test
                assert img_w != -1 and img_h != -1, "test stage need the img size param"
                img_wh = torch.tensor([img_w, img_h], dtype=torch.int).cuda()
                img_whwh = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float).cuda()

            for f in range(nf):
                if gt_targets:
                    f_temp_boxes = temp_boxes[:, f]  # [num_insts, 4]
                    gt_boxes.append(Boxes(f_temp_boxes[indices[b][1]]))

                    f_temp_masks = temp_roi_masks[:, f]  # [num_insts, 28, 28]
                    gt_masks.append(f_temp_masks[indices[b][1]])

                    pred_boxes = bboxes[b][f][indices[b][0]]  # relative value
                else:
                    pred_boxes = bboxes[b][f]

                pred_boxes = pred_boxes * img_whwh
                pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)  # absolute value
                pred_boxes_ = torch.zeros_like(pred_boxes)
                pred_boxes_[:, 0::2] = pred_boxes[:, 0::2].clamp(0, img_wh[0])
                pred_boxes_[:, 1::2] = pred_boxes[:, 1::2].clamp(0, img_wh[1])
                proposal_boxes_pred.append(Boxes(pred_boxes_))

                b_masks = masks[b][:, :, f]  # [1, num_insts, h, w]
                b_masks_ori = F.interpolate(b_masks, (img_wh[1], img_wh[0]), mode="bilinear").sigmoid().squeeze(0)
                tmp_mask = self.get_roi_masks(pred_boxes_, b_masks_ori)  # [num_insts, 28, 28]
                tmp_mask2 = torch.full_like(tmp_mask, 0).cuda()
                tmp_mask2[tmp_mask > 0.4] = 1
                masks_pred.append(tmp_mask2)

        # get recognition roi region
        # enc_memory  # [bs, nf, \sum_lvl{h * w}, channel]
        f_bs, f_nf, _, f_c = enc_memory.shape
        feats = enc_memory  # [bs, nf, \sum_lvl{h * w}, channel]
        feats = feats.reshape(-1, *feats.shape[-2:])  # [bs * nf, \sum_lvl{h * w}, channel]
        assert f_bs == bs and f_nf == nf, "logic error"

        features = []
        spatial_indx = 0
        for h, w in spatial_shapes:
            f_lvl = feats[:, spatial_indx: spatial_indx + h * w].reshape(f_bs * f_nf, h, w, f_c).permute(0, 3, 1, 2)
            features.append(f_lvl)
            spatial_indx += h * w

        if gt_targets:
            gt_roi_features = self.box_pooler_rec(features, gt_boxes)
            pred_roi_features = self.box_pooler_rec(features, proposal_boxes_pred)
            masks_pred = torch.cat(masks_pred).cuda()
            gt_masks = torch.cat(gt_masks).cuda()
            rec_map = torch.cat((gt_roi_features, pred_roi_features), 0)
            gt_masks = torch.cat((gt_masks, masks_pred), 0)

            rec_map = rec_map[:self.rec_batch_size]

            return gt_masks[:self.rec_batch_size], idx, rec_map, target_rec[:self.rec_batch_size]
        else:
            pred_roi_features = self.box_pooler_rec(features, proposal_boxes_pred)
            # TODO: squeeze() ?
            masks_pred = torch.cat(masks_pred).cuda()

            return masks_pred, pred_roi_features


    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts):
        bs, n_f, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, :, spatial_indx: spatial_indx + h * w, :].reshape(bs, self.detr.num_frames, h, w,
                                                                               c).permute(0, 4, 1, 2, 3)
            encod_feat_l.append(mem_l)
            spatial_indx += h * w
        pred_masks = []
        for iframe in range(self.detr.num_frames):
            encod_feat_f = []  # 同一帧 所有不同尺度的feature
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :])  # [bs, C, hi, wi]

            decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
            # decod_feat_f = self.spatial_decoder(encod_feat_f)[0]
            # [bs, C/32, H/8, W/8]
            reference_points_i = reference_points[:, iframe]
            ######### conv ##########
            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points_i, mask_head_params,
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord)
            # mask_logits: [1, num_queries_all, H/4, W/4]

            # mask_f = mask_logits.unsqueeze(2).reshape(bs, nq, 1, decod_feat_f.shape[-2], decod_feat_f.shape[-1])  # [bs, selected_queries, 1, H/4, W/4]
            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)

            # outputs['pred_masks'] = torch.cat(pred_masks, 2) # [bs, selected_queries, num_frames, H/4, W/4]
        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))

        outputs['pred_masks'] = output_pred_masks
        return outputs

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts,
                                 mask_feat_stride, rel_coord=True):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]

        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)

        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)

        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask, outputs_obj):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_obj is None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c, 'pred_objs': d}
                    for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1], outputs_obj[:-1])]


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # used after upsampling to reduce dimention of fused features!
        self.lay1 = torch.nn.Conv2d(dim, dim // 4, 3, padding=1)
        self.lay2 = torch.nn.Conv2d(dim // 4, dim // 32, 3, padding=1)
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        if fpn_dims != None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, fpns):
        if fpns != None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]
        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        fused_x = self.dcn(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay1(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        return fused_x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        # output single / multi frames
        assert len(orig_target_sizes) == len(max_target_sizes)
        # max_h, max_w = max_target_sizes.max(0)[0].tolist()

        # pred_logits: [bs, num_querries, num_classes]
        # pred_masks: [bs, num_querries, num_frames, H/8, W/8]

        out_refs = outputs['reference_points']
        outputs_masks = outputs["pred_masks"]
        out_logits = outputs['pred_logits']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        outputs_masks = [out_m[topk_boxes[i]].unsqueeze(0) for i, out_m in enumerate(outputs_masks)]
        outputs_masks = torch.cat(outputs_masks)
        bs, _, num_frames, H, W = outputs_masks.shape

        # outputs_masks = F.interpolate(outputs_masks.flatten(0,1), size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = F.interpolate(outputs_masks.flatten(0, 1), size=(H * 4, W * 4), mode="bilinear",
                                      align_corners=False)
        # outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        outputs_masks = outputs_masks.sigmoid() > self.threshold

        # [bs, num_frames, 10, H, W]
        outputs_masks = outputs_masks.reshape(bs, -1, num_frames, outputs_masks.shape[-2],
                                              outputs_masks.shape[-1]).permute(0, 2, 1, 3, 4)

        # reference points for each instance
        references = [refs[topk_boxes[i]].unsqueeze(0) for i, refs in enumerate(out_refs)]
        references = torch.cat(references)

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["scores"] = scores[i]
            results[i]["labels"] = labels[i]
            results[i]['reference_points'] = references[i]

            results[i]["masks"] = cur_mask[:, :, :img_h, :img_w]
            results[i]["masks"] = F.interpolate(results[i]["masks"].float(), size=tuple(tt.tolist()),
                                                mode="nearest").byte()
            results[i]["masks"] = results[i]["masks"].permute(1, 0, 2, 3)

        # required dim of results:
        #   scores: [num_ins]
        #   labels: [num_ins]
        #   reference_points: [num_ins, num_frames, 2]
        #   masks: [num_ins, num_frames, H, W]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
