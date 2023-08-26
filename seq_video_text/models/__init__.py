# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .seqformer import build
from .clip_output import Clips, Videos, binary_mask_to_quad
from .roi_align import ROIAlign
from .poolers import ROIPooler
from .rec_stage import REC_STAGE


def build_model(args):
    return build(args)


__all__ = ['Clips', 'Videos', 'binary_mask_to_quad', 'build_model', 'ROIAlign', 'ROIPooler', 'REC_STAGE']
