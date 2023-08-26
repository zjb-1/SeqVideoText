# ------------------------------------------------------------------------
# SeqFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_clip as T
import random


class SynthTextDataset(TvCocoDetection):
    def __init__(self, img_folder, ann_file, seq_length,
                 transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(SynthTextDataset, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.seq_length = seq_length

    def __getitem__(self, idx):

        instance_check = False
        while not instance_check:
            img, target = super(SynthTextDataset, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)

            if self._transforms is not None:
                img, target = self._transforms(img, target, self.seq_length)

            if len(target['labels']) == 0:  # None instance
                idx = random.randint(0, self.__len__() - 1)
            else:
                instance_check = True

        target['boxes'] = target['boxes'].clamp(1e-6)
        return torch.cat(img, dim=0), target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for i, seg in enumerate(segmentations):
        if seg == 'None' or seg is None:
            mask = torch.zeros((height, width), dtype=torch.uint8)
        else:
            seg = coco_mask.frPyObjects(seg[0], height, width)
            mask = coco_mask.decode(seg)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    # scales = [296, 328, 360, 392]

    if image_set == 'train':
        return T.Compose([
            T.ImgToClip(shift_min_ratio=0.1,
                        shift_max_ratio=0.4,
                        scale_min_ratio=0.1,
                        scale_max_ratio=0.3,
                        filter_sbox=True),
            T.RandomMotionBlur(),
            # T.RandomHorizontalFlip(),
            T.PhotometricDistort_(),
            T.RandomResize([512, 576, 608, 640], 1200),
            # T.Test_Show(),
            T.Check(),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, ds_ind=0):
    root = Path(args.ytvis_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    PATHS = {
        "train": (root / args.train_img_prefix[ds_ind], root / args.train_ann_file[ds_ind]),
        "test": (root / args.test_img_prefix, root / args.test_ann_file),
    }

    img_folder, ann_file = PATHS[image_set]
    print('use SynthText dataset')
    dataset = SynthTextDataset(img_folder, ann_file, args.num_frames, transforms=make_coco_transforms(image_set),
                               return_masks=args.masks, cache_mode=False, local_rank=get_local_rank(),
                               local_size=get_local_size())
    return dataset


