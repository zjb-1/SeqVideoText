# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import os
import random
import numpy as np
import cv2 as cv

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]
    if "rec" in target:
        fields.append("rec")

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False
    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]
    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomSizeCropwithInstance(object):
    def __init__(self, crop_ratio_min, crop_ratio_max, crop_type="relative_range", crop_instance=True):
        assert 0.0 <= crop_ratio_min <= crop_ratio_max <= 1.0, "ratio_max mast larger than ratio_min"
        self.crop_ratio_min = crop_ratio_min
        self.crop_ratio_max = crop_ratio_max
        self.crop_instance = crop_instance
        self.crop_type = crop_type

    def get_crop_size(self, image_size, crop_size):
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return min(crop_size[0], h), min(crop_size[1], w)
        elif self.crop_type == "absolute_range":
            assert crop_size[0] <= crop_size[1]
            ch = np.random.randint(min(h, crop_size[0]), min(h, crop_size[1]) + 1)
            cw = np.random.randint(min(w, crop_size[0]), min(w, crop_size[1]) + 1)
            return ch, cw

    def __call__(self, img: PIL.Image.Image, target: dict):
        img_w, img_h = img.size
        image_size = (img_h, img_w)
        random_ratio = np.random.uniform(self.crop_ratio_min, self.crop_ratio_max)

        crop_size = self.get_crop_size(image_size, (random_ratio, random_ratio))  # h, w
        crop_size = np.asarray(crop_size, dtype=np.int32)

        bboxes = target["boxes"].numpy().tolist()
        choiced_box_index = np.random.randint(0, len(bboxes))
        bbox = bboxes[choiced_box_index]  # 以它为中心进行裁剪
        center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
        min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
        max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
        max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))
        y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
        x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

        if not self.crop_instance:  # don't crop instance, iter
            num_modifications = 0
            modified = True
            crop_size = crop_size.astype(np.float32)
            while modified:
                modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, bboxes)
                num_modifications += 1
                if num_modifications > 100:
                    raise ValueError(
                        "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                            len(bboxes)
                        )
                    )

        if (x0 < 0) or (y0 < 0):
            x0 = np.maximum(x0, 0)
            y0 = np.maximum(y0, 0)

        region = tuple(map(int, (y0, x0, crop_size[0], crop_size[1])))
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Test_Show(object):
    def __init__(self, save_root="/lustre/home/jbzhang/SeqFormer/crop_debug/", show_mask=False):
        self.show_mask = show_mask
        self.save_root = save_root

    def __call__(self, img, target=None):
        boxes_num = target['boxes'].shape[0]
        boxes = target['boxes'].numpy().astype(np.int32)
        image = img.copy()

        draw = PIL.ImageDraw.Draw(image)
        for box_i in range(boxes_num):
            box = boxes[box_i]
            if (box == 0).all():
                continue

            draw.rectangle(box.tolist(), outline=(0, 255, 0), width=2)
        del draw
        image.save(os.path.join(self.save_root, "ddd.jpg"))
        if self.show_mask:
            masks = target["masks"].numpy().astype(np.int32) * 255
            for i in range(masks.shape[0]):
                m_path = os.path.join(self.save_root, f"m_{i}.jpg")
                cv.imwrite(m_path, masks[i])

        import pdb; pdb.set_trace()
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            target["boxes_xyxy"] = boxes.clone()
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
            if "valid" in target:
                target["valid"] = torch.ones(boxes.shape[0], dtype=torch.uint8)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

