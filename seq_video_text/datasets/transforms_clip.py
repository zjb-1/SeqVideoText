# ------------------------------------------------------------------------
# Transforms and data augmentation for sequence level images, bboxes and masks.
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------

import random
import copy

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh, box_iou
from util.misc import interpolate
import numpy as np
from numpy import random as rand
from PIL import Image
import cv2


class Check(object):
    def __init__(self,):
        pass

    def __call__(self,  img, target, now_frames):
        fields = ["labels"]  #, "area", "iscrowd"]
        if "boxes" in target:
            fields.append("boxes")
        if "masks" in target:
            fields.append("masks")
        if "valid" in target:
            fields.append("valid")
        if "area" in target:
            fields.append("area")
        if "iscrowd" in target:
            fields.append("iscrowd")
        if "rec" in target:
            fields.append("rec")

        ### check if box or mask still exist after transforms
        if "boxes" in target or "masks" in target:
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            num_frames = now_frames
            class_keep = []
            if False in keep:
                for k in range(len(keep)):
                    if not keep[k]:
                        target['boxes'][k] = target['boxes'][k]//10000   # 整除10000，把数值变为0
                
            for inst in range(len(target['labels'])): 
                inst_range = [k for k in range(inst * num_frames, inst * num_frames + num_frames)]
                keep_inst = keep[inst_range].any()  # if exist, keep all frames
                keep[inst_range] = keep_inst
                class_keep.append(keep_inst)
            class_keep = torch.tensor(class_keep)

            for field in fields:
                if field == 'labels':
                    target[field] = target[field][class_keep]
                else:
                    target[field] = target[field][keep]

        return img, target


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    assert mode in ['iou', 'iof']
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def crop(clip, target, region):
    cropped_image = []
    for image in clip:
        cropped_image.append(F.crop(image, *region))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "valid"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        target["valid"][area == 0] = 0    # add.zjb
        if "rec" in target:
            target["rec"][area == 0] = 0
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    return cropped_image, target


def crop_with_filter(clip, target, region):
    """ 过滤 尺寸很小的box  """
    cropped_image = []
    for image in clip:
        cropped_image.append(F.crop(image, *region))

    target = target.copy()
    y0, x0, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "valid"]

    if "boxes" in target:
        boxes_ori = target["boxes"]
        boxes_ori_ = boxes_ori.reshape(-1, 2, 2)
        boxes_ori_wh = boxes_ori_[:, 1, :] - boxes_ori_[:, 0, :]

        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes_ori - torch.as_tensor([x0, y0, x0, y0])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)

        cropped_boxes_wh = cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]   # [num, 2]

        # 根据 w h 的比率过滤
        box_wh_ratios = cropped_boxes_wh / boxes_ori_wh  # [num, 2]
        need_filter = (box_wh_ratios <= 0.1).any(1)
        cropped_boxes = cropped_boxes.reshape(-1, 4)
        # box area 置 0
        cropped_boxes[need_filter] = 0
        area[need_filter] = 0

        target["boxes"] = cropped_boxes
        target["area"] = area
        target["valid"][area == 0] = 0    # add.zjb
        if "rec" in target:
            target["rec"][area == 0] = 0
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        box_masks = target['masks'][:, y0:y0 + h, x0:x0 + w]
        box_masks[need_filter.type(torch.bool)] = False

        target['masks'] = box_masks

        fields.append("masks")

    return cropped_image, target


def hflip(clip, target):
    flipped_image = []
    for image in clip:
        flipped_image.append(F.hflip(image))

    w, h = clip[0].size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    
    return flipped_image, target

def vflip(image,target):
    flipped_image = []
    for image in clip:
        flipped_image.append(F.vflip(image))
    w, h = clip[0].size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(1)

    return flipped_image, target

def resize(clip, target, size, max_size=None):
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

    size = get_size(clip[0].size, size, max_size)
    rescaled_image = []
    for image in clip:
        rescaled_image.append(F.resize(image, size))

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image[0].size, clip[0].size))
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
        if target['masks'].shape[0]>0:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        else:
            target['masks'] = torch.zeros((target['masks'].shape[0], h, w))
    return rescaled_image, target


def pad(clip, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = []
    for image in clip:
        padded_image.append(F.pad(image, (0, 0, padding[0], padding[1])))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[0].size[::-1])
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

    def __call__(self, img: PIL.Image.Image, target: dict, now_frames):
        w = random.randint(self.min_size, min(img[0].width, self.max_size))
        h = random.randint(self.min_size, min(img[0].height, self.max_size))
        region = T.RandomCrop.get_params(img[0], [h, w])
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


class RandomCropWithInstance(object):
    def __init__(self, crop_ratio_min, crop_ratio_max, crop_type="relative_range", filter_sbox=True, crop_instance=True):
        assert 0.0 <= crop_ratio_min <= crop_ratio_max <= 1.0, "ratio_max mast larger than ratio_min"
        self.crop_ratio_min = crop_ratio_min
        self.crop_ratio_max = crop_ratio_max
        self.filter_box = filter_sbox
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

    def __call__(self, img, target: dict, now_frames):
        img_w, img_h = img[0].size
        image_size = (img_h, img_w)
        random_ratio = np.random.uniform(self.crop_ratio_min, self.crop_ratio_max)

        crop_size = self.get_crop_size(image_size, (random_ratio, random_ratio))  # h, w
        crop_size = np.asarray(crop_size, dtype=np.int32)

        # random choice the box as the reference
        t_valid = target["valid"].numpy()
        bboxes = target["boxes"].numpy()
        valid_box_indexs = np.where(t_valid == 1)[0]
        choiced_box_index = np.random.choice(valid_box_indexs)
        bbox = bboxes[choiced_box_index]   # 以它为中心进行裁剪

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

        if self.filter_box:
            return crop_with_filter(img, target, region)
        else:
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


class MinIoURandomCrop(object):
    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, target):
        w,h = img.size
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return img,target
            min_iou = mode
            boxes = target['boxes'].numpy()
            labels = target['labels']

            for i in range(50):
                new_w = rand.uniform(self.min_crop_size * w, w)
                new_h = rand.uniform(self.min_crop_size * h, h)
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue
                left = rand.uniform(w - new_w)
                top = rand.uniform(h - new_h)
                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue
                
                if len(overlaps) > 0:
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                        return mask
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if False in mask:
                        continue
                    #TODO: use no center boxes
                    #if not mask.any():
                    #    continue

                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    target['boxes'] = torch.tensor(boxes)
                
                img = np.asarray(img)[patch[1]:patch[3], patch[0]:patch[2]]
                img = Image.fromarray(img)
                width, height = img.size
                target['orig_size'] = torch.tensor([height,width])
                target['size'] = torch.tensor([height,width])
                return img,target 


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, target, now_frames):
        
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, target

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta

        return image, target

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target, now_frames):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, target

class RandomHue(object): #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target, now_frames):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, image, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target, now_frames):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort_(object):
    def __init__(self):
        pass

    def __call__(self, clip, target, now_frames):
        imgs = []
        for img in clip:
            img = PIL.ImageEnhance.Contrast(img).enhance(rand.uniform(0.4, 1.0))
            img = PIL.ImageEnhance.Brightness(img).enhance(rand.uniform(0.4, 1.0))

            imgs.append(img)

        return imgs, target


class RandomMotionBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def motion_blur(self, image, degree=12, angle=45):
        image = np.array(image)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        return Image.fromarray(blurred)

    def __call__(self, clip, target, now_frames):
        if np.random.rand() > self.p:
            blur_clip = []
            for img in clip:
                degree = np.random.randint(1, 10)
                blur_img = self.motion_blur(img, degree, angle=45)
                blur_clip.append(blur_img)

            return blur_clip, target
        else:
            return clip, target


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
    
    def __call__(self,clip,target,now_frames):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            # img, target = distort(img, target, now_frames)
            img, target = self.rand_light_noise(img, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target

#NOTICE: if used for mask, need to change
class Expand(object):
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, clip, target):
        if rand.randint(2):
            return clip,target
        imgs = []
        masks = []
        image = np.asarray(clip[0]).astype('float32')
        height, width, depth = image.shape
        ratio = rand.uniform(1, 4)
        left = rand.uniform(0, width*ratio - width)
        top = rand.uniform(0, height*ratio - height)
        for i in range(len(clip)):
            image = np.asarray(clip[i]).astype('float32')
            expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),int(left):int(left + width)] = image
            imgs.append(Image.fromarray(expand_image.astype('uint8')))
            expand_mask = torch.zeros((int(height*ratio), int(width*ratio)),dtype=torch.uint8)
            expand_mask[int(top):int(top + height),int(left):int(left + width)] = target['masks'][i]
            masks.append(expand_mask)
        boxes = target['boxes'].numpy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = torch.tensor(boxes)
        target['masks']=torch.stack(masks)
        return imgs, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, target, now_frames):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, now_frames=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class Test_Show(object):
    def __call__(self, img, target=None, now_frames=None):
        boxes_num = target['boxes'].shape[0]
        boxes = target['boxes'].numpy().astype(np.int32)
        for i in range(now_frames):
            image = img[i]
            draw = PIL.ImageDraw.Draw(image)
            for box_i in range(0 + i, boxes_num, now_frames):
                box = boxes[box_i]
                if (box == 0).all():
                    continue

                draw.rectangle(box.tolist(), outline=(0, 255, 0), width=2)
            del draw
            image.save(f"/lustre/home/jbzhang/SeqFormer/crop_debug/{i}.jpg")

        import pdb; pdb.set_trace()
        return img, target


def MyShift(img, target, offset_w, offset_h, filter_sbox=True):
    w, h = img.size
    img = F.affine(img, 0.0, [offset_w, offset_h], 1.0, 0.0)

    boxes = target["boxes"].numpy()
    ori_boxes = copy.deepcopy(boxes)  # used to filter

    boxes += np.array([offset_w, offset_h, offset_w, offset_h])
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], a_min=0, a_max=w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], a_min=0, a_max=h)

    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    boxes[areas == 0] = 0

    masks = target["masks"]
    for i in range(masks.shape[0]):
        masks[i] = F.affine(masks[i].unsqueeze(0), 0.0, [offset_w, offset_h], 1.0, 0.0).squeeze(0)

    # filter small box based ratios of w and h
    if filter_sbox:
        ori_boxes = ori_boxes.reshape(-1, 2, 2)
        ori_boxes_wh = ori_boxes[:, 1, :] - ori_boxes[:, 0, :]
        boxes_ = boxes.reshape(-1, 2, 2)
        boxes_wh = boxes_[:, 1, :] - boxes_[:, 0, :]
        box_wh_ratios = boxes_wh / (ori_boxes_wh + 1e-6)
        need_filter = (box_wh_ratios <= 0.25).any(1)
        boxes[need_filter] = 0
        areas[need_filter] = 0
        masks[torch.from_numpy(need_filter)] = 0

    target["boxes"] = torch.from_numpy(boxes)
    target["area"] = torch.from_numpy(areas)
    target["masks"] = masks

    return img, target


def MyScale(img, target, scale, filter_sbox=True):
    w, h = img.size
    t_w, t_h = w * scale, h * scale
    img = F.affine(img, 0.0, [0, 0], scale, 0.0)  # 中心缩放，保持原尺度不变

    masks = target["masks"]
    for i in range(masks.shape[0]):
        masks[i] = F.affine(masks[i].unsqueeze(0), 0.0, [0, 0], scale, 0.0).squeeze(0)

    boxes = target["boxes"].numpy()
    boxes *= scale
    ori_boxes = copy.deepcopy(boxes)  # used to filter

    if scale < 1.0:
        shift_w = (w - t_w) / 2
        shift_h = (h - t_h) / 2
        boxes[:, 0::2] += shift_w
        boxes[:, 1::2] += shift_h
    elif scale > 1.0:
        shift_w = (t_w - w) / 2
        shift_h = (t_h - h) / 2
        boxes[:, 0::2] -= shift_w
        boxes[:, 1::2] -= shift_h

    boxes = boxes.astype(np.int32)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], a_min=0, a_max=w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], a_min=0, a_max=h)
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    boxes[areas == 0] = 0

    if filter_sbox:
        ori_boxes = ori_boxes.reshape(-1, 2, 2)
        ori_boxes_wh = ori_boxes[:, 1, :] - ori_boxes[:, 0, :]
        boxes_ = boxes.reshape(-1, 2, 2)
        boxes_wh = boxes_[:, 1, :] - boxes_[:, 0, :]
        box_wh_ratios = boxes_wh / (ori_boxes_wh + 1e-6)
        need_filter = (box_wh_ratios <= 0.25).any(1)
        boxes[need_filter] = 0
        areas[need_filter] = 0
        masks[torch.from_numpy(need_filter)] = 0

    target["boxes"] = torch.from_numpy(boxes)
    target["area"] = torch.from_numpy(areas)
    target["masks"] = masks

    return img, target


class ImgToClip(object):
    def __init__(self, p=0.5,
                 shift_min_ratio=0.1,
                 shift_max_ratio=0.5,
                 scale_min_ratio=0.1,
                 scale_max_ratio=0.5,
                 angle_range=30,
                 filter_sbox=True):
        self.shift_min_ratio = shift_min_ratio
        self.shift_max_ratio = shift_max_ratio
        self.scale_min_ratio = scale_min_ratio
        self.scale_max_ratio = scale_max_ratio
        self.angle_range = angle_range
        self.p = p
        self.filter_sbox = filter_sbox
        self.choice = ["shift", "scale", "shift_scale"]

    def get_shift_step(self, img_wh, now_frames):
        shift_ratio = np.random.uniform(self.shift_min_ratio, self.shift_max_ratio)
        shift_w, shift_h = int(img_wh[0] * shift_ratio), int(img_wh[1] * shift_ratio)
        if np.random.rand() > self.p:
            shift_w *= -1
        if np.random.rand() > self.p:
            shift_h *= -1

        step_w = np.linspace(0, shift_w, now_frames).astype(np.int32)
        step_h = np.linspace(0, shift_h, now_frames).astype(np.int32)

        return step_w, step_h

    def get_scale_step(self, now_frames):
        scale_ratio = np.random.uniform(self.scale_min_ratio, self.scale_max_ratio)
        if np.random.rand() > self.p:
            scale_ratio = 1 + scale_ratio
        else:
            scale_ratio = 1 - scale_ratio

        step_scale = np.linspace(1, scale_ratio, now_frames)

        return step_scale

    def process(self, img_set, target_set, now_frames):
        img_box_num = target_set[0]["boxes"].shape[0]
        _, mask_h, mask_w = target_set[0]["masks"].shape
        clip_boxes = torch.zeros((img_box_num*now_frames, 4), dtype=target_set[0]["boxes"].dtype)
        clip_masks = torch.zeros((img_box_num*now_frames, mask_h, mask_w), dtype=target_set[0]["masks"].dtype)
        clip_areas = torch.zeros(img_box_num*now_frames, dtype=target_set[0]["area"].dtype)
        clip_labels = torch.zeros(img_box_num, dtype=torch.int64)
        clip_iscrowd = torch.zeros(img_box_num*now_frames, dtype=torch.int64)


        for i in range(now_frames):
            clip_boxes[i::now_frames] = target_set[i]["boxes"]
            clip_masks[i::now_frames] = target_set[i]["masks"]
            clip_areas[i::now_frames] = target_set[i]["area"]

        clip_valid = clip_areas != 0
        target = {}
        target["size"] = target_set[0]["size"]
        target["orig_size"] = target_set[0]["orig_size"]
        target["image_id"] = target_set[0]["image_id"]
        target["boxes"] = clip_boxes
        target["masks"] = clip_masks
        target["areas"] = clip_areas
        target["labels"] = clip_labels
        target["iscrowd"] = clip_iscrowd
        target["valid"] = clip_valid

        return img_set, target

    def __call__(self, img: PIL.Image, target, now_frames):
        w, h = img.size
        img_set = [img]
        target_set = [target]

        aug_type = np.random.choice(self.choice)
        step_w, step_h = self.get_shift_step((w, h), now_frames)
        step_scale = self.get_scale_step(now_frames)

        for i in range(1, now_frames):
            img_i = copy.deepcopy(img)
            target_i = copy.deepcopy(target)

            if "shift" in aug_type:
                img_i, target_i = MyShift(img_i, target_i, step_w[i], step_h[i], filter_sbox=self.filter_sbox)
            if "scale" in aug_type:
                img_i, target_i = MyScale(img_i, target_i, step_scale[i], filter_sbox=self.filter_sbox)
            if "angle" in aug_type:
                pass

            img_set.append(img_i)
            target_set.append(target_i)

        return self.process(img_set, target_set, now_frames)


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

    def __call__(self, img, target, now_frames):
        if random.random() < self.p:
            return self.transforms1(img, target, now_frames)
        return self.transforms2(img, target, now_frames)


class ToTensor(object):
    def __call__(self, clip, target, now_frames):
        img = []
        for im in clip:
            img.append(F.to_tensor(im))
        return img, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip, target=None, now_frames=None):
        image = []
        for im in clip:
            image.append(F.normalize(im, mean=self.mean, std=self.std))
        if target is None:
            return image, None
        target = target.copy()
        h, w = image[0].shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            target["boxes_xyxy"] = boxes.clone()
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, now_frames):
        for t in self.transforms:            
            image, target = t(image, target, now_frames)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string