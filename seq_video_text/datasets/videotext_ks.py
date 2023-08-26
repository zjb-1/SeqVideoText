# ------------------------------------------------------------------------
# SeqFormer data loader
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------


from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms_clip as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import numpy as np
import cv2
import random
import math
import time
import pickle


class VideoTextDataset_ks:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames, with_rec, test_mode=False):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self._transforms_BOVText = make_coco_transforms_vt("train", 1)
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.test_mode = test_mode

        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.prepare = ConvertCocoPolysToMask(self.cat2label, return_masks, with_rec)
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)

        if test_mode:  # build test dataset
            self.img_ids = []
            for idx, vid_info in enumerate(self.vid_infos):
                for frame_id in range(len(vid_info['filenames'])):
                    self.img_ids.append((idx, frame_id))

        else:          # build train dataset
            if "BOVText" in self.img_folder.__reduce__()[-1]:
                with open("/lustre/datasharing/jbzhang/data/BOVText/BOV_train_5_img_ids.pkl", 'rb') as f:
                    self.img_ids = pickle.load(f)
            elif "nlpr_video_dataset_new" in self.img_folder.__reduce__()[-1]:
                with open("/lustre/datasharing/jbzhang/nlpr_video_dataset_new/BiRViT1K_img_ids.pkl", 'rb') as f:
                    self.img_ids = pickle.load(f)
            elif "rt_1k" in self.img_folder.__reduce__()[-1]:
                with open("/lustre/datasharing/jbzhang/data/rt_1k/rt1k_img_ids.pkl", 'rb') as f:
                    self.img_ids = pickle.load(f)
            else:
                self.img_ids = []
                for idx, vid_info in enumerate(self.vid_infos):
                    for frame_id in range(len(vid_info['filenames'])):
                        self.img_ids.append((idx, frame_id))

                # filter images with no annotation during training
                valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                    if len(self.get_ann_info(v, f)['bboxes'])]
                self.img_ids = [self.img_ids[i] for i in valid_inds]

        self.img_ids_set = set(self.img_ids)  # when use "in" operation, using 'set' is much faster than using 'list'
        print('\n video num:', len(self.vid_ids), '  clip num:', len(self.img_ids))
        print('\n')

    def __len__(self):
        return len(self.img_ids)

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        gt_bboxes = []
        gt_labels = []
        gt_ids = []

        if with_mask:
            gt_masks = []
        for i, ann in enumerate(ann_info):

            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox == 'None' or bbox is None:
                continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]

            gt_bboxes.append(bbox)
            gt_ids.append(ann['id'])
            gt_labels.append(self.cat2label[ann['category_id']])

            if with_mask:
                gt_masks.append(self.ytvos.annToMask(ann, frame_id))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            obj_ids=gt_ids)
        if with_mask:
            ann['masks'] = gt_masks
        return ann

    def prepare_train_clip(self, idx):
        instance_check = False
        while not instance_check:
            idx = 146570
            import pdb; pdb.set_trace()
            vid, frame_id = self.img_ids[idx]
            vid_id = self.vid_infos[vid]['id']
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames))
            num_frames = self.num_frames
            # random sample
            sample_indx = []
            cur_rand_index = randint(1, self.num_frames)  # 随机采相邻的 num_frames个 images
            for i in range(1, self.num_frames + 1):
                samp_id = min(vid_len - 1, max(0, frame_id + i - cur_rand_index))
                sample_indx.append(samp_id)

            img = []
            for i in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[i]])
                img.append(Image.open(img_path).convert('RGB'))

            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            target = {'video_id': vid, 'annotations': target}
            target_inds = inds
            target = self.prepare(img[0], target, target_inds, sample_inds=sample_indx)
            clip_img_ids = [i - frame_id + idx for i in sample_indx]  # 根据帧id来计算 图像的idx
            target["image_id"] = torch.tensor(clip_img_ids)

            if self._transforms is not None:

                img, target = self._transforms(img, target, num_frames)

            if len(target['labels']) == 0:  # None instance    TODO: waste time
                idx = random.randint(0, self.__len__()-1)
            else:
                instance_check = True
        target['boxes'] = target['boxes'].clamp(1e-6)

        ########### debug
        target['video_id'] = torch.tensor(vid)
        target['sample_indx'] = torch.tensor(sample_indx)
        ##################

        return torch.cat(img, dim=0), target

    def __getitem__(self, idx):
        return self.prepare_train_clip(idx)


def convert_coco_poly_to_mask(segmentations, height, width, is_crowd):
    masks = []
    for i, seg in enumerate(segmentations):
        if seg == 'None' or seg is None:
            mask = torch.zeros((height, width), dtype=torch.uint8)
        else:
            if not is_crowd[i]:
                seg = coco_mask.frPyObjects(seg, height, width)
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
    def __init__(self, cat2label, return_masks=False, with_rec=False):
        self.return_masks = return_masks
        self.cat2label = cat2label
        self.with_rec = with_rec

    def __call__(self, image, target, target_inds, sample_inds):
        w, h = image.size
        video_id = target['video_id']

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        video_len = len(anno[0]['bboxes'])
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        recs = []

        # add valid flag for bboxes
        for i, ann in enumerate(anno):  # boxes number
            inst_flag = False
            boxes_temp = []
            area_temp = []
            segmentations_temp = []
            iscrowd_temp = []
            valid_temp = []
            rec_temp = []
            for id in target_inds:  # frames id
                bbox = ann['bboxes'][sample_inds[id]]
                areas = ann['areas'][sample_inds[id]]
                segm = ann['segmentations'][sample_inds[id]]
                if self.with_rec:
                    assert 'rec' in ann, "annotations don't contain the att: rec"
                    rec = ann["rec"][sample_inds[id]]

                # clas = ann["category_id"]
                # for empty boxes
                if bbox == 'None' or bbox is None:
                    bbox = [0, 0, 0, 0]
                    areas = 0
                    valid_temp.append(0)
                    rec = [0] * 35
                else:
                    valid_temp.append(1)
                    inst_flag = True
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes_temp.append(bbox)
                area_temp.append(areas)
                segmentations_temp.append(segm)
                # classes.append(clas)
                iscrowd_temp.append(crowd)
                if self.with_rec:
                    rec_temp.append(rec)

            if inst_flag:
                boxes.extend(boxes_temp)
                area.extend(area_temp)
                segmentations.extend(segmentations_temp)
                iscrowd.extend(iscrowd_temp)
                valid.extend(valid_temp)
                classes.append(self.cat2label[ann["category_id"]])
                if self.with_rec:
                    recs.extend(rec_temp)

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w, iscrowd)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks

        # image_id = [sample_inds[id] + video_id * 1000 for id in target_inds]  # TODO:  *1000 ??
        # image_id = torch.tensor(image_id)
        # target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area)
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        if self.with_rec:
            target["rec"] = torch.tensor(recs, dtype=torch.int64)
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return target


def make_coco_transforms_vt(image_set, scale=0):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [296, 328, 360, 392]
    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        if scale == 0:
            return T.Compose([
                # T.RandomHorizontalFlip(),
                T.PhotometricDistort_(),
                T.RandomCropWithInstance(0.5, 1.0, filter_sbox=True, crop_instance=False),
                # T.RandomResize([608, 640, 672, 736, 768], 1400),
                T.RandomResize([608, 640, 672, 736], 1333),     # [608, 640, 672, 736, 768], 1333
                # T.Test_Show(),
                T.Check(),
                # T.RandomSelect(
                #     T.Compose([
                #         T.RandomResize(scales, max_size=768),
                #         T.Check(),
                #     ]),
                #     T.Compose([
                #         T.RandomResize([400, 500, 600]),
                #         T.RandomSizeCrop(384, 600),
                #         T.RandomResize(scales, max_size=768),
                #         T.Check(),
                #     ])
                # ),
                normalize,
            ])
        else:
            return T.Compose([
                # T.RandomHorizontalFlip(),
                T.PhotometricDistort_(),
                T.RandomCropWithInstance(0.5, 1.0, filter_sbox=True, crop_instance=False),
                # T.RandomResize([608, 640, 672, 736, 768], 1400),
                T.RandomResize([448, 480, 512], 960),     # [608, 640, 672, 736, 768], 1333
                # T.Test_Show(),
                T.Check(),
                # T.RandomSelect(
                #     T.Compose([
                #         T.RandomResize(scales, max_size=768),
                #         T.Check(),
                #     ]),
                #     T.Compose([
                #         T.RandomResize([400, 500, 600]),
                #         T.RandomSizeCrop(384, 600),
                #         T.RandomResize(scales, max_size=768),
                #         T.Check(),
                #     ])
                # ),
                normalize,
            ])

    if image_set == 'test':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [296, 328, 360, 392]
    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, ds_ind=0):
    root = Path(args.ytvis_path)
    assert root.exists(), f'provided path {root} does not exist'

    assert (args.test_img_prefix != None and args.test_ann_file != None) or \
           (args.train_img_prefix != None and args.train_ann_file != None), 'please provide the file path'
    PATHS = {
        "test": (root / args.test_img_prefix, root / args.test_ann_file),
    }
    if image_set == "train":
        PATHS["train"] = (root / args.train_img_prefix[ds_ind], root / args.train_ann_file[ds_ind])

    img_folder, ann_file = PATHS[image_set]
    print('use VideoText dataset')
    dataset = VideoTextDataset_ks(img_folder, ann_file, transforms=make_coco_transforms_vt(image_set, 0),
                                return_masks=args.masks, num_frames=args.num_frames,
                                with_rec=args.rec, test_mode=image_set=="test")


    return dataset
