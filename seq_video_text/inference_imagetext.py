'''
Inference code for SeqFormer
'''
import argparse
import datetime
import json
import random
import time
import shutil
from pathlib import Path

import numpy as np
import torch

import datasets
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from models import build_model
import torchvision.transforms as T
from torchvision.ops import nms  # BC-compat
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import json
import pycocotools.mask as mask_util
import sys
import cv2
from evaluate_PRF import eval_PRF


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--with_box_refine', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone

    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--rel_coord', default=True, action='store_true')

    # Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_out_stride', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--img_path', default="/lustre/data_sharing/jbzhang/data/ICDAR2015Video/data/ch3_test/")
    parser.add_argument('--ann_path', default="/lustre/data_sharing/jbzhang/data/ICDAR2015Video/data/ch3_test.json/")
    parser.add_argument('--submit_path', default="./submit")
    parser.add_argument('--gt_zip_path', default=None, type=str)
    parser.add_argument('--show_path', default=None, type=str)
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='YoutubeVIS')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


# CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
#          'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
#          'train','horse','turtle','bear','motorbike','giraffe','leopard',
#          'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
#          'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
#          'tennis_racket']

CLASSES = ['person']

transform = T.Compose([
    T.Resize(640),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if os.path.exists(args.submit_path):
        shutil.rmtree(args.submit_path)
    os.mkdir(args.submit_path)

    if args.show_path is not None:
        if os.path.exists(args.show_path):
            shutil.rmtree(args.show_path)
            print("remove the old show root file")
        os.mkdir(args.show_path)

    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        model.eval()
        model.detr.num_frames = 1

        folder = args.img_path
        images = json.load(open(args.ann_path, 'rb'))['images']
        img_num = len(images)
        result = []

        for i in range(img_num):
            print("Process img: ", i)
            id_ = images[i]['id']
            file_name = images[i]['file_name']
            im = Image.open(os.path.join(folder, file_name))
            w, h = im.size
            img_set = [transform(im).unsqueeze(0).cuda()]
            img = torch.cat(img_set, 0)

            outputs = model.inference(img, img.shape[-1], img.shape[-2])
            logits = outputs['pred_logits'][0]  # [query_num, 1]
            output_mask = outputs['pred_masks'][0]  # [query_num, frames_num, h, w]
            output_boxes = outputs['pred_boxes'][0]  # [frames_num, query_num, 4]
            assert logits.shape[1] == 1, "multi-class, need revise the process"
            scores = logits.sigmoid().cpu().detach().numpy()  # [query_num, 1]
            pred_boxes = output_boxes.cpu().detach()

            # first filter based score
            topkv, indices = torch.topk(logits.sigmoid().cpu().detach().flatten(0), k=100)
            first_boxes = pred_boxes[:, indices, :]
            import pdb; pdb.set_trace()

            # second filter based box coor
            thres = 1e-4
            cool_bool = first_boxes > thres  # [act_frame_num, box_num, 4]
            cool_bool_trans = cool_bool.permute(1, 0, 2).flatten(1)   # [box_num, act_frame_num * 4]
            boxes_index = cool_bool_trans.any(1)  # [box_num]
            second_boxes = first_boxes[:, boxes_index, :]  # [act_frame_num, box_num_, 4]
            boxes_score = topkv[boxes_index]  # [box_num_, classes]

            saved_boxes = second_boxes * torch.tensor([w, h, w, h])
            saved_boxes = box_cxcywh_to_xyxy(saved_boxes)
            saved_boxes.round_()
            saved_boxes[:, :, 0::2] = saved_boxes[:, :, 0::2].clamp(0, w)
            saved_boxes[:, :, 1::2] = saved_boxes[:, :, 1::2].clamp(0, h)

            # if args.show_path is not None:
            #     PostProcess_show(folder, file_names, cur_frames_s, saved_boxes, args.show_path)

            PostProcess(folder, [file_name], i, i, saved_boxes,
                        boxes_score, args.show_path, args.submit_path, nms_iou=0.5)

        # eval P R F
        if args.gt_zip_path is not None:
            res = eval_PRF(args.submit_path, args.gt_zip_path)  # eval P R F


def PostProcess_show(folder, file_names, frame_index_s, pred_bboxes: np.ndarray, save_root):
    """
    :param folder:
    :param file_names:
    :param frame_index_s:
    :param pred_bboxes:  [fn, box_num, 4]   numpy
    :param save_root:
    :return:
    """
    if pred_bboxes.ndim == 2:
        pred_bboxes = pred_bboxes[np.newaxis, :]
    pred_bboxes = pred_bboxes.astype(np.int32)

    frames_num = pred_bboxes.shape[0]

    for i in range(frames_num):
        img_path = os.path.join(folder, file_names[i])
        img_name = file_names[i]
        save_path = os.path.join(save_root, img_name)

        img = cv2.imread(img_path)
        frame_boxes = pred_bboxes[i]  # [box_num, 4]

        for b in range(frame_boxes.shape[0]):
            bbox = frame_boxes[b]
            if (bbox < 1).all():
                continue

            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)

        cv2.imwrite(save_path, img)
        print(save_path)


def PostProcess(folder,
                file_names,
                frame_index_s,
                img_idx_based_dataset,
                pred_bboxes,
                boxes_score,
                save_root,
                submit_path,
                nms_iou=0.5):
    """
    :param folder:
    :param file_names:
    :param frame_index_s:
    :param img_idx_based_dataset:
    :param pred_bboxes: [fn, box_num, 4]
    :param boxes_score: [box_num]
    :param save_root:
    :param submit_path:
    :param nms_iou:
    :return:
    """
    assert pred_bboxes.dim() == 3, "dim error"
    frames_num = pred_bboxes.shape[0]

    for i in range(frames_num):
        frame_boxes = pred_bboxes[i]  # [box_num, 4]

        ind = (frame_boxes >= 1).any(1)
        filter_boxes = frame_boxes[ind]  # 把 4个点是1e-7的box过滤掉
        frame_box_score = boxes_score[ind]

        if submit_path is not None:
            txt_name = 'res_img_' + str(img_idx_based_dataset + i + 1) + '.txt'
            txt_path = os.path.join(submit_path, txt_name)
            with open(txt_path, 'w') as f:  # 创建/覆盖 为空文件
                f.write('')
        if filter_boxes.shape[0] == 0:
            if save_root is not None:
                PostProcess_show(folder, file_names, frame_index_s + i, filter_boxes.numpy(), save_root)
            continue

        box_keep = nms(filter_boxes, frame_box_score, nms_iou)
        print(filter_boxes.shape[0] - len(box_keep))
        assert box_keep.dim() == 1, "nms output dim error"
        nms_boxes = filter_boxes[box_keep].numpy()
        nms_box_score = frame_box_score[box_keep].numpy()

        if submit_path is not None:
            for box_i in range(nms_boxes.shape[0]):
                box = nms_boxes[box_i].tolist()[:4]
                box = list(map(round, box))

                s = f"{box[0]},{box[1]},{box[2]},{box[1]},{box[2]},{box[3]},{box[0]},{box[3]}\n"
                with open(txt_path, 'a') as f:
                    f.write(s)

        if save_root is not None:
            PostProcess_show(folder, file_names, frame_index_s + i, nms_boxes, save_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
