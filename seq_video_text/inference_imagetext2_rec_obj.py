'''
Inference code for SeqFormer
'''
import argparse
import datetime
import json
import random
import time
import shutil
import re
from pathlib import Path

import numpy as np
import torch

import datasets
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from util.mask_ops import mask_to_quads
from models import build_model
import torchvision.transforms as T
from torchvision.ops import nms  # BC-compat
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
import math
import torch.nn.functional as F
import json
import pycocotools.mask as mask_util
import sys
import cv2
from eval_tools import eval_PRF


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

    # * object confidence
    parser.add_argument('--obj', action='store_true',
                        help="Train object head if the flag is provided")

    # * recognition
    parser.add_argument('--rec', action='store_true',
                        help="Train recognition head if the flag is provided")
    parser.add_argument('--roi_mask_size', default=28, type=int)
    parser.add_argument('--rec_batch_size', default=128, type=int,
                        help="the batch size of the recognize head")
    parser.add_argument('--rec_num_classes', default=107, type=int,
                        help="the classes number of the recognize head")
    parser.add_argument('--rec_resolution_w', default=32, type=int,
                        help="the width of the recognize head")
    parser.add_argument('--rec_resolution_h', default=32, type=int,
                        help="the height of the recognize head")

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
    parser.add_argument('--rec_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=5, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--img_path', default=None, type=str)
    parser.add_argument('--ann_path', default=None, type=str)
    parser.add_argument('--submit_path', default="./submit")
    parser.add_argument('--gt_zip_path', default=None, type=str)
    parser.add_argument('--show_path', default=None, type=str)
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='YoutubeVIS')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--confidence_threshold', default=0.4, type=float)

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--det_res_save_path', default=None)
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
    T.Resize(960),
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
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.detr.num_frames = 1
        num_classes = model.detr.num_classes

        folder = args.img_path
        images = json.load(open(args.ann_path, 'rb'))['images']
        img_num = len(images)
        result = []

        for i in range(img_num):
            print("Process img: ", i)
            id_ = images[i]['id']
            file_name = images[i]['file_name']
            im = Image.open(os.path.join(folder, file_name))
            im = ImageOps.exif_transpose(im)

            w, h = im.size
            img_set = [transform(im).unsqueeze(0).cuda()]
            img = torch.cat(img_set, 0)

            outputs = model.inference(img, img.shape[-1], img.shape[-2])
            logits = outputs['pred_logits'][0]  # [query_num, num_classes]
            output_mask = outputs['pred_masks'][0]  # [query_num, frames_num, h, w]
            output_boxes = outputs['pred_boxes'][0]  # [frames_num, query_num, 4]
            output_objs = outputs["pred_objs"][0].squeeze(-1)  # [frames_num, query_num]
            if args.rec:
                output_rec = outputs['pred_rec'][0]  # [frames_num, query_num, 100]
                pred_recs = output_rec.cpu().detach()
            scores = logits.sigmoid().cpu().detach().numpy()  # [query_num, 1]
            pred_boxes = output_boxes.cpu().detach()
            pred_masks = output_mask.cpu().detach()
            pred_objs = output_objs.sigmoid().cpu().detach()

            # first filter, topk
            topkv, indices = torch.topk(logits.sigmoid().cpu().detach().flatten(0), k=200)
            box_pred_per_image = pred_boxes.unsqueeze(-2).repeat(1, 1, num_classes, 1).flatten(1, 2)  # [frames_num, query_num * num_class, 4]
            mask_pred_per_image = pred_masks.unsqueeze(1).repeat(1, num_classes, 1, 1, 1).flatten(0, 1)
            if args.rec:
                rec_pred_per_image = pred_recs.unsqueeze(-2).repeat(1, 1, num_classes, 1).flatten(1, 2)

            first_boxes = box_pred_per_image[:, indices, :]
            first_objs = pred_objs[:, indices]
            first_masks = mask_pred_per_image[indices]
            if args.rec:
                first_recs = rec_pred_per_image[:, indices, :]

            # second filter, confidence
            flag_class_process = True
            if flag_class_process:
                ind2 = topkv >= args.confidence_threshold
                topkv = topkv[ind2]
                first_boxes = first_boxes[:, ind2, :]
                first_masks = first_masks[ind2]
                if args.rec:
                    first_recs = first_recs[:, ind2, :]
            else:
                objs_index = first_objs < args.confidence_threshold
                first_objs[objs_index] = 0.0
                first_boxes[objs_index] = 1e-7
                first_masks[objs_index.transpose(0, 1)] = -1e3
                if args.rec:
                    first_recs[objs_index] = 0

            # third filter based box coor
            thres = 1e-4
            cool_bool = first_boxes > thres  # [act_frame_num, box_num, 4]
            cool_bool_trans = cool_bool.permute(1, 0, 2).flatten(1)   # [box_num, act_frame_num * 4]
            boxes_index = cool_bool_trans.any(1)  # [box_num]
            second_boxes = first_boxes[:, boxes_index, :]  # [act_frame_num, box_num_, 4]
            second_masks = first_masks[boxes_index]  # [box_num_, act_frame_num, h, w]
            if flag_class_process:
                boxes_score = topkv[boxes_index]  # [box_num_, classes]
            else:
                boxes_score = first_objs[:, boxes_index]
            if args.rec:
                second_recs = first_recs[:, boxes_index, :]

            # obtain the final boxes
            saved_boxes = second_boxes * torch.tensor([w, h, w, h])
            saved_boxes = box_cxcywh_to_xyxy(saved_boxes)
            saved_boxes.round_()
            saved_boxes[:, :, 0::2] = saved_boxes[:, :, 0::2].clamp(0, w)
            saved_boxes[:, :, 1::2] = saved_boxes[:, :, 1::2].clamp(0, h)

            # obtain the final mask of each box
            saved_masks = F.interpolate(second_masks, (h, w), mode="bilinear").sigmoid()
            saved_masks = saved_masks > 0.5

            # obtain the final recognize of each box
            saved_recs = second_recs if args.rec else None
            # if args.show_path is not None:
            #     PostProcess_show(folder, file_names, cur_frames_s, saved_boxes, args.show_path)

            PostProcess(folder, [file_name], i, i, saved_boxes, saved_masks, saved_recs,
                        boxes_score, args.show_path, args.submit_path,
                        submit_mode="mask", nms_iou=-1, draw_masks=False)

        # eval P R F
        if args.gt_zip_path is not None:
            res = eval_PRF(args.submit_path, args.gt_zip_path)  # eval P R F
            test_res = f"p:{res[0]:0.5f}, r:{res[1]:0.5f}, f1:{res[2]:0.5f} \n"
            if args.det_res_save_path is not None and re.search(r"\.txt$", args.det_res_save_path) is not None:
                with open(args.det_res_save_path, 'a') as f:
                    f.write(test_res)


def draw_box_mask(img, mask, color, alpha=0.5):
    """
    :param img:
    :param mask: [h, w]
    :param color:
    :param alpha:
    :return:
    """
    assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1], "size error"
    assert len(color) == 3
    mask_3d = mask[..., np.newaxis].repeat(3, axis=-1)  # [h, w, 3]
    mask_3d[mask == 1] = color

    region = (mask != 0).astype(bool)
    img[region] = img[region] * alpha + mask_3d[region] * (1 - alpha)

    return img


def draw_text(img, bbox, irec, b_score):
    def _decode_recognition(rec):
        # CTLABELS = "_0123456789abcdefghijklmnopqrstuvwxyz"
        CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[',
                    '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
        # ctc decoding
        last_char = False
        s = ''
        for c in rec:
            c = int(c)
            if 0 < c < 96:
                # if last_char != c:
                if CTLABELS[c - 1] in "_0123456789abcdefghijklmnopqrstuvwxyz":
                    s += CTLABELS[c - 1]
                    last_char = c
            elif c == 96:
                s += u''
            elif c == 97:
                if len(s) == 0:
                    s = ' '
                return s
        if len(s) == 0:
            s = ' '
        return s

    b_str = ""
    if b_score is not None:
        b_str = b_str + "%.2f" % b_score
    if irec is not None:
        rec = irec.copy()
        text = _decode_recognition(rec)
        b_str = b_str + "|" + text

    if len(b_str) > 0 and b_str != ' ':
        text_pos = (bbox[0], bbox[1])
        img = cv2.putText(img, b_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA,)

    return img


def PostProcess_show(folder, file_names, frame_index_s, pred_bboxes: np.ndarray, save_root,
                     pred_masks=None, pred_recs=None, draw_masks=False, pred_scores=None):
    """
    :param pred_scores:
    :param pred_recs:  [box_num, word_max_len]
    :param draw_masks:
    :param folder:
    :param file_names:
    :param frame_index_s:
    :param pred_bboxes:  [box_num, 4]   numpy
    :param pred_masks:  [box_num, h, w]   numpy
    :param save_root:
    :return:
    """
    if pred_bboxes.ndim == 2:
        pred_bboxes = pred_bboxes[np.newaxis]
    pred_bboxes = pred_bboxes.astype(np.int32)

    if pred_masks is not None and pred_masks.ndim == 3:
        pred_masks = pred_masks[np.newaxis]
        pred_masks = pred_masks.astype(np.uint8)

    if pred_recs is not None and pred_recs.ndim == 2:
        pred_recs = pred_recs[np.newaxis]
        pred_recs = pred_recs.astype(np.int32)

    if pred_scores is not None and pred_scores.ndim == 1:
        pred_scores = pred_scores[np.newaxis]

    frames_num = pred_bboxes.shape[0]

    for i in range(frames_num):
        img_path = os.path.join(folder, file_names[i])
        img_name = file_names[i]
        save_path = os.path.join(save_root, img_name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        img = cv2.imread(img_path)
        frame_boxes = pred_bboxes[i]  # [box_num, 4]

        for b in range(frame_boxes.shape[0]):
            bbox = frame_boxes[b]
            if (bbox < 1).all():
                continue

            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
            if draw_masks and pred_masks is not None:
                img = draw_box_mask(img, pred_masks[i][b], color=(50, 200, 50))

            d_rec = pred_recs[i][b] if pred_recs is not None else None
            d_score = pred_scores[i][b] if pred_scores is not None else None
            img = draw_text(img, bbox, d_rec, d_score)

        cv2.imwrite(save_path, img)
        print(save_path)


def PostProcess(folder,
                file_names,
                frame_index_s,
                img_idx_based_dataset,
                pred_bboxes,
                pred_masks,
                pred_recs,
                boxes_score,
                save_root,
                submit_path,
                submit_mode="det",
                nms_iou=-1.0,
                draw_masks=False):
    """
    :param pred_recs: [fn, box_num, word_max_len]
    :param submit_mode: det result or mask based result
    :param draw_masks: draw mask in image
    :param folder:
    :param file_names:
    :param frame_index_s:
    :param img_idx_based_dataset:
    :param pred_bboxes: [fn, box_num, 4]
    :param pred_masks: [box_num, fn, h, w]
    :param boxes_score: [box_num]
    :param save_root:
    :param submit_path:
    :param nms_iou:
    :return:
    """
    assert pred_bboxes.dim() == 3, "dim error"
    assert submit_mode in ["det", "mask"], "only 'det' or 'mask'"
    frames_num = pred_bboxes.shape[0]
    if boxes_score.dim() == 1:   # sequence class confidence
        boxes_score = boxes_score.repeat(frames_num, 1)

    for i in range(frames_num):
        frame_boxes = pred_bboxes[i]  # [box_num, 4]
        frame_masks = pred_masks[:, i]  # [box_num, h, w]

        ind = (frame_boxes >= 1).any(1)
        filter_boxes = frame_boxes[ind]  # 把 4个点是1e-7的box过滤掉
        filter_masks = frame_masks[ind]
        if pred_recs is not None:
            filter_recs = pred_recs[i][ind]
        frame_box_score = boxes_score[i][ind]

        if submit_path is not None:
            txt_name = 'res_img_' + str(img_idx_based_dataset + i + 1) + '.txt'
            txt_path = os.path.join(submit_path, txt_name)
            with open(txt_path, 'w') as f:  # 创建/覆盖 为空文件
                f.write('')
        if filter_boxes.shape[0] == 0:
            if save_root is not None:
                PostProcess_show(folder, file_names, frame_index_s + i, filter_boxes.numpy(), save_root)
            continue

        if nms_iou >= 0.0:
            box_keep = nms(filter_boxes, frame_box_score, nms_iou)
        else:  # no nms
            box_keep = torch.arange(filter_boxes.shape[0], dtype=torch.int64)

        assert box_keep.dim() == 1, "nms output dim error"
        nms_boxes = filter_boxes[box_keep].numpy()
        nms_masks = filter_masks[box_keep].numpy()
        if pred_recs is not None:
            nms_recs = filter_recs[box_keep].numpy()
        else:
            nms_recs = None
        nms_box_score = frame_box_score[box_keep].numpy()

        if submit_path is not None:
            if submit_mode == "det":
                for box_i in range(nms_boxes.shape[0]):
                    box = nms_boxes[box_i].tolist()[:4]
                    box = list(map(round, box))

                    s = f"{box[0]},{box[1]},{box[2]},{box[1]},{box[2]},{box[3]},{box[0]},{box[3]}\n"
                    with open(txt_path, 'a') as f:
                        f.write(s)
            else:
                for mask_i in range(nms_masks.shape[0]):
                    mask = nms_masks[mask_i]

                    quad = mask_to_quads(mask, mask.shape[0], mask.shape[1])
                    if quad is None:
                        continue
                    s = ",".join(map(str, quad)) + "\n"
                    with open(txt_path, 'a') as f:
                        f.write(s)

        if save_root is not None:
            PostProcess_show(folder, file_names, frame_index_s + i, nms_boxes, save_root, nms_masks, nms_recs,
                             draw_masks=draw_masks, pred_scores=nms_box_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
