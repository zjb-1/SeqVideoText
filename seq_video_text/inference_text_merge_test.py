'''
Inference code for SeqFormer
'''
import argparse
import datetime
import json
import random
import time
import shutil
import pickle
import copy
from pathlib import Path

import numpy as np
import torch
import mmcv

from datasets import build_dataset
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from models import build_model, Clips, Videos
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
from eval_tools import eval_PRF, results2json_videoseg, results2json_mota, eval_mot, results2submit


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

    parser.add_argument('--is_ks', action='store_true',
                        help="Train object head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

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
    parser.add_argument('--obj_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--dataset_name_need_small_scale', default=None, nargs='+')

    # dataset parameters
    parser.add_argument('--ytvis_path', type=str)
    parser.add_argument('--test_img_prefix', type=str)
    parser.add_argument('--test_ann_file', type=str)
    parser.add_argument('--submit_path', default="./submit")
    parser.add_argument('--gt_zip_path', default=None, type=str)
    parser.add_argument('--eval_mota_pred_json', default=None, type=str)
    parser.add_argument('--eval_mota_gt_json', default=None, type=str)
    parser.add_argument('--show_path', default=None, type=str)
    parser.add_argument('--show_mode', default=None, type=str)
    parser.add_argument('--out', default='output.pkl')
    parser.add_argument('--with_out', action='store_true',
                        help='using existing result ')
    parser.add_argument('--dataset_file', default='VideoText')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--clip_length', default=5, type=int)
    parser.add_argument('--clip_stride', default=5, type=int)
    parser.add_argument('--confidence_threshold', default=0.4, type=float)

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
    parser.add_argument('--eval_results_save_path', default=None)

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
    T.Resize(768),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    assert args.clip_length >= args.clip_stride, "clip length must large than clip stride"
    args.num_frames = args.clip_length
    device = torch.device(args.device)

    folder = os.path.join(args.ytvis_path, args.test_img_prefix)
    ann_path = os.path.join(args.ytvis_path, args.test_ann_file)

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

    # * cal fps
    fps_sum_frames = 0
    fps_frames_dataset = 0
    fps_sum_time = 0.0
    fps_clip_count = 0
    fps_skip_clips = 5
    fps_dict = {"one_clip_fps": 0.0, "fast_one_clip_fps": 0.0, "now_fps": 0.0, "fast_fps": 0.0,
                "fps_dataset": 0.0, "fast_fps_dataset": 0.0}

    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path)['model']
        print(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()

        # used to eval
        dataset = build_dataset("test", args)
        print("dataset class init success")

        videos = json.load(open(ann_path, 'rb'))['videos']
        vis_num = len(videos)
        # postprocess = PostProcessSegm_ifc()
        results = []
        overlap_num = args.clip_length - args.clip_stride  # 交叉帧的长度

        if args.with_out:
            results = mmcv.load(args.out)
        else:
            for i in range(vis_num):
                print("Process video: ", i)
                id_ = videos[i]['id']
                vid_len = videos[i]['length']
                if vid_len == 0:
                    continue
                file_names = videos[i]['file_names']
                # if file_names[0].split("/")[0] != "4":
                #     continue

                video_output = Videos(folder, file_names, vid_len,
                                      (videos[i]['height'], videos[i]['width']), overlap_num, args.show_path)

                flag_video_end = False
                for cur_frames_s in range(0, vid_len, args.clip_stride):
                    act_frame_num = args.clip_length if (cur_frames_s + args.clip_length) <= vid_len else vid_len - cur_frames_s
                    if cur_frames_s + act_frame_num >= vid_len:  # 视频最后一段，将要结束
                        flag_video_end = True

                    img_set = []
                    for k in range(act_frame_num):
                        im = Image.open(os.path.join(folder, file_names[cur_frames_s + k]))
                        im = ImageOps.exif_transpose(im)
                        w, h = im.size
                        sizes = torch.as_tensor([int(h), int(w)])
                        img_set.append(transform(im).unsqueeze(0).cuda())

                    img = torch.cat(img_set, 0)
                    model.detr.num_frames = act_frame_num

                    # * fps start
                    fps_clip_count += 1
                    fps_temp_start_time = time.time()
                    ###########
                    outputs = model.inference(img, img.shape[-1], img.shape[-2])
                    # * fps end
                    if fps_clip_count > fps_skip_clips:
                        if fps_clip_count == fps_skip_clips + 1:
                            skip_frames = cur_frames_s
                        fps_one_clip_time = time.time() - fps_temp_start_time
                        fps_one = act_frame_num / fps_one_clip_time

                        fps_sum_frames += act_frame_num
                        fps_sum_time += fps_one_clip_time
                        fps_total = fps_sum_frames / fps_sum_time
                        fps_dataset = (fps_frames_dataset + cur_frames_s + act_frame_num - skip_frames) / fps_sum_time

                        fps_dict["fast_one_clip_fps"] = max(round(fps_one, 1), fps_dict["fast_one_clip_fps"])
                        fps_dict["one_clip_fps"] = round(fps_one, 1)
                        fps_dict["fast_fps"] = max(round(fps_total, 1), fps_dict["fast_fps"])
                        fps_dict["now_fps"] = round(fps_total, 1)
                        fps_dict["fast_fps_dataset"] = max(round(fps_dataset, 1), fps_dict["fast_fps_dataset"])
                        fps_dict["fps_dataset"] = round(fps_dataset, 1)

                        # print(fps_dict)
                    ###########
                    logits = outputs['pred_logits'][0]
                    output_mask = outputs['pred_masks'][0]  # [query_num, frames_num, h, w]
                    output_boxes = outputs['pred_boxes'][0]  # [frames_num, query_num, 4]
                    output_objs = outputs["pred_objs"][0]  # [frames_num, query_num, 1]
                    if args.rec:
                        output_rec = outputs['pred_rec'][0]  # [frames_num, query_num, 100]
                        pred_recs = output_rec.cpu().detach()
                    assert logits.shape[1] == 1, "multi-class need revise the process"
                    assert output_objs.shape[-1] == 1, "multi-class need revise the process"
                    output_objs.squeeze_(-1)    # [frames_num, query_num]

                    scores = logits.sigmoid().cpu().detach()  # [query_num, 1]
                    pred_boxes = output_boxes.cpu().detach()  # [act_frame_num, query_num, 4]
                    pred_masks = output_mask.cpu().detach()
                    pred_objs = output_objs.sigmoid().cpu().detach()

                    # first filter based instance score
                    # topkv, indices = torch.topk(logits.sigmoid().cpu().detach().flatten(0), k=100)
                    first_boxes = pred_boxes
                    first_objs = pred_objs
                    first_masks = pred_masks
                    first_recs = pred_recs if args.rec else None
                    # if int(file_names[cur_frames_s].split("/")[-1].split(".")[0]) >= 127:
                    #     import pdb; pdb.set_trace()
                    # second filter, confidence
                    flag_class_process = True
                    if flag_class_process:
                        scores = scores.flatten(0)
                        ind2 = scores >= args.confidence_threshold
                        topkv = scores[ind2]
                        first_boxes = first_boxes[:, ind2, :]
                        first_masks = first_masks[ind2]
                        first_objs = first_objs[:, ind2]
                        if args.rec:
                            first_recs = first_recs[:, ind2, :]
                    else:
                        objs_index = first_objs < args.confidence_threshold
                        first_objs[objs_index] = 0.0
                        first_boxes[objs_index] = 1e-7
                        first_masks[objs_index.transpose(0, 1)] = -1e3
                        if args.rec:
                            first_recs[objs_index] = 0

                    # second filter based box coor
                    second_boxes, second_masks, second_objs, second_recs, boxes_index = Filter_boxes_on_clip(first_boxes, first_masks, first_objs, first_recs, thres=1e-3)
                    if flag_class_process:
                        boxes_score = topkv[boxes_index]  # [box_num_, classes]
                    else:
                        boxes_score = first_objs[:, boxes_index]

                    saved_boxes = second_boxes * torch.tensor([w, h, w, h])
                    saved_boxes = box_cxcywh_to_xyxy(saved_boxes)
                    saved_boxes.round_()
                    saved_boxes[:, :, 0::2] = saved_boxes[:, :, 0::2].clamp(0, w)
                    saved_boxes[:, :, 1::2] = saved_boxes[:, :, 1::2].clamp(0, h)

                    # 进行nms处理，不丢掉，只是把参数置0
                    # saved_boxes, saved_masks = Handle_boxes_by_nms(saved_boxes, second_masks, boxes_score, nms_iou=0.5)
                    # filter based box coor again
                    saved_boxes, saved_masks, saved_objs, saved_recs, boxes_index = Filter_boxes_on_clip(saved_boxes, second_masks, second_objs, second_recs, thres=0)
                    if flag_class_process:
                        boxes_score = boxes_score[boxes_index]  # [box_num_, classes]
                    else:
                        boxes_score = boxes_score[:, boxes_index][0]

                    clip_results = Clips(cur_frames_s, act_frame_num, boxes_score, saved_boxes, saved_masks, saved_objs, saved_recs)
                    video_output.update(clip_results, result_mode=args.show_mode, box_color=False)

                    if flag_video_end:
                        break

                results.extend(video_output.get_video_results())
                fps_frames_dataset += vid_len

            # save the param: results
            print(f'\nwriting results to {args.out}')
            with open(args.out, 'wb') as f:
                pickle.dump(results, f)

        print('Starting evaluate...')
        result_file = args.out + '.json'
        results2json_videoseg(dataset, results, result_file)

        if args.eval_mota_pred_json is not None:
            results2json_mota(dataset, results, args.eval_mota_pred_json, mode=args.show_mode)  # 生成测试mota的json文件
            if args.eval_mota_gt_json is not None:
                res = eval_mot(args.eval_mota_pred_json, args.eval_mota_gt_json)  # eval mota motp
                if args.eval_results_save_path is not None:
                    test_track = f"-------------------\n mota:{res[0]:0.4f}, motp:{res[1]:0.4f} \n"
                    with open(args.eval_results_save_path, 'a') as f:
                        f.write(test_track)

        # eval P R F
        if args.gt_zip_path is not None:
            results2submit(dataset, results, args.submit_path, mode=args.show_mode)  # 生成测试P R F 的txt文件
            res = eval_PRF(args.submit_path, args.gt_zip_path)  # eval P R F
            if args.eval_results_save_path is not None:
                test_det = f"p:{res[0]:0.5f}, r:{res[1]:0.5f}, f1:{res[2]:0.5f} \n"
                with open(args.eval_results_save_path, 'a') as f:
                    f.write(test_det)


def Filter_boxes_on_clip(clip_boxes, clip_masks, clip_objs, clip_recs=None, thres=1e-3):
    """
    删除 没有用的 query ，即 clip box坐标都为0或者 1e-7
    :param thres:
    :param clip_objs:   #
    :param clip_recs:   # [fn, box_num, word_length]
    :param clip_boxes:  # [fn, box_num, 4]
    :param clip_masks:  # [box_num, fn, h, w]
    :return:
    """
    cool_bool = clip_boxes > thres  # [fn, box_num, 4]
    cool_bool_trans = cool_bool.permute(1, 0, 2).flatten(1)  # [box_num, fn * 4]
    boxes_index = cool_bool_trans.any(1)  # [box_num]
    f_boxes = clip_boxes[:, boxes_index, :]  # [act_frame_num, box_num_, 4]
    f_masks = clip_masks[boxes_index]  # [box_num_, act_frame_num, h, w]
    f_objs = clip_objs[:, boxes_index]  # [act_frame_num, box_num_]

    f_recs = clip_recs[:, boxes_index, :] if clip_recs is not None else None  # [act_frame_num, box_num_, word_length]

    return f_boxes, f_masks, f_objs, f_recs, boxes_index


def Handle_boxes_by_nms(pred_bboxes, pred_masks, boxes_score, nms_iou=0.5):
    """
    对clip中的frame进行nms, 把过滤的box 坐标置0, mask 置 -1000
    :param pred_bboxes:   [fn, box_num, 4]
    :param pred_masks:    [box_num, fn, h, w]
    :param boxes_score:   [box_num]
    :param nms_iou:
    :return:
    """
    frames_num = pred_bboxes.shape[0]
    pred_boxes_copy = copy.deepcopy(pred_bboxes)
    pred_masks_copy = copy.deepcopy(pred_masks)

    for i in range(frames_num):
        frame_boxes = pred_bboxes[i]  # [box_num, 4]
        box_bool = (frame_boxes >= 1).any(1)

        pred_masks_copy[~box_bool, i] = -1e3  # 经过sigmoid后 接近0
        filter_boxes = frame_boxes[box_bool]  # 把 4个点是1e-7的box过滤掉
        frame_box_score = boxes_score[box_bool]
        box_index = torch.arange(0, box_bool.shape[0], dtype=torch.long)[box_bool]  # 真box 的index

        box_keep_index = nms(filter_boxes, frame_box_score, nms_iou)
        temp = torch.ones_like(box_index, dtype=torch.bool)
        temp[box_keep_index] = False

        nmsed_box_index = box_index[temp]  # 找到filter后 被nms的 box index
        pred_boxes_copy[i, nmsed_box_index] = 0  # 把nms的box坐标置0
        pred_masks_copy[nmsed_box_index, i] = -1e3  # mask置 -1000

    return pred_boxes_copy, pred_masks_copy


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
        img_path = os.path.join(folder, file_names[frame_index_s + i])
        video_name, img_name = file_names[frame_index_s + i].split("/")
        if not os.path.exists(os.path.join(save_root, video_name)):
            os.mkdir(os.path.join(save_root, video_name))
        save_path = os.path.join(save_root, video_name, img_name)

        img = cv2.imread(img_path)
        frame_boxes = pred_bboxes[i]  # [box_num, 4]

        for b in range(frame_boxes.shape[0]):
            bbox = frame_boxes[b]
            if (bbox < 1).all():
                continue

            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

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
