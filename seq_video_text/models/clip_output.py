import os
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def random_color():
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)

    return (b, g, r)


class Clips(object):
    def __init__(self, clip_frame_start, clip_length, query_scores, pred_boxes, pred_masks, pred_objs, pred_recs=None):
        self.frame_idx = [clip_frame_start + i for i in range(clip_length)]
        self.frame_set = set(self.frame_idx)
        self.clip_length = clip_length

        self.scores = query_scores   # [box_num, 1]
        self.boxes = pred_boxes      # [fn, box_num, 4]
        self.recs = pred_recs        # [fn, box_num, word_length]
        self.obj_scores = pred_objs        # [fn, box_num]
        self.masks = pred_masks
        self.masks_probs = pred_masks.sigmoid()
        self.num_instance = pred_boxes.shape[1]

        assert self.scores.shape[0] == self.boxes.shape[1] == self.masks.shape[0], "the box number is not equal"

        # self.classes = results.pred_classes
        # self.scores = results.scores
        # self.cls_probs = results.cls_probs
        # self.mask_logits = results.pred_masks
        # self.mask_probs = results.pred_masks.sigmoid()
        #
        # self.num_instance = len(self.scores)


class Videos:
    def __init__(self, folder, file_names, video_length, image_hw, overlap_frame_num, draw_root=None):
        self.folder = folder
        self.file_names = file_names
        self.video_length = video_length
        self.image_hw = image_hw
        self.overlap_frame_num = overlap_frame_num
        self.draw_root = draw_root
        self.match_threshold = 0.01
        self.video_results = [({}, [{}]) for _ in range(video_length)]

        self.last_overlap_masks = None
        self.last_box_ids = []
        self.box_id = 0
        self.id_to_color = {}

    def write_results(self,
                      clip_frame_idxs: list,
                      box_ids: list,
                      obj_scores: np.ndarray,
                      pred_boxes: np.ndarray,
                      pred_masks: np.ndarray):
        """
        :param obj_scores: object score [fn, box_num]
        :param clip_frame_idxs:
        :param pred_boxes:     [fn, box_num, 4]
        :param pred_masks:     [box_num, fn, h, w]
        :param box_ids:        [box_num]
        :return:
        """
        pred_boxes = pred_boxes.astype(obj_scores.dtype)
        for b, b_id in enumerate(box_ids):
            for f, frame_idx in enumerate(clip_frame_idxs):
                if obj_scores[f][b] < 0.01:
                    continue
                if pred_masks[b][f].max() == 0 or pred_boxes[f][b].sum() < 1:  # 没有box
                    continue
                if b_id in self.video_results[frame_idx][0]:  # 已经保存了 id box
                    continue

                rotate_box, rbox_num = binary_mask_to_quad(pred_masks[b][f])
                if rbox_num == 0:  # box太小，都被filter了
                    continue
                mask_dict = {'size': list(self.image_hw),
                             'counts': " ".join('%s' % coor for coor in rotate_box.flatten())}
                self.video_results[frame_idx][1][0][b_id] = mask_dict

                box_dict = {'bbox': np.concatenate([pred_boxes[f][b], obj_scores[f, b, np.newaxis]]), 'label': 0}
                self.video_results[frame_idx][0][b_id] = box_dict

    def show_boxes(self, input_clip: Clips, mode="mask", box_color=False):
        assert mode in [None, "det", "mask"], "mode error"
        if input_clip.frame_idx[0] == 0:  # video start
            draw_img_idxs = input_clip.frame_idx
        else:
            # draw_img_idxs = input_clip.frame_idx[self.overlap_frame_num:]
            draw_img_idxs = input_clip.frame_idx    # revised.zjb

        for img_idx in draw_img_idxs:
            img_path = os.path.join(self.folder, self.file_names[img_idx])
            video_name, img_name = self.file_names[img_idx].split("/")
            if not os.path.exists(os.path.join(self.draw_root, video_name)):
                os.mkdir(os.path.join(self.draw_root, video_name))
            save_path = os.path.join(self.draw_root, video_name, img_name)

            img = cv.imread(img_path)
            img_results = self.video_results[img_idx]
            if mode == "det":
                for b_id, bbox_ in img_results[0].items():
                    if b_id not in self.id_to_color:
                        self.id_to_color[b_id] = random_color()
                    color = self.id_to_color[b_id] if box_color else (0, 0, 255)

                    bbox = bbox_['bbox'].astype(np.int32)
                    img = cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=1)
                    img = cv.putText(img, str(b_id), (bbox[0], bbox[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                     (255, 0, 0), 1, cv.LINE_AA)

            else:               # quad boxes based masks
                for b_id, quad_ in img_results[1][0].items():
                    if b_id not in self.id_to_color:
                        self.id_to_color[b_id] = random_color()
                    color = self.id_to_color[b_id] if box_color else (0, 0, 255)

                    Points = quad_['counts'].split(" ")
                    points = []
                    if len(Points) != 8:
                        t = len(Points) // 8
                        for i in range(t):
                            if not (np.array(Points[i * 8: (i + 1) * 8]) == '0').all():
                                points = Points[i * 8: (i + 1) * 8]
                    else:
                        points = Points
                    points = list(map(int, points))
                    box = np.array(points).reshape(-1, 2).astype(np.int32)
                    img = cv.polylines(img, [box], isClosed=True, color=color, thickness=1,
                                       lineType=cv.LINE_AA)
                    img = cv.putText(img, str(b_id), (points[6], points[7]), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                     (255, 0, 0), 1, cv.LINE_AA)

            cv.imwrite(save_path, img)
            print(save_path)

    def update(self, input_clip: Clips, result_mode=None, box_color=False):
        """
        :param result_mode: 画图
        :param input_clip:
               input_clip.clip_boxes:  [fn, box_num, 4]
               input_clip.clip_masks:  [box_num, fn, h, w]
        :return:
        """
        if self.box_id == 0:  # video start
            cur_box_ids = np.arange(0, input_clip.num_instance, dtype=np.int32)
            self.box_id += input_clip.num_instance
        else:
            cur_overlap_masks = input_clip.masks_probs[:, :self.overlap_frame_num]  # [N_input, over_fn, h, w]
            scores = get_siou(cur_overlap_masks, self.last_overlap_masks)  # N_saved, N_input

            # bipartite match
            above_thres = scores > self.match_threshold
            scores = scores * above_thres.float()

            row_idx, col_idx = linear_sum_assignment(scores.cpu(), maximize=True)
            cur_box_ids = np.zeros(input_clip.num_instance, dtype=np.int32)

            existed_idx = []
            for is_above, r, c in zip(above_thres[row_idx, col_idx], row_idx, col_idx):
                if not is_above:
                    continue
                cur_box_ids[c] = self.last_box_ids[r]
                existed_idx.append(c)

            no_match_idxs = [i for i in range(input_clip.num_instance) if i not in existed_idx]  # 没有匹配上的
            for idx in no_match_idxs:
                cur_box_ids[idx] = self.box_id
                self.box_id += 1

        self.last_box_ids = cur_box_ids
        self.last_overlap_masks = input_clip.masks_probs[:, input_clip.clip_length - self.overlap_frame_num:]

        # 插值 mask  TODO: 存在冗余计算
        pred_masks = F.interpolate(input_clip.masks, self.image_hw, mode="bilinear")
        pred_masks = pred_masks.sigmoid().cpu().detach().numpy() > 0.5

        # # zjb.add !!!!!!!!!!!!!!!!!!!!!
        # temp_masks = pred_masks * 255   # [box_num, fn, h, w]
        # for i in range(temp_masks.shape[0]):
        #     for j in range(temp_masks.shape[1]):
        #         zzzz = temp_masks[i][j]
        #         cv.imwrite(f"/lustre/home/jbzhang/SeqFormer_rec/exp_yvt_finetune_from_nlpr/temp_show/b{i}_f{j}.jpg", zzzz)
        #
        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # 从mask解出四边形box，写入result
        self.write_results(input_clip.frame_idx, cur_box_ids.tolist(),
                           input_clip.obj_scores.numpy(), input_clip.boxes.numpy(), pred_masks)

        if result_mode is not None and self.draw_root is not None:
            self.show_boxes(input_clip, result_mode, box_color)

    def get_video_results(self):
        return self.video_results


def get_siou(input_masks, saved_masks):
    # input_masks : N_i, T, H, W
    # saved_masks : N_s, T, H, W
    # import pdb; pdb.set_trace()
    input_masks = input_masks.flatten(-2)  # N_i, T, HW
    saved_masks = saved_masks.flatten(-2)  # N_s, T, HW

    input_masks = input_masks[None]  # 1  , N_i, T, HW
    saved_masks = saved_masks.unsqueeze(1)  # N_s, 1  , T, HW

    # N_s, N_i, T, HW
    numerator = saved_masks * input_masks
    denominator = saved_masks + input_masks - saved_masks * input_masks

    numerator = numerator.sum(dim=(-1, -2))
    denominator = denominator.sum(dim=(-1, -2))

    siou = numerator / (denominator + 1e-6)  # N_s, N_i

    return siou


def get_mini_boxes(contour):
    bounding_box = cv.minAreaRect(contour)
    points = sorted(list(cv.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def binary_mask_to_quad(_bitmap):
    """
    :param _bitmap:  [h, w]
    :return:
    """
    assert len(_bitmap.shape) == 2
    bitmap = _bitmap  # .cpu().numpy()  # The first channel
    # pred = pred.cpu().detach().numpy()
    height, width = bitmap.shape
    contours, _ = cv.findContours((bitmap * 255).astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    num_contours = min(len(contours), 100)  # self.max_candidates)
    boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
    # scores = np.zeros((num_contours,), dtype=np.float32)
    act_box_num = 0

    for index in range(num_contours):
        contour = contours[index].squeeze(1)
        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue
        points = np.array(points)
        # import pdb; pdb.set_trace()
        # box = unclip(points, unclip_ratio=1.0).reshape(-1, 1, 2)
        box = points
        box, sside = get_mini_boxes(box)
        if sside < 5:
            continue
        box = np.array(box)

        box[:, 0] = np.clip(box[:, 0], 0, width)
        box[:, 1] = np.clip(box[:, 1], 0, height)
        boxes[index, :, :] = box.astype(np.int16)
        act_box_num += 1

    return boxes, act_box_num