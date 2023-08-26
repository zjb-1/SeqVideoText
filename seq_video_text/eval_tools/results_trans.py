import numpy as np
import pickle
import json
import os
import shutil


def results2json_videoseg(dataset, results, out_file):
    json_results = []
    vid_objs = {}
    for idx in range(len(dataset)):
        # assume results is ordered
        #if idx > 400:
        #   break
        vid_id, frame_id = dataset.img_ids[idx]
        if idx == len(dataset) - 1:
            is_last = True
        else:
            _, frame_id_next = dataset.img_ids[idx+1]
            is_last = frame_id_next == 0
        det, seg = results[idx]
        for obj_id in det:
            bbox = det[obj_id]['bbox']
            segm = seg[0][obj_id]
            label = det[obj_id]['label']
            if obj_id not in vid_objs:
                vid_objs[obj_id] = {'scores':[],'cats':[], 'segms':{}}
            vid_objs[obj_id]['scores'].append(bbox[4])
            vid_objs[obj_id]['cats'].append(label)
            ##segm['counts'] = segm['counts'].decode()
            vid_objs[obj_id]['segms'][frame_id] = segm
        if is_last:
            # store results of  the current video
            for obj_id, obj in vid_objs.items():
                data = dict()

                data['video_id'] = vid_id + 1
                data['score'] = np.array(obj['scores']).mean().item()
                # majority voting for sequence category
                data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
                vid_seg = []
                for fid in range(frame_id + 1):
                    if fid in obj['segms']:
                        vid_seg.append(obj['segms'][fid])
                    else:
                        vid_seg.append(None)
                data['segmentations'] = vid_seg
                json_results.append(data)
            vid_objs = {}
    #import pdb;pdb.set_trace()
    with open(out_file, "w") as f:
        json.dump(json_results, f)


def results2json_mota(dataset, results, out_file, mode="mask"):
    assert mode in ["det", "mask"], "mode error"
    out_res = {}
    for idx in range(len(dataset)):
        vid_id, frame_id = dataset.img_ids[idx]
        img_filename = dataset.vid_infos[vid_id]['filenames'][frame_id]  # "video_name/img_name"
        video_name = img_filename.split('/')[0]
        img_name = img_filename.split('/')[-1].split('.')[0]

        det, seg = results[idx]
        if len(det) == 0:  # 图片没识别出 文本
            continue

        if video_name not in out_res:
            out_res[video_name] = {}
        if mode == "det":
            for obj_id, box_info in det.items():
                if str(obj_id) not in out_res[video_name]:
                    out_res[video_name][str(obj_id)] = {}
                    out_res[video_name][str(obj_id)]["track"] = []

                box = box_info['bbox'].tolist()[:4]
                box = list(map(round, box))

                s = f"{img_name},{box[0]}_{box[1]}_{box[2]}_{box[1]}_{box[2]}_{box[3]}_{box[0]}_{box[3]}"
                out_res[video_name][str(obj_id)]["track"].append(s)

        else:
            segm = seg[0]
            for obj_id, box_seg in segm.items():  # segm:{obj_id: segm}
                if str(obj_id) not in out_res[video_name]:
                    out_res[video_name][str(obj_id)] = {}
                    out_res[video_name][str(obj_id)]["track"] = []

                if box_seg == None:
                    continue

                Points = box_seg['counts'].split(" ")
                points = []
                if len(Points) != 8:
                    t = len(Points) // 8
                    for i in range(t):
                        if not (np.array(Points[i * 8: (i + 1) * 8]) == '0').all():
                            points = Points[i * 8: (i + 1) * 8]
                else:
                    points = Points

                if len(points) == 0:
                    continue

                s = img_name + ','
                s += '_'.join(map(str, points))  # "frame,x1_y1_x2 ..."
                out_res[video_name][str(obj_id)]["track"].append(s)

    with open(out_file, "w") as f:
        json.dump(out_res, f)
    print("inference 自测mota文件 已生成...")


def results2submit(dataset, results, submit_path, mode="mask"):
    assert type(submit_path) == str
    assert mode in ["det", "mask"], "mode error"
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    for idx in range(len(dataset)):
        vid_id, frame_id = dataset.img_ids[idx]
        img_filename = dataset.vid_infos[vid_id]['filenames'][frame_id]  # "video_name/img_name"

        txt_name = 'res_img_' + str(idx+1) + '.txt'
        txt_path = os.path.join(submit_path, txt_name)
        with open(txt_path, 'w') as f:  # 创建/覆盖 为空文件
            f.write('')

        det, seg = results[idx]
        if len(det) == 0:  # 图片没识别出 文本
            continue

        if mode == "det":
            for obj_id, box_info in det.items():
                box = box_info['bbox'].tolist()[:4]
                box = list(map(round, box))

                s = f"{box[0]},{box[1]},{box[2]},{box[1]},{box[2]},{box[3]},{box[0]},{box[3]}\n"
                with open(txt_path, 'a') as f:
                    f.write(s)
        else:
            segm = seg[0]
            for obj_id, box_seg in segm.items():  # segm:{obj_id: segm}
                Points = box_seg['counts'].split(" ")
                points = []
                if len(Points) != 8:
                    t = len(Points) // 8
                    for i in range(t):
                        if not (np.array(Points[i * 8: (i + 1) * 8]) == '0').all():
                            points = Points[i * 8: (i + 1) * 8]
                else:
                    points = Points

                s = ','.join(points) + '\n'
                with open(txt_path, 'a') as f:
                    f.write(s)

    print("submit files 文件生成..")

