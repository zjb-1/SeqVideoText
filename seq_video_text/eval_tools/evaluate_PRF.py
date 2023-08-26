import time
import subprocess
import os
import numpy as np
import shutil
import json


def eval_PRF(submit_path, gt_zip_path):
    print("P R F eval ing....")
    ori_path = os.getcwd()
    start_time = time.time()

    submit_name = submit_path.split('/')[-1]

    os.chdir(submit_path)  # 进入submit路径
    # res = subprocess.getoutput(f'zip -q {submit_name}.zip *.txt')
    res = subprocess.getoutput(f'ls|xargs -n 10000 zip -q {submit_name}.zip')  # 针对文件太多
    res = subprocess.getoutput(f'mv {submit_name}.zip ../')
    os.chdir('../')

    # res = subprocess.getoutput('python ./script.py –g=./evaluate/gt.zip –s=./submit.zip')
    res = subprocess.getoutput(f'python /lustre/home/jbzhang/experiment/Swin-Track/tools/script.py –g={gt_zip_path} –s=./{submit_name}.zip')
    print(res)
    os.remove(f'./{submit_name}.zip')

    res = json.loads(res[res.index("{"):])
    P, R, F1 = res['precision'], res['recall'], res['hmean']
    return P, R, F1
    #print('eval time is {}'.format(time.time() - start_time))


def eval_PRF_sep(submit_root, gt_zip_root):
    p_list = []
    r_list = []
    f_list = []
    for i in range(1, 6):
        print(f"-----------eval video name:{i}-------------")
        submit_path = os.path.join(submit_root, f"submit_{i}")
        gt_zip_path = os.path.join(gt_zip_root, f"gt_{i}.zip")

        p, r, f = eval_PRF(submit_path, gt_zip_path)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        print("--------------------------------------------")

    p_avg = np.array(p_list).mean()
    r_avg = np.array(r_list).mean()
    f_avg = np.array(f_list).mean()
    print(f"***avg all videos: precision:{p_avg:.6f} , recall:{r_avg:.6f}, hmean:{f_avg:.6f}***\n")