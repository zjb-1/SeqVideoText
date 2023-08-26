CUDA_VISIBLE_DEVICES=4 python ../seq_video_text/inference_text_merge_test.py \
    --backbone resnet50 \
    --obj \
    --masks \
    --ytvis_path /lustre/datasharing/jbzhang/data/YVT/ \
    --test_img_prefix test_images \
    --test_ann_file video_test.json \
    --gt_zip_path /lustre/datasharing/jbzhang/data/YVT/yvt_test_gt.zip \
    --eval_mota_pred_json pred_eval.json \
    --eval_mota_gt_json /lustre/datasharing/jbzhang/data/YVT/mota_test_gt.json \
    --show_mode mask \
    --out output_e50c04.pkl \
    --clip_length 5 \
    --clip_stride 2 \
    --confidence_threshold 0.4 \
    --model_path ./r2/checkpoint0049.pth \
    # --show_path ./show/ \

