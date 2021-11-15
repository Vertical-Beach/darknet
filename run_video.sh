#!/bin/bash
for n in {0..73}; do
    video_id=`printf %02d $n`
    echo $video_id
    python3 darknet_tracking.py \
        --input ../dataset/test_videos/test_${video_id}.mp4 \
        --out_filename out_test_${video_id}.mp4 \
        --config_file ./yolov4-tiny-custom.cfg \
        --data_file ./cfg/signate.data \
        --weights ./yolov4-tiny-custom_best.weights
done
