indir="/nfs/home/zhan/code/mmpose"
cd $indir

python demo/topdown_demo_with_mmdet.py \
       demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
       https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py \
       https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth \
       --input data/test/record.png \
       --output-root outputs/B1_RTM_1 \
       --device cuda:0 \
       --bbox-thr 0.5 \
       --kpt-thr 0.5 \
       --nms-thr 0.3 \
       --radius 8 \
       --thickness 4 \
       --draw-bbox \
       --draw-heatmap \
       --show-kpt-idx

