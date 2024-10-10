## Train
> DWD dataset
  <python tools/train.py configs configs/OA-DG/dwd/faster_rcnn_r101_dc5_1x_dwd.py --work-dir /home/intern/minkyoung/dataset/DWD/faster_rcnn_r101_dc5_1X_dwd_oadg --gpu-ids 3>
> Cityscpaes dataset
 <python -u tools/train.py configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py --work-dir /home/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/exp2>
---
## Test
<python -u tools/analysis_tools/test_robustness.py configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py /home/intern/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/epoch_1.pth --out /home/intern/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/test_robustness_result_1epoch.pkl --corruptions benchmark --eval bbox>
