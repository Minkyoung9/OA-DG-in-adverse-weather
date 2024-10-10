#Object Detection in adverse weather

## OA-DG Method Summary
**Object-Aware Domain Generalization**
Type: Single Domain\
Model : Faster R-CNN (2-stage Detector)\
Method : Image augmentation, Domain Generalization\
Dataset : DWD, Cityscapes

### Train
**DWD dataset (Diverse Weather Dataset)**
 
    python tools/train.py configs configs/OA-DG/dwd/faster_rcnn_r101_dc5_1x_dwd.py --work-dir /home/intern/minkyoung/dataset/DWD/faster_rcnn_r101_dc5_1X_dwd_oadg --gpu-ids 3


**Cityscapes dataset**
Used classes -> ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle','bicycle')

    python -u tools/train.py configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py --work-dir /home/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/exp2
---
### Eval
**cityscapes dataset**

    python -u tools/analysis_tools/test_robustness.py configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py /home/intern/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/epoch_1.pth --out /home/intern/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/test_robustness_result_1epoch.pkl --corruptions benchmark --eval bbox
---
## Reference
[GitHub - WoojuLee24/OA-DG: Object-Aware Domain Generalization for Object Detection](https://github.com/WoojuLee24/OA-DG?tab=readme-ov-file), 
[OA-DG Paper](https://arxiv.org/pdf/2312.12133v1)
