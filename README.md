# Object Detection in adverse weather
📢 This project is based on the following GitHub:
[GitHub - WoojuLee24/OA-DG](https://github.com/WoojuLee24/OA-DG?tab=readme-ov-file)

## OA-DG Method Summary
**Object-Aware Domain Generalization**\
Type: Single Domain\
Model : Faster R-CNN (2-stage Detector)\
Method : Image augmentation, Domain Generalization\
Dataset : DWD, Cityscapes
  - [Cityscapes](https://www.cityscapes-dataset.com/): A dataset that contains urban street scenes from 50 cities with detailed annotations.
   - [Diverse Weather Dataset](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B): This dataset includes various weather conditions for robust testing and development of models, essential for applications in autonomous driving.
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
[GitHub - AmingWu/Single-DGOD](https://github.com/AmingWu/Single-DGOD)\
[OA-DG Paper](https://arxiv.org/pdf/2312.12133v1)
