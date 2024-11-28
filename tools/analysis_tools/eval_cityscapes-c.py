import pickle
import csv
import numpy as np

# 경로 설정 및 corruptions, severity 리스트
pkl_file_path = "/home/intern/minkyoung/OA-DG/work_dir/faster_rcnn_r50_fpn_1x_cityscapes_oadg/results/test_robustness_epoch2_results.pkl"
csv_file_path = "/home/intern/minkyoung/dataset/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg2/output_result_file/clear_cityscapes-c_oadg_results.csv"
corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
severity = [0, 1, 2, 3, 4, 5]

# pkl 파일 로드
with open(pkl_file_path, 'rb') as f:
    results = pickle.load(f)
    print(results)

# mPC 계산 및 csv 파일 저장
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 헤더 작성
    writer.writerow(['Corruption Type', 'mAP'])

    # 모든 corruption 타입에 대한 mPC 값을 저장할 리스트
    map_values = []

    for corruption in corruptions:
        if corruption == 'gaussian_noise':
            # Clean AP 값을 gaussian_noise의 severity 0 값으로 설정
            clean_ap = results[corruption][0]['bbox']['AP']
            writer.writerow(['clean', f"{clean_ap:.3f}"])
        
        # 각 corruption 타입별 AP 값 계산 (severity 1~5 평균)
        aps = []
        for sev in severity[1:]:  # severity 0 제외
            aps.append(results[corruption][sev]['bbox']['AP'])
        writer.writerow([f"{ap:.3f}" for ap in aps])
        mAP = np.mean(aps)  # mAP 계산
        map_values.append(mAP)  # mAP 값을 리스트에 저장

        # csv 파일에 작성
        writer.writerow([corruption, f"{mAP:.3f}"])

    # 총 mPC 계산 (모든 corruption에 대한 mPC 평균)
    mPC = np.mean(map_values)
    writer.writerow(['mPC', f"{mPC:.3f}"])

print(f"Results saved to {csv_file_path}")
