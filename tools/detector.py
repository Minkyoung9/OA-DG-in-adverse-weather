import os
import torch
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import matplotlib.pyplot as plt  # 추가

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_folder', help='Folder containing images')  # 폴더 경로로 변경
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:4', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

def process_images(img_folder, model, score_thr=0.3, device='cuda:0'):
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        result = inference_detector(model, img_path)  # 이미지 처리
        
        # 결과를 시각화하고 out_file로 저장
        #out_file = os.path.join(img_folder,'results', f"output_{img_file}")  # output 경로 설정
        show_result_pyplot(
            model, 
            img_path, 
            result, 
            score_thr=score_thr, 
            out_file=None  # out_file을 전달
        )

async def async_process_images(img_folder, model, score_thr=0.3, device='cuda:0'):
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    tasks = []
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        tasks.append(inference_detector(model, img_path))  # 비동기 작업 생성

    results = await asyncio.gather(*tasks)  # 비동기 결과 처리
    for idx, result in enumerate(results):
        # 결과를 시각화하고 out_file로 저장
        result_image = model.show_result(
            img_files[idx], 
            result, 
            score_thr=score_thr, 
            show=False  # show=False로 설정하여 이미지만 반환
        )
        
        # matplotlib을 사용하여 결과 이미지 저장
        plt.imshow(result_image)
        plt.axis('off')
        plt.savefig(os.path.join(img_folder, f"output_{img_files[idx]}"))
        plt.close()

def main(args):
    # 모델 로딩
    with torch.no_grad():
        model = init_detector(args.config, args.checkpoint, device=args.device)

    # 이미지 처리
    process_images(args.img_folder, model, score_thr=args.score_thr, device=args.device)

async def async_main(args):
    # 모델 로딩
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 비동기 이미지 처리
    await async_process_images(args.img_folder, model, score_thr=args.score_thr, device=args.device)

if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
