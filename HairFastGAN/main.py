import argparse
import os
import sys
from pathlib import Path

from torchvision.utils import save_image
from tqdm.auto import tqdm

from hair_swap import HairFast, get_parser
import torch
from PIL import Image
import numpy as np

# 모델 초기화 및 로드 함수 정의
def load_model(model_args):
    print("Loading the model...")
    model = HairFast(model_args)
    return model

# 이미지를 불러와서 전처리하는 함수 정의
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image.unsqueeze(0)  # 배치 차원 추가

# 모델을 사용하여 이미지 처리 함수 정의
def process_images(model, face_img_path, shape_img_path, color_img_path, benchmark=False, exp_name=None):
    face_img = preprocess_image(face_img_path)
    shape_img = preprocess_image(shape_img_path)
    color_img = preprocess_image(color_img_path)
    
    with torch.no_grad():  # 평가 모드
        result = model.swap(face_img, shape_img, color_img, benchmark=benchmark, exp_name=exp_name)
    return result

def main(model_args, args):
    # 모델 로드
    model = load_model(model_args)

    experiments = []
    if args.file_path is not None:
        with open(args.file_path, 'r') as file:
            experiments.extend(file.readlines())

    if all(path is not None for path in (args.face_path, args.shape_path, args.color_path)):
        experiments.append((args.face_path, args.shape_path, args.color_path))

    for exp in tqdm(experiments):
        if isinstance(exp, str):
            file_1, file_2, file_3 = exp.strip().split()
        else:
            file_1, file_2, file_3 = exp

        face_path = args.input_dir / file_1
        shape_path = args.input_dir / file_2
        color_path = args.input_dir / file_3

        base_name = '_'.join([path.stem for path in (face_path, shape_path, color_path)])
        exp_name = base_name if model_args.save_all else None

        if isinstance(exp, str) or args.result_path is None:
            os.makedirs(args.output_dir, exist_ok=True)
            output_image_path = args.output_dir / f'{base_name}.png'
        else:
            os.makedirs(args.result_path.parent, exist_ok=True)
            output_image_path = args.result_path

        final_image = process_images(model, face_path, shape_path, color_path, benchmark=args.benchmark, exp_name=exp_name)
        save_image(final_image.squeeze(), output_image_path)  # .squeeze()를 추가하여 차원을 맞춤

if __name__ == "__main__":
    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')
    parser.add_argument('--input_dir', type=Path, default='', help='The directory of the images to be inverted')
    parser.add_argument('--benchmark', action='store_true', help='Calculates the speed of the method during the session')

    # Arguments for a set of experiments
    parser.add_argument('--file_path', type=Path, default=None,
                        help='File with experiments with the format "face_path.png shape_path.png color_path.png"')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')

    # Arguments for single experiment
    parser.add_argument('--face_path', type=Path, default=None, help='Path to the face image')
    parser.add_argument('--shape_path', type=Path, default=None, help='Path to the shape image')
    parser.add_argument('--color_path', type=Path, default=None, help='Path to the color image')
    parser.add_argument('--result_path', type=Path, default=None, help='Path to save the result')

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_)
        model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)

    main(model_args, args)
