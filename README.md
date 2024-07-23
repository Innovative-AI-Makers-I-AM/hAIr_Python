# 환경 구성 명령어
## Conda 환경 생성
conda create -n iam_proj python=3.10
## Conda 환경 활성화
conda activate iam_proj
## PyTorch 및 관련 패키지 설치
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
## MKL 설치
conda install mkl

pip install -r requirements.txt 
