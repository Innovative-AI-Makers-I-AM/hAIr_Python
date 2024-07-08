# 환경 구성 명령어

conda create -n iam_proj python=3.9

conda activate iam_proj

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install mkl

pip install -r requirements.txt 
