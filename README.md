환경 구성 명령어

conda create -n iam_proj python=3.10 

conda activate iam_proj

pip install -r requirements.txt 

pip install "uvicorn[standard]" 

uvicorn main:app --reload

npm install

npm start