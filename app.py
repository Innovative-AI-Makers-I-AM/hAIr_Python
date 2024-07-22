# 로컬 test를 위한 FastAPI
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import StreamingResponse
# from PIL import Image
# import torch
# import io
# import numpy as np
# from hair_swap import HairFast, get_parser

# app = FastAPI()
# hair_fast = HairFast(get_parser().parse_args([]))

# @app.post("/hair_transfer/")
# async def hair_transfer(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
#     face_img = Image.open(io.BytesIO(await face.read()))
#     print(face_img)
#     shape_img = Image.open(io.BytesIO(await shape.read()))
#     color_img = Image.open(io.BytesIO(await color.read()))
#     result_tensor = hair_fast(face_img, shape_img, color_img)

#     # 텐서의 형식을 확인하고 PIL 이미지로 변환
#     result_tensor = result_tensor.squeeze().cpu().detach().numpy()

#     if result_tensor.ndim == 3 and result_tensor.shape[0] in [1, 3]:
#         result_tensor = np.transpose(result_tensor, (1, 2, 0))
#     result_img = Image.fromarray((result_tensor * 255).astype(np.uint8))

#     # 이미지를 메모리에 저장
#     img_byte_arr = io.BytesIO()
#     result_img.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)

#     return StreamingResponse(img_byte_arr, media_type="image/png")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# React 연결을 위한 FastAPI

# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import StreamingResponse
# from PIL import Image
# import io
# import numpy as np
# from hair_swap import HairFast, get_parser

# app = FastAPI()

# hair_fast = HairFast(get_parser().parse_args([]))

# @app.post("/hair_transfer")
# async def hair_transfer(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
#     face_img = Image.open(io.BytesIO(await face.read()))
#     shape_img = Image.open(io.BytesIO(await shape.read()))
#     color_img = Image.open(io.BytesIO(await color.read()))

#     result_tensor = hair_fast(face_img, shape_img, color_img)

#     result_tensor = result_tensor.squeeze().cpu().detach().numpy()
#     if result_tensor.ndim == 3 and result_tensor.shape[0] in [1, 3]:
#         result_tensor = np.transpose(result_tensor, (1, 2, 0))

#     result_img = Image.fromarray((result_tensor * 255).astype(np.uint8))

#     img_byte_arr = io.BytesIO()
#     result_img.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)

#     return StreamingResponse(img_byte_arr, media_type="image/png")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from hair_swap import HairFast, get_parser

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용, 필요에 따라 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hair_fast = HairFast(get_parser().parse_args([]))

@app.post("/hair_transfer/")
async def hair_transfer(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
    face_img = Image.open(io.BytesIO(await face.read()))
    shape_img = Image.open(io.BytesIO(await shape.read()))
    color_img = Image.open(io.BytesIO(await color.read()))
    result_tensor = hair_fast(face_img, shape_img, color_img)
    result_tensor = result_tensor.squeeze().cpu().detach().numpy()

    if result_tensor.ndim == 3 and result_tensor.shape[0] in [1, 3]:
        result_tensor = np.transpose(result_tensor, (1, 2, 0))

    result_img = Image.fromarray((result_tensor * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)