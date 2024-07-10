from fastapi import FastAPI, UploadFile, File

from PIL import Image
import torch
import io
import numpy as np
from hair_swap import HairFast, get_parser

app = FastAPI()

hair_fast = HairFast(get_parser().parse_args([]))

@app.post("/hair_transfer/")
async def hair_transfer(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
    face_img = Image.open(io.BytesIO(await face.read()))
    shape_img = Image.open(io.BytesIO(await shape.read()))
    color_img = Image.open(io.BytesIO(await color.read()))

    result_tensor = hair_fast(face_img, shape_img, color_img)
    
    # 텐서의 형식을 확인하고 PIL 이미지로 변환
    result_tensor = result_tensor.squeeze().cpu().detach().numpy()
    if result_tensor.ndim == 3 and result_tensor.shape[0] in [1, 3]:
        result_tensor = np.transpose(result_tensor, (1, 2, 0))
    
    result_img = Image.fromarray((result_tensor * 255).astype(np.uint8))
    result_img.save("output/result14.png")
    
    return {"message": "Hair transfer completed", "result_path": "output/result.png"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
