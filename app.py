# HiggingFace API를 FastAPI에 연동
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tempfile
import os
from gradio_client import Client, handle_file

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용, 필요에 따라 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("AIRI-Institute/HairFastGAN")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the HairFastGAN API. Use the /hair_transfer endpoint to upload images."}

@app.post("/hair_transfer")
async def hair_transfer(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
    # Read the uploaded images into memory
    face_image = Image.open(io.BytesIO(await face.read()))
    shape_image = Image.open(io.BytesIO(await shape.read()))
    color_image = Image.open(io.BytesIO(await color.read()))

    # Save the images to temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as face_tempfile:
        face_image.save(face_tempfile, format="JPEG")
        face_tempfile_path = face_tempfile.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as shape_tempfile:
        shape_image.save(shape_tempfile, format="PNG")
        shape_tempfile_path = shape_tempfile.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as color_tempfile:
        color_image.save(color_tempfile, format="PNG")
        color_tempfile_path = color_tempfile.name

    result_image_path = None  # 초기화

    try:
        # Step 1: Resize the face image
        resized_face = client.predict(
            img=handle_file(face_tempfile_path),
            align=["Face"],
            api_name="/resize_inner"
        )

        # Step 2: Resize the hairstyle image
        resized_shape = client.predict(
            img=handle_file(shape_tempfile_path),
            align=["Shape"],
            api_name="/resize_inner_1"
        )

        # Step 3: Resize the hair color image
        resized_color = client.predict(
            img=handle_file(color_tempfile_path),
            align=["Color"],
            api_name="/resize_inner_2"
        )

        # Extracting the resized image paths
        resized_face_path = resized_face if isinstance(resized_face, str) else resized_face[0]
        resized_shape_path = resized_shape if isinstance(resized_shape, str) else resized_shape[0]
        resized_color_path = resized_color if isinstance(resized_color, str) else resized_color[0]

        # Read the resized images into memory
        resized_face_image = Image.open(resized_face_path)
        resized_shape_image = Image.open(resized_shape_path)
        resized_color_image = Image.open(resized_color_path)

        resized_face_bytes = io.BytesIO()
        resized_shape_bytes = io.BytesIO()
        resized_color_bytes = io.BytesIO()

        resized_face_image.save(resized_face_bytes, format="PNG")
        resized_shape_image.save(resized_shape_bytes, format="PNG")
        resized_color_image.save(resized_color_bytes, format="PNG")

        resized_face_bytes.seek(0)
        resized_shape_bytes.seek(0)
        resized_color_bytes.seek(0)

        # Step 4: Swap hair using resized images
        result = client.predict(
            face=handle_file(resized_face_path),
            shape=handle_file(resized_shape_path),
            color=handle_file(resized_color_path),
            blending="Article",
            poisson_iters=0,
            poisson_erosion=15,
            api_name="/swap_hair"
        )

        # Extract the result image path
        result_image_path = result[0]['value']

        # Read the result image
        result_image = Image.open(result_image_path)
        result_image_bytes = io.BytesIO()
        result_image.save(result_image_bytes, format="PNG")
        result_image_bytes.seek(0)

        return StreamingResponse(result_image_bytes, media_type="image/png")

    finally:
        # Clean up temporary files
        os.remove(face_tempfile_path)
        os.remove(shape_tempfile_path)
        os.remove(color_tempfile_path)
        if resized_face_path and os.path.exists(resized_face_path):
            os.remove(resized_face_path)
        if resized_shape_path and os.path.exists(resized_shape_path):
            os.remove(resized_shape_path)
        if resized_color_path and os.path.exists(resized_color_path):
            os.remove(resized_color_path)
        if result_image_path and os.path.exists(result_image_path):
            os.remove(result_image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
