from gradio_client import Client, handle_file
from PIL import Image
import shutil

client = Client("AIRI-Institute/HairFastGAN")

# Function to save the temporary image to local
def save_temp_image(temp_path, local_path):
    shutil.copy(temp_path, local_path)
    return local_path

# Step 1: Resize the face image
resized_face = client.predict(
    img=handle_file('10.jpg'),  # 얼굴 이미지 파일 경로
    align=["Face", "Shape", "Color"],
    api_name="/resize_inner"
)
print("Resized face path:", resized_face)

# Step 2: Resize the hairstyle image
resized_shape = client.predict(
    img=handle_file('7.png'),  # 헤어스타일 이미지 파일 경로
    align=["Face", "Shape", "Color"],
    api_name="/resize_inner_1"
)
print("Resized shape path:", resized_shape)

# Step 3: Resize the hair color image
resized_color = client.predict(
    img=handle_file('2.png'),  # 헤어 색상 이미지 파일 경로
    align=["Face", "Shape", "Color"],
    api_name="/resize_inner_2"
)
print("Resized color path:", resized_color)

# Extracting the resized image paths
resized_face_path = resized_face if isinstance(resized_face, str) else resized_face[0]
resized_shape_path = resized_shape if isinstance(resized_shape, str) else resized_shape[0]
resized_color_path = resized_color if isinstance(resized_color, str) else resized_color[0]

# Verify if the paths are correct before copying
print("Verified face path:", resized_face_path)
print("Verified shape path:", resized_shape_path)
print("Verified color path:", resized_color_path)

# Save the temporary images to local
resized_face_path = save_temp_image(resized_face_path, 'resized_face.png')
resized_shape_path = save_temp_image(resized_shape_path, 'resized_shape.png')
resized_color_path = save_temp_image(resized_color_path, 'resized_color.png')

# Step 4: Swap hair using resized images
result = client.predict(
    face=handle_file(resized_face_path),  # 리사이즈된 얼굴 이미지 파일 경로
    shape=handle_file(resized_shape_path),  # 리사이즈된 헤어스타일 이미지 파일 경로
    color=handle_file(resized_color_path),  # 리사이즈된 헤어 색상 이미지 파일 경로
    blending="Article",
    poisson_iters=0,
    poisson_erosion=15,
    api_name="/swap_hair"
)
print(result)

# Extract the image path from the result
result_image_path = result[0]['value']

# Display the result image
result_image = Image.open(result_image_path)
result_image.show()