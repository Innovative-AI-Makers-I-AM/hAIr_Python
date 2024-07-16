# image_utils.py
import cv2
import numpy as np
from database import get_session, Image

session = get_session()

# 파일 경로를 매개변수로 받아 이미지 파일 저장하기
def save_image_to_db(file_path):
    try:
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            new_image = Image(data=binary_data)
            session.add(new_image)
            session.commit()
            print(f"Image saved with ID: {new_image.id}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error saving image to DB: {e}")
    finally:
        session.close()

# 모든 이미지 삭제 
def delete_all_images():
    try:
        images = session.query(Image).all()
        if images:
            for image in images:
                session.delete(image)
            session.commit()
            print(f"{len(images)} images deleted successfully.")
        else:
            print("No images found to delete.")
    except Exception as e:
        session.rollback()
        print(f"Error deleting images from DB: {e}")
    finally:
        session.close()

# id의 범위 값으로 이미지 지우기
def delete_images_by_id_range(start_id, end_id):
    try:
        images = session.query(Image).filter(Image.id >= start_id, Image.id <= end_id).all()
        if images:
            for image in images:
                session.delete(image)
            session.commit()
            print(f"{len(images)} images deleted successfully.")
        else:
            print("No images found to delete in the specified range.")
    except Exception as e:
        session.rollback()
        print(f"Error deleting images from DB: {e}")
    finally:
        session.close()

# id 값을 기능으로 이미지 가져오기
def show_image(image_id):
    try:
        # 이미지 데이터베이스에서 읽기
        image = session.query(Image).filter(Image.id == image_id).first()
        if image is not None:
            # 이미지 데이터를 numpy 배열로 변환
            nparr = np.frombuffer(image.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 이미지 보여주기
            cv2.imshow(f'Image ID: {image_id}', img)
            cv2.waitKey(0)  # 키 입력 대기
            cv2.destroyAllWindows()  # 모든 창 닫기
        else:
            print(f"No image found with ID {image_id}")
    except Exception as e:
        print(f"Error retrieving image from DB: {e}")

# 데이터베이스에 저장된 이미지 확인
def list_images():
    try:
        images = session.query(Image).all()
        for image in images:
            print(f"Image ID: {image.id}, Data length: {len(image.data)}")
    except Exception as e:
        print(f"Error listing images: {e}")