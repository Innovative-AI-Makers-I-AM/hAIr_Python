# image_utils.py
import cv2
import numpy as np
from database import get_session, Image, ImageMetadata

session = get_session()

# 이미지와 메타데이터를 같이 저장하기
def save_image_metadata_to_db(file_path, metadata):
    # 세션 가져오기 각 함수마다 세션을 생성해야지 세션 관리가 쉬워집니다.
    session = get_session()
    try:
        with open(file_path, 'rb') as file:
            # file_path에 있는 파일을 읽어와 binary_data로 변환
            binary_data = file.read()
            # binary_data를 Image 클래스에 넣기
            new_image = Image(data=binary_data)
            # new_image를 세션에 넣기
            session.add(new_image)
            # 세션 작업 완료
            session.commit()

            # 메타데이터 저장
            new_meta = ImageMetadata(
                # new_image의 아이디를 image id로 저장
                image_id=new_image.id,
                sex=metadata["sex"],
                length=metadata["length"],
                style=metadata["style"],
                designer=metadata["designer"],
                shop_name=metadata["shop_name"],
                hashtag1=metadata["hashtag1"],
                hashtag2=metadata["hashtag2"],
                hashtag3=metadata["hashtag3"]
            )
            session.add(new_meta)
            session.commit()

            print(f"Image and metadata saved with Image ID: {new_image.id}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        session.rollback()
        print(f"Error saving image to DB: {e}")
    finally:
        session.close()

# 파일 경로를 매개변수로 받아 이미지 파일 저장하기
def save_image_to_db(file_path):
    # 세션 가져오기 각 함수마다 세션을 생성해야지 세션 관리가 쉬워집니다.
    session = get_session()
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
    # 세션 가져오기 각 함수마다 세션을 생성해야지 세션 관리가 쉬워집니다.
    session = get_session()
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
    # 세션 가져오기 각 함수마다 세션을 생성해야지 세션 관리가 쉬워집니다.
    session = get_session()
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
    # 세션 가져오기 각 함수마다 세션을 생성해야지 세션 관리가 쉬워집니다.
    session = get_session()
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
    # 세션 가져오기 각 함수마다 세션을 생성해야지 세션 관리가 쉬워집니다.
    session = get_session()
    try:
        images = session.query(Image).all()
        for image in images:
            print(f"Image ID: {image.id}, Data length: {len(image.data)}")
    except Exception as e:
        print(f"Error listing images: {e}")