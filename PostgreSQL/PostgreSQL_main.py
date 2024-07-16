from database import init_db
from images_utils import save_image_to_db, delete_all_images, delete_images_by_id_range, show_image, list_images

# 데이터베이스 초기화
init_db()

# 예제: 이미지 저장
image_path = 'F:/Repository/hAIr_Python/images/Autumn Gaze.jpg'  # 실제 경로로 변경하세요
# save_image_to_db(image_path)

# 예제: 모든 이미지 삭제
# delete_all_images()

# 예제: 특정 범위의 이미지 삭제
# delete_images_by_id_range(1, 10)

if __name__ == "__main__":
    # 필요한 함수 호출
    # save_image_to_db(image_path)
    show_image(15)
    # list_images()
    # delete_all_images()
    # delete_images_by_id_range(1, 10)
