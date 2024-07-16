from database import init_db
from images_utils import save_image_to_db, delete_all_images, delete_images_by_id_range, show_image, list_images, save_image_metadata_to_db

# 데이터베이스 초기화
init_db()

# 예제: 이미지 저장
image_path = 'F:/Repository/hAIr_Python/images/Autumn Gaze.jpg'  # 실제 경로로 변경하세요

metadata = {
    "sex": "여성",
    "length": "롱",
    "style": "레이어드컷",
    "designer": "디자이너 정재원",
    "shop_name": "모어온헤어 신논현점",
    "hashtag1": "C컬펌",
    "hashtag2": "레이어드펌",
    "hashtag3": "레이어드컷"
}
# save_image_to_db(image_path)

# 예제: 모든 이미지 삭제
# delete_all_images()

# 예제: 특정 범위의 이미지 삭제
# delete_images_by_id_range(1, 10)

if __name__ == "__main__":
    # 필요한 함수 호출
    # save_image_to_db(image_path)
    # show_image(15)
    # list_images()
    save_image_metadata_to_db(image_path, metadata)
    # delete_all_images()
    # delete_images_by_id_range(1, 10)
