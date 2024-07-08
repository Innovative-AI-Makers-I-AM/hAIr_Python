import numpy as np
import cv2
import dlib
from sklearn.cluster import KMeans
import math
from math import degrees

# 파일 경로 설정
imagepath = "user_image.jpg"
face_cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Haar Cascade 및 dlib 예측기 생성
faceCascade = cv2.CascadeClassifier(face_cascade_path)
predictor = dlib.shape_predictor(predictor_path)

# 이미지 읽기 및 전처리
image = cv2.imread(imagepath)
image = cv2.resize(image, (500, 500))
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gray, (3, 3), 0)

# 얼굴 검출
faces = faceCascade.detectMultiScale(gauss, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
print(f"Found {len(faces)} faces!")

for (x, y, w, h) in faces:
    dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    detected_landmarks = predictor(image, dlib_rect).parts()
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    
    results = original.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(results, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 이마 영역 설정 및 KMeans 클러스터링 적용
        forehead = original[y:y+int(0.25*h), x:x+w]
        rows, cols, bands = forehead.shape
        X = forehead.reshape(rows*cols, bands)
        
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(X)
        for i in range(0, rows):
            for j in range(0, cols):
                if y_kmeans[i*cols+j] == 1:
                    forehead[i][j] = [255, 255, 255]
                else:
                    forehead[i][j] = [0, 0, 0]
        
        forehead_mid = [int(cols/2), int(rows/2)]
        lef = 0
        pixel_value = forehead[forehead_mid[1], forehead_mid[0]]
        for i in range(0, cols):
            if forehead[forehead_mid[1], forehead_mid[0]-i].all() != pixel_value.all():
                lef = forehead_mid[0]-i
                break
        left = [lef, forehead_mid[1]]
        rig = 0
        for i in range(0, cols):
            if forehead[forehead_mid[1], forehead_mid[0]+i].all() != pixel_value.all():
                rig = forehead_mid[0]+i
                break
        right = [rig, forehead_mid[1]]
        
        line1 = np.subtract(right, left)[0]
        cv2.line(results, tuple(np.add(left, [x, y])), tuple(np.add(right, [x, y])), color=(0, 255, 0), thickness=2)
        cv2.putText(results, 'Line 1', tuple(np.add(left, [x, y])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, tuple(np.add(left, [x, y])), 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, tuple(np.add(right, [x, y])), 5, color=(255, 0, 0), thickness=-1)
        
        # 주요 랜드마크 점 설정 및 선 그리기
        linepointleft = (landmarks[1, 0], landmarks[1, 1])
        linepointright = (landmarks[15, 0], landmarks[15, 1])
        line2 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, 'Line 2', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)
        
        linepointleft = (landmarks[3, 0], landmarks[3, 1])
        linepointright = (landmarks[13, 0], landmarks[13, 1])
        line3 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, 'Line 3', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)
        
        linepointbottom = (landmarks[8, 0], landmarks[8, 1])
        linepointtop = (landmarks[8, 0], y)
        line4 = np.subtract(linepointbottom, linepointtop)[1]
        cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
        cv2.putText(results, 'Line 4', linepointbottom, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)
        
        # 얼굴 형태 분석
        similarity = np.std([line1, line2, line3])
        ovalsimilarity = np.std([line2, line4])
        
        ax, ay = landmarks[3, 0], landmarks[3, 1]
        bx, by = landmarks[4, 0], landmarks[4, 1]
        cx, cy = landmarks[5, 0], landmarks[5, 1]
        dx, dy = landmarks[6, 0], landmarks[6, 1]
        
        alpha0 = math.atan2(cy-ay, cx-ax)
        alpha1 = math.atan2(dy-by, dx-bx)
        alpha = alpha1-alpha0
        angle = abs(degrees(alpha))
        angle = 180 - angle
        
        for i in range(1):
            if similarity < 10:
                if angle < 160:
                    print('Squared shape. Jawlines are more angular')
                    break
                else:
                    print('Round shape. Jawlines are not that angular')
                    break
            if line3 > line1:
                if angle < 160:
                    print('Triangle shape. Forehead is more wider')
                    break
            if ovalsimilarity < 10:
                print('Diamond shape. Line2 & Line4 are similar and Line2 is slightly larger')
                break
            if line4 > line2:
                if angle < 160:
                    print('Rectangular. Face length is largest and jawline are angular')
                    break
                else:
                    print('Oblong. Face length is largest and jawlines are not angular')
                    break
            print("Damn! Contact the developer")

output = np.concatenate((original, results), axis=1)
cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()