# OpenCV(오픈소스 컴퓨터 비전 라이브러리) 개요

## 1. 한줄 요약
OpenCV는 실시간 이미지·비디오 처리를 위한 오픈소스 라이브러리로, 이미지 필터링·기하 변환·특징 검출·객체 인식·카메라 캘리브레이션·머신러닝 등 컴퓨터 비전 전반을 폭넓게 지원합니다.

---

## 2. 주요 특징
- 크로스 플랫폼 (Windows, Linux, macOS, Android, iOS)
- C/C++ 기반이며 Python, Java 바인딩 제공
- 실시간 영상 처리에 최적화된 빠른 구현
- 풍부한 알고리즘(특징 추출, 매칭, 분할, 추적 등)
- DNN 모듈로 딥러닝 모델(TensorFlow, Caffe, ONNX 등) 로드 가능

---

## 3. 설치 (Python 예시)
- pip:
```bash
pip install opencv-python        # 핵심 패키지 (대부분 사용)
pip install opencv-python-headless  # GUI/Display 불필요한 환경
pip install opencv-contrib-python # 추가 기여 모듈(추가 알고리즘)
```

- Conda:
```bash
conda install -c conda-forge opencv
```

---

## 4. 구조(모듈 개요) — 표
| 모듈 | 설명 |
|---|---|
| core | 행렬(Mat) 연산, 기본 자료구조 |
| imgproc | 필터, 변환, 기하학적 연산(리사이즈, 회전, 자르기 등) |
| highgui | 이미지·비디오 입출력, 창 관리 |
| video | 동영상 분석(광류, 배경차분 등) |
| features2d | 코너·특징점 검출과 디스크립터(SIFT/SURF/ORB 등) |
| calib3d | 카메라 캘리브레이션, 스테레오 매칭, 투영 변환 |
| objdetect | Haar/LSH 기반 객체 검출(예: 얼굴 검출) |
| ml | 전통 ML 알고리즘(SVM, Decision Trees 등) |
| dnn | 딥러닝 모델 로드 및 추론 인터페이스 |

---

## 5. 전형적인 처리 파이프라인 (다이어그램)
```mermaid
graph LR
  A[입력: 이미지/비디오] --> B[전처리: 리사이즈, 필터링]
  B --> C[특징 추출: 코너/특징점/디스크립터]
  C --> D[매칭/클러스터링/추적]
  D --> E[분류/추론(DNN 또는 ML)]
  E --> F[후처리: 위치 보정, 시각화]
  F --> G[출력: 화면/파일/데이터]
```

---

## 6. 간단한 예제 (Python)
- 이미지 읽기, 그레이스케일, Canny 엣지 검출
```python
import cv2

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 1.4)
edges = cv2.Canny(blur, 50, 150)

cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- 비디오 캡처와 얼굴 검출 (Haar Cascade)
```python
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

- DNN으로 이미지 분류(ONNX 모델 예시)
```python
import cv2
net = cv2.dnn.readNetFromONNX("model.onnx")
img = cv2.imread("image.jpg")
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(224,224), mean=(0,0,0), swapRB=True)
net.setInput(blob)
out = net.forward()
pred_class = out.argmax()
```

---

## 7. 자주 쓰이는 알고리즘/기능 요약
- 필터: GaussianBlur, medianBlur, bilateralFilter
- 엣지: Canny
- 변환: warpAffine, warpPerspective, resize
- 형태학: erode, dilate, morphologyEx
- 특징: FAST, ORB, SIFT(제한적 라이선스 주의), SURF
- 매칭: BFMatcher, FLANN
- 추적: KCF, CSRT, MIL, MOSSE
- 카메라 캘리브레이션: findChessboardCorners, calibrateCamera
- 객체 검출: CascadeClassifier, DNN 기반 YOLO/SSD 계열

---

## 8. 성능 팁
- 연산이 많은 루프는 C++로 구현하거나 Numpy 벡터 연산 사용
- ROI(관심 영역)만 처리하여 불필요 연산 최소화
- CV_8U 타입 유지: 변환을 자주 하지 말 것
- 멀티스레드/하드웨어 가속(예: OpenCL, CUDA) 사용 고려 (빌드 옵션 필요)

---

## 9. 라이선스 & 커뮤니티
- 대부분 BSD 스타일 라이선스(상업적 이용 가능)
- opencv.org, GitHub 저장소, StackOverflow, 다양한 튜토리얼과 강의 존재

---

## 10. 참고 리소스
- 공식 사이트: https://opencv.org
- 문서: https://docs.opencv.org
- GitHub: https://github.com/opencv/opencv

---

끝맺음: OpenCV는 실무와 연구에서 모두 널리 쓰이는 라이브러리입니다. 위 내용은 입문자에서 중급자까지 빠르게 참고할 수 있는 핵심 요약이며, 필요하면 특정 주제(예: 카메라 캘리브레이션, 딥러닝 연동, CUDA 가속 등)에 대해 더 상세한 예제와 튜토리얼을 추가로 만들겠습니다.
