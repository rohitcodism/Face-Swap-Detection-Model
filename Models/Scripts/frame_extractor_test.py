import cv2 as cv
from mtcnn import MTCNN

detector = MTCNN()
video_path = 'Datasets/manipulated_sequences/01_02__outside_talking_still_laughing__YVGY8LOK.mp4'
video = cv.VideoCapture(video_path)

print("Hello World")

if not video.isOpened():
    exit()

frame_num = 0

frame_rate = video.get(cv.CAP_PROP_FPS)
new_frame_rate = frame_rate / 5

while True:
    isExist,frame = video.read()
    if not isExist:
        break
    if frame_num%5==0:
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
        cropped = frame[y-50:y+height+50,x-50:x+width+50]
        frame_file = f'frames/frame_{frame_num/5}.jpg'
        cv.imwrite(frame_file,cropped)
    frame_num+=1

frame_num=frame_num/5
# Releasing
video.release()