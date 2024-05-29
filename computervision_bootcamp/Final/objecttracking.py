from ultralytics import YOLO
import cv2
import pyttsx3

model = YOLO('yolov8n.pt')

video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)
ls = []
ret = True
while ret:
    ret,frame = cap.read() # ret if read is success
    results = model.track(frame,persist=True) # remenber past frame objects
    #print(results)
    #engine.say(results)
    #engine.runAndWait()
    frame_ = results[0].plot()
    #ls.append(frame_[0][0][0])
    #engine.say(results)
    #engine.runAndWait()

    cv2.imshow('frame',frame_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(ls)