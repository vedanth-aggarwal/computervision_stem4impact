from ultralytics import YOLO
import cv2
"""
 Attributes:
        data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.

    Properties:
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray, optional): Tracking IDs for each box, if available.
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, calculated on demand.

"""
# Load yolov8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = 'data/testvid.mp4'
cap = cv2.VideoCapture(0)

ret = True
# Read frames
while True:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, device="mps", persist=True)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        label = results[0].boxes.cls.cpu().numpy().astype(int)
        print(label)
        d = {1:'person',44:'bottle'}
        for box, id in zip(boxes, ids):

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Id {id}/ Label {d[label]}",
                (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()