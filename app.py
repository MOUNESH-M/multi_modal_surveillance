import cv2
from ultralytics import YOLO

def start_detection():
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt') 

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot receive frame")
            break

       
        results = model.predict(source=frame, conf=0.5, stream=True)

        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                conf = box.conf[0]  
                cls = int(box.cls[0])  

               
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

       
        cv2.imshow('YOLO Detection', frame)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()
