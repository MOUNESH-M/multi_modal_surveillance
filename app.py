import cv2
import pytesseract
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def start_detection():
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt') 

    suspicious_objects=['backpack','knife','bottle','suitcase']

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
            names=result.names
            for box in boxes:
                cls_id=int(box.cls[0])
                label=names[cls_id]

                x1,y1,x2,y2=map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0),2)

                if label in suspicious_objects:
                    cv2.putText(
                        frame,"ALERT: Suspicious object detected..!",(50,50),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2
                        )
                    
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text=pytesseract.image_to_string(gray)
       

        if text.strip()!="":
            cv2.putText(frame, f'Text Detected: {text.strip()[:30]}', (10,450), cv2.FONT_HERSHEY_COMPLEX,7.5,(255,0,0),1)
       
        cv2.imshow('Multi-Modal Surveillance', frame)

       
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()
