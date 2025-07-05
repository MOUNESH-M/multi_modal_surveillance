import cv2
from ultralytics import YOLO
import pytesseract
import pyttsx3



#tesseract path installed in the local system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Speech enginee intialization
enginee = pyttsx3.init()
#Reducing the speed fo the voice alert
enginee.setProperty('rate', 125)

#function to make the voice alert
def speak(text):
     enginee.say(text)
     enginee.runAndWait()

def start_detection():
    cap = cv2.VideoCapture(0) #for video caputring
    model = YOLO('yolov8n.pt')  #pre trained nano You Only Look Once model developed by ultralytics

    #listing the suspicious objects
    suspicious_objects = ['backpack', 'knife', 'bottle', 'suitcase']

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read() #the read function reuturn two attributes, ret(boolen) - whether the frame is read or not & frame - actuall frame
        if not ret:
            print("Error: Cannot receive frame")
            break

        results = model.predict(source=frame, conf=0.5, stream=True)
        #to draw the bounding box
        for result in results: #results is a object that contians the list of predictions of the frame, result is the detection of single frame
            boxes = result.boxes #result.boxes - all detected bounding boxes of that frame
            names = result.names #result.names - dictionary mapping of class indices to the object names
            for box in boxes: #iterating over all the detected boxes in a single frame
                cls_id = int(box.cls[0]) #extracts the classID of the current detection from the class index tensor
                label = names[cls_id] #get the corresponding Classname for the ClassID from the dictionary

                x1, y1, x2, y2 = map(int, box.xyxy[0]) #box.xyxy gives the corrdinates of the detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #To draw the rectangel frame around the detected object
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) #to write the detected object name on the rectangle box

                #To identify the suspecious objects
                if label in suspicious_objects:
                    cv2.putText(
                        frame, "ALERT: Suspicious object detected..!", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
                    )
                    speak(f"Alert suspicious {label} detected")
        #text detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting to gray scale
        text = pytesseract.image_to_string(gray)#extracting text from the frame

        if text.strip() != "":
            cv2.putText(frame, f'Text Detected: {text.strip()[:30]}', (10, 450), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Multi-Modal Surveillance', frame)#to display the video

        if cv2.waitKey(1) == ord('q'): #to stop the video capturing
            break

    cap.release()#releasing the resource allocations
    cv2.destroyAllWindows()#Clossing all the windows

if __name__ == "__main__":
    start_detection()