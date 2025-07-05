import cv2
from ultralytics import YOLO
import pytesseract
import speech_recognition as sr
import pyttsx3
import threading
import difflib


pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

enginee=pyttsx3.init()
enginee.setProperty('rate', 125)

camera_active=True

def speak(text):
    enginee.say(text)
    enginee.runAndWait()

def listen_to_commands():
    global camera_active
    recognizer=sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Listing to the voice...")
            audio=recognizer.listen(source, timeout=7, phrase_time_limit=5)
            try:
                command=recognizer.recognize_google(audio).lower()
                if difflib.get_close_matches(command, ['start', 'stop', 'alert'], cutoff=0.7):

                    if 'start' in command:
                        camera_active=True
                        speak("camera activated")
                    elif 'stop' in command:
                        camera_active=False
                        speak("Camera Stopped")
                        break
                    elif 'alert' in command:
                        speak("Security system in activation")
            except sr.WaitTimeoutError:
                print("Wait time over")
                continue
            except sr.UnknownValueError:
                print("Cannot understand audio")
                speak("unrecognitsed command")
                continue
            except sr.RequestError:
                print("speech regcognisation error")
                continue

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
                    speak(f"Alert suspicious {label} detected")
                    
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text=pytesseract.image_to_string(gray)
       

        if text.strip()!="":
            cv2.putText(frame, f'Text Detected: {text.strip()[:30]}', (10,450), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
       
        cv2.imshow('Multi-Modal Surveillance', frame)

       
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=listen_to_commands).start()
    start_detection()