import cv2

def start_camera():

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open the camera")
        return
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Fail to caputure...(Stream end)")
            break
        cv2.putText(frame,"Surveillance camera active..",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Multi-modal Surveillance",frame)

        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    start_camera()