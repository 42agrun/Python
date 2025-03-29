import cv2
import time

# These variables are used to differentiate between smiling and not-smiling while counting instances
# _time1 & _time2 serve as a "cooldown" to prevent the counter (_smiles) from uncontrollably incrementing
_time1, _time2, _smiles = 0, 0, 0
_is_smiling = False

def detect_face(frame):
    global _smiles, _is_smiling, _time1, _time2
    # These cascade models are freely available by simply googling them
    # There's also a model for eyes ("haarcascade_eye.xml") for those eager to explore
    face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("cascades/haarcascade_smile.xml")

    # The cascade model seems to favor input via graytones, too many colors are deemed too challenging
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # The four variables used here refer to the camera dimensions and the pattern rectangles inside
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
        face = frame[y: y + h, x: x + w]
        gray_face = gray[y: y + h, x: x + w]

        smiles = smile_cascade.detectMultiScale(gray_face,1.5, minNeighbors=9)
        for (xp, yp, wp, hp) in smiles:
            face = cv2.rectangle(face, (xp, yp), (xp + wp, yp + hp), color=(0, 0, 255), thickness=5)
            # Rectangles are only drawn if a smile was detected, this is when the counting occurs
            if _is_smiling is False:
                _time1 = time.time()
                _smiles += 1
                _is_smiling = True
            else:
                _time2 = time.time()
                # Built-in cooldown of one second between smiles
                if _time2 - _time1 > 1:
                    _is_smiling = False
    return frame

def run_webcam():
    global _smiles
    stream = cv2.VideoCapture(0)
    if not stream.isOpened():
        print("Couldn't connect to any webcam. Shutting down.")
        exit()
    print("Connection has been established. Press Q to exit.")

    ### WEBCAM SETUP ###
    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
    fps = stream.get(cv2.CAP_PROP_FPS)
    width = int(stream.get(3))
    height = int(stream.get(4))
    output = cv2.VideoWriter("test_stream.mp4", fourcc, fps=fps, frameSize=(width, height))

    ### WEBCAM LOOP ###
    while True:
        ret, frame = stream.read()
        if not ret:
            print("Connection to webcam lost.")
            break
        frame = detect_face(frame)
        output.write(frame)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    ### END OF LOOP ###

    stream.release()
    cv2.destroyAllWindows()
    print(f"Total number of smiles: {_smiles}")

if __name__ == "__main__":
    print("This app will count how many times you're smiling.")
    print("Setting up the webcam-connection may take up to 10 seconds.")
    print("Please wait...")
    run_webcam()