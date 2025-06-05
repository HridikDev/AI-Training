import cv2
from deepface import DeepFace
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    emotion = results[0]['dominant_emotion']
    cv2.putText(frame, f'Emotion: {emotion}', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),2)

    cv2.imshow("Emotion Recognition", frame)

    if emotion != voices:
        engine.say(f"I am {emotion}")
        engine.runAndWait()
        last_emotion = emotion

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        cap.release()
        cv2.destroyALLWindiows()