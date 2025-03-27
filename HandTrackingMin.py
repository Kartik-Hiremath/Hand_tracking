import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()             # help(mp.solutions.hands.Hands)
mpDraw = mp.solutions.drawing_utils # There are totally 21 landmarks.
pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) 
    if results.multi_hand_landmarks:# If hand is detected then we will get some information.
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm) # We are getting only the ratio to get the pixel values look below.
                h, w, c = img.shape # height, width, channels.
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id in [4 ,8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 25, (245, 66, 239), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (0, 0, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
