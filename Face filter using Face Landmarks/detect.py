import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture("data/video.mp4")
prevTime = 0

# total 468 landmarks
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3) # max_num_faces=1
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1200, 650))
    # mediapipe needs RGB image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                frame, 
                face_landmarks, 
                mpFaceMesh.FACE_CONNECTIONS,
                drawSpec,
                drawSpec
            )
            # get each landmark info
            for lm_id, lm in enumerate(face_landmarks.landmark):
                # getting original value
                h, w, c = rgb.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.putText(
                    frame,
                    str(lm_id),
                    (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, 
                    (0,0,255),
                    1
                )

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(
        frame, 
        f"FPS: {str(round(fps))}", 
        (20, 70), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0,0,255),
        3
    )

    cv2.imshow("output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()