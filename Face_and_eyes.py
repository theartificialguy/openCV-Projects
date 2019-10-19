import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cap.release() #releasing video object
cv2.destroyAllWindows()