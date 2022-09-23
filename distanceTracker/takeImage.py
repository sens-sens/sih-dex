import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
_, frame = cap.read()
img = cv2.imwrite('image.jpg', frame)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
