import cv2 as cv
detect = cv.CascadeClassifier("content\haarcascade_frontalface_default.xml")
eye = cv.CascadeClassifier("content\haarcascade_eye.xml")
cam = cv.VideoCapture(0)
#algo for face recogniser
while True:
    ret, img = cam.read()
    faces = detect.detectMultiScale(img, 1.3, 5)
    for (x ,y,w,h) in faces:
        cv.rectangle(img ,(x,y), (x+w, y+h),(100,105,306),3)
        eyes = eye.detectMultiScale(img)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(img,(ew,eh),(ex+ew,ey+eh),(0,255,0),2)
        cv.imshow('frame',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
