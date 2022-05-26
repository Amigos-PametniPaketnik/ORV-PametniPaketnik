import cv2

def detekcijaIzrezObraza(img_path):
    valid= False
    #vir xml datotek za iskanje obraza:https://github.com/opencv/opencv/tree/4.x/data/haarcascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    img = cv2.imread(img_path)
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if(len(faces)==1):
        valid = True

        for (x, y, w, h) in faces:
            crop_img = gray[y:y+h, x:x+w]
        return crop_img,valid
    
    else:
        return img,valid
    

    
    

#img,valid=detekcijaIzrezObraza('test.jpg')
#cv2.imshow("cropped", img)
#cv2.waitKey()