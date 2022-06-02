import sys
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

slika, valid = detekcijaIzrezObraza("images/"+sys.argv[1]) # Detekcija obraza se bo izvedla nad naloženo sliko podane v prvem argumentu te skripte
if valid == True :
    slika = cv2.resize(slika, (64, 64)) # Pomanjšamo izhodni izrez obraza, pri strojnem učenju mora biti nabor vseh slik enakih dimenzij
    # TODO: Izločitev značilnic iz slike, zapis/dodajanje novega vektorja značilnic v polje značilnic učne množice
    cv2.imwrite("faces/"+sys.argv[1]+".png", slika) # Zapišemo zaznan obraz v izhodno datoteko sviniske slike v mapi faces