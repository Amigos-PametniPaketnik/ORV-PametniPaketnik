import sys
import numpy as np
import cv2
import pickle
from sklearn.neural_network import MLPClassifier 

# Izračun matrike smeri Gradientov v kotih izraženih v stopinjah
def smerGradienta(gx, gy):
    theta = np.arctan2(gy, gx) # Izračun smeri gradientov iz horizontalne in vertikalne slike robov
    theta[theta > 0] *= 180 / 3.14 # Dobljene kote iz radianov pretvorimo v stopinje, negativne kote pretvorimo v pozitivne
    theta[theta < 0] = (theta[theta < 0] + 3.14) * 180 / 3.14
    return theta.astype(np.uint8)

# Algoritem izločanja lokalnega histograma glede na velikost škatle M
def izracunajLoklaniHistogram(izrezSmeri, izrezGradienta):
    histogram = np.zeros(9)

    for i in np.arange(0, izrezSmeri.shape[0]):
        for j in np.arange(0, izrezGradienta.shape[0]):
            for x in np.arange(0, 9): # Razvrščanje smeri v razrede kotov
                if (izrezSmeri[i,j] >= (x * 20) and izrezSmeri[i,j] < ((x+1) * 20)):
                    if (izrezSmeri[i,j] == (x * 20)): # Kot je enak razredu
                        histogram[x] += izrezGradienta[i,j]
                    elif (izrezSmeri[i,j] > (x * 20) and izrezSmeri[i,j] < ((x+1) * 20)):
                        if (((x+1) * 20) < 180): # Kot se nahaja med dvema razredoma; izračunamo deleže po razredih
                            spodnjaMeja = x * 20
                            zgornjaMeja = (x+1) * 20
                            interval = zgornjaMeja - spodnjaMeja
                            histogram[x] += ((izrezSmeri[i,j] - spodnjaMeja) / interval) * izrezGradienta[i,j]
                            histogram[x+1] += ((1 - (izrezSmeri[i, j] - spodnjaMeja) / interval)) * izrezGradienta[i, j]
                        else: # Kot je v zadnjem razredu, večji deleži se preslikajo oz. prištejejo v prvi razred
                            spodnjaMeja = x * 20
                            zgornjaMeja = (x + 1) * 20
                            interval = zgornjaMeja - spodnjaMeja
                            histogram[x] += ((izrezSmeri[i, j] - spodnjaMeja) / interval) * izrezGradienta[i, j]
                            histogram[0] += ((1 - (izrezSmeri[i, j] - spodnjaMeja) / interval)) * izrezGradienta[i, j]
    return histogram

# Algoritem izločanja značilnic histograma usmerjenih gradientov
def histogramUsmerjenihGradientov(slika):
    dodatniRobovi = [0,0,0,0] # zgoraj, spodaj, levo, desno
    N = 8
    B = 9
    M = 2
    #dodatniRobovi[0] = int((slika.shape[0] & 7) / 2)
    #dodatniRobovi[1] = int((slika.shape[1] & 7) / 2)
    #slika = cv2.copyMakeBorder(slika, dodatniRobovi[0], dodatniRobovi[0], dodatniRobovi[1], dodatniRobovi[1]) # V primeru, da slika ni deljiva z 8, razširimo njene robove
    gx = cv2.Sobel(slika, cv2.CV_32F, 1,0, ksize=3)  # Izračunamo sliko robov po x osi
    gy = cv2.Sobel(slika, cv2.CV_32F, 0,1, ksize=3)  # Izračunamo sliko robov po y osi

    smer = smerGradienta(gx, gy) # Izračunamo smeri gradientov v stopinjah
    gx = np.uint8(np.absolute(gx))
    gy = np.uint8(np.absolute(gy))
    g = np.sqrt(gx**2 + gy**2).astype(np.uint8) # Izračunamo še skupno sliko robov združeno po x in y osi

    histogram = np.ndarray((int(g.shape[0]/N), int(g.shape[1]/N)), dtype=object)
    for i in np.arange(0, g.shape[0], N): # Po celicah 8x8 pikslov izračunamo histograme
        for j in np.arange(0, g.shape[1], N):
            izrezSmeri = smer[i:i+N,j:j+N]
            izrezGradienta = g[i:i+N,j:j+N]
            lokalniHistogram = izracunajLoklaniHistogram(izrezSmeri, izrezGradienta)
            histogram[int(i/N),int(j/N)] = lokalniHistogram

    koncenNormaliziranV = np.array([])
    for i in np.arange(0, histogram.shape[0]-1): # Izvedemo normalizacijo histogramov po velikosti škatle 2x2 (M - 16x16 pikslov)
        for j in np.arange(0, histogram.shape[0]-1):
            v = np.array([])
            v = np.append(v, histogram[i,j])
            v = np.append(v, histogram[i,j+1])
            v = np.append(v, histogram[i+1,j])
            v = np.append(v, histogram[i+1,j+1])
            k = np.sqrt((v**2).sum())
            normaliziranV = v/k
            koncenNormaliziranV = np.append(koncenNormaliziranV, normaliziranV)

    return koncenNormaliziranV

# Algoritem izločanja značilnic s lokalnimi binarnimi vzorci
def lokalniBinarniVzorci(slika): # Funkcija za izvedbo konvolucije
    (visinaSlike, sirinaSlike) = slika.shape[:2]
    slika = cv2.copyMakeBorder(slika, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)
    izhodnaSlika = np.zeros((visinaSlike, sirinaSlike), dtype=np.uint8) # Incializiramo novo matriko za izhod konvolucije
    for i in np.arange(1, visinaSlike): # Izvedemo postopek nad celotno vhodno sliko
        for j in np.arange(1, sirinaSlike):
            izrez = slika[i-1:i+2,j-1:j+2]
            binarnaMatrikaIzreza = cv2.threshold(izrez, izrez[1,1]-1, 1, cv2.THRESH_BINARY)
            bin = 0                                            # Bite iz binarne slike okolice piksla dodajamo v izhodno vrednost
            bin = (bin << 1) | binarnaMatrikaIzreza[1][1,0]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][2,0]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][2,1]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][2,1]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][1,2]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][0,2]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][0,1]
            bin = (bin << 1) | binarnaMatrikaIzreza[1][0,0]
            izhodnaSlika[i,j] = bin  # V centralni piksel zapišemo dobljeno vrednost
    histogram = np.histogram(izhodnaSlika, 256, [0, 256])[0] # Sestavimo histogram sivin za celotno sliko iz LBP značilnic
    return histogram

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


operacija = sys.argv[1] # opcije -> 0: preverjanje indentitete osebe na podlagi dobljene slike; 1: dodajanje nove slike osebe in učenje mreže

if operacija == "1":
    slika, valid = detekcijaIzrezObraza("images/"+sys.argv[2]) # Detekcija obraza se bo izvedla nad naloženo sliko podane v prvem argumentu te skripte
    if valid == True :
        slika = cv2.resize(slika, (64, 64)) # Pomanjšamo izhodni izrez obraza, pri strojnem učenju mora biti nabor vseh slik enakih dimenzij
        ucnaMnozica = np.array([])
        labeleUcneMnozice = np.array([])
        with open("featuresmodels/ucnamnozica.npy", "rb") as d:
            ucnaMnozica = np.load(d)
            labeleUcneMnozice = np.load(d)
        
        lbp = lokalniBinarniVzorci(slika).astype(np.float64)  # Izračunamo značilnice LBP iz posamezne slike
        #lbp = skimage.feature.local_binary_pattern(slikice[i], 8 * 1, 1)
        hog = histogramUsmerjenihGradientov(slika) # Izračunamo značilnice Histograma usmerjwnih gradientov
        #hog = skimage.feature.hog(slikice[i], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        znacilnice = np.concatenate((lbp, hog)) # LBP in HOG značilnice združimo v skupen vektor

        ucnaMnozica = np.vstack([ucnaMnozica, znacilnice])
        labeleUcneMnozice = np.append(labeleUcneMnozice, int(sys.argv[3]))
        with open("featuresmodels/ucnamnozica.npy", "wb") as d:
            np.save(d, ucnaMnozica)
            np.save(d, labeleUcneMnozice)
        cv2.imwrite("faces/"+sys.argv[2]+".png", slika) # Zapišemo zaznan obraz v izhodno datoteko sviniske slike v mapi faces
        clf = MLPClassifier(hidden_layer_sizes=(512,), solver='lbfgs', max_iter=500, verbose=True).fit(mnozica, labele)
        pickle.dump(clf, open("naucenmodel.sav", 'wb'))

if operacija == "0":
    naucenModelMreze = pickle.load(open("featuresmodels/naucenmodelClanov.sav", 'rb'))
    slika, valid = detekcijaIzrezObraza("images/"+sys.argv[2]) # Detekcija obraza se bo izvedla nad naloženo sliko podane v prvem argumentu te skripte

    if valid == True :
        slika = cv2.resize(slika, (64, 64)) # Pomanjšamo izhodni izrez obraza, pri strojnem učenju mora biti nabor vseh slik enakih dimenzij

        lbp = lokalniBinarniVzorci(slika).astype(np.float64)  # Izračunamo značilnice LBP iz posamezne slike
        #lbp = skimage.feature.local_binary_pattern(slikice[i], 8 * 1, 1)
        hog = histogramUsmerjenihGradientov(slika) # Izračunamo značilnice Histograma usmerjwnih gradientov
        #hog = skimage.feature.hog(slikice[i], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        znacilnice = np.concatenate((lbp, hog)) # LBP in HOG značilnice združimo v skupen vektor

        rezultat = naucenModelMreze.predict_proba(znacilnice.reshape(1, -1))
        if np.max(rezultat) > 0.80:
            print(np.argmax(rezultat))
