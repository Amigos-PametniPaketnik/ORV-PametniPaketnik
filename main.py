import numpy as np
import cv2

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