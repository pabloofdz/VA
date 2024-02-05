#Pablo Fernández Pérez (pablo.fperez@udc.es)

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color, filters, morphology, measure
from scipy.spatial.distance import cdist


#Convierte la imagen a escala de grises si es necesario
def toGrayScale(inImage):
    if inImage.shape[-1] == 3:
        return color.rgb2gray(inImage)
    else:
        return inImage


#Realiza una expansion del histograma para distinguir mejor los objetos, ignorando el 0, que es el fondo.
def adjustIntensityWithout0(inImage, inRange=[], outRange=[0, 1]):
    if inRange == []:
        imin = np.min(inImage[inImage > 0])
        imax = np.max(inImage)
    else:
        imin = inRange[0]
        imax = inRange[1]
    
    omin = outRange[0]
    omax = outRange[1]
    
    #Alteración del rango dinámico
    outImage = omin + ((omax - omin)*(inImage - imin))/(imax - imin) 

    outImage = np.where(inImage == 0, 0, outImage)
    
    return outImage


#Calcula el umbral de Otsu. Solo se tiene en cuenta lo que hay dentro del circulo para que el fondo negro no influya
def calculateOtsuThreshold(inImage):
    #Se crea una máscara circular del mismo tamaño que la imagen
    y, x = np.indices(inImage.shape)

    #Se calcula el centro de la imagen (asumiendo que es cuadrada)
    centro = inImage.shape[0]//2

    #El radio es la mitad del lado de la imagen
    radio = centro                 

    #Se crea una máscara circular usando la ecuación del círculo
    circularMask = (x - centro)**2 + (y - centro)**2 <= radio**2

    #Se aplica la máscara para calcular el umbral de Otsu solo dentro del círculo
    otsuThreshold = filters.threshold_otsu(inImage[circularMask])

    return otsuThreshold


#Se eliminan los objetos grandes (area>maxSize) y pequeños (max>area>min)(para tratar de eliminar el fondo y la tráquea). Se devuelve una imagen binaria.
def removeObjects(labels, min, max, maxSize):
    result = np.copy(labels)
    regions = measure.regionprops(result)
    for region in regions:
        if min < region.area < max or region.area > maxSize:
            result[result == region.label] = 0
        else:
            result[result == region.label] = 1
    
    return result


#Permite separar la imagen etiquetada en dos imagenes, una con cada pulmón
def splitLungs(lungs):
    labels = measure.label(lungs.copy())

    regions = measure.regionprops(labels)

    #Se verifica que haya al menos dos regiones
    if len(regions) < 2:
        raise ValueError("No se han conseguido detectar los dos pulmones. Compruebe que la imagen de entrada es correcta, en caso afirmativo, podría ser necesario un ajuste de parámetros en alguna operación.")

    #Se ordenan las regiones por área
    regions.sort(key=lambda x: x.area, reverse=True)

    #Se obtienen las etiquetas de las dos regiones con mayor área
    label1 = regions[0].label
    label2 = regions[1].label

    #Se crean dos imágenes (una para cada pulmón)
    lung1 = np.zeros_like(labels)
    lung2 = np.zeros_like(labels)

    #Nos quedamos con los dos objetos más grandes
    for region in regions:
        if region.label == label1:
            lung1[labels == region.label] = 1
        elif region.label == label2:
            lung2[labels == region.label] = 1

    return lung1, lung2


#Rellena los huecos presentes en los objetos de la imagen
def fillGaps(inImage, threshold=128):
    inImage = (inImage * 255).astype(np.uint8)
    _, binaryImage = cv2.threshold(inImage, threshold, 255, cv2.THRESH_BINARY)

    #Se encuentran los contornos en la imagen binarizada
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Se rellenan los huecos en cada objeto
    cv2.drawContours(binaryImage, contours, -1, 255, -1)

    return binaryImage


#Marca los pulmones con un borde rojo
def markLungs(inImage, closedLungsMask):
    #Se obtienen las coordenadas de los bordes
    borders = measure.find_contours(closedLungsMask, level=0.5)

    inImage = color.gray2rgb(inImage)
    imageWithBorders = np.copy(inImage)

    for border in borders:
        border = np.array(border, dtype=int)
        imageWithBorders[border[:, 0], border[:, 1], :] = [255, 0, 0]#Marcar en rojo

    return imageWithBorders


#Algoritmo que permite la deteccion de ambos pulmones
def lungsSegmentation(inImage):

    inImage = toGrayScale(inImage)

    #Expandimos el histograma de la imagen para distinguir mejor los distintos objetos
    adjustedImage = adjustIntensityWithout0(inImage)
    
    adjustedImage[adjustedImage < 0] = 0
    
    otsuThreshold = calculateOtsuThreshold(adjustedImage)

    #Se disminuye un poco el valor de otsu para que se distinga mejor la separación de los elementos cuando están muy pegados (como la tráquea)
    binaryImage = adjustedImage > otsuThreshold - 0.055
    
    #Se invierte la imagen para detectar componentes conexas
    invertedImage = np.invert(binaryImage)

    #Se detectan las componentes conexas
    labels = measure.label(invertedImage)
    
    #Se eliminan los objetos grandes y pequeños (se trata en este paso de eliminar la tráquea, así como el fondo)
    lungsMask = removeObjects(labels, 100, 1000, 30000)

    #Hacemos un cierre no muy agresivo simplemente para asegurarnos de que cada pulmón tiene una única etiqueta y así poder separarlos
    closedLungsMask = morphology.binary_closing(lungsMask, morphology.disk(7))

    #Ahora aplicamos una apertura para asegurarnos de que los pulmones no se unan en el paso anterior
    closedLungsMask = morphology.binary_opening(closedLungsMask, morphology.disk(1))

    #Separamos los pulmones para evitar que ambos se unan durante las operaciones siguientes
    lung1, lung2 = splitLungs(np.copy(closedLungsMask))

    #Ahora se aplica una operación de cierre más agresiva para incluir los nódulos pegados a los bordes
    closedLung1 = morphology.binary_closing(lung1, morphology.disk(25))
    closedLung2 = morphology.binary_closing(lung2, morphology.disk(25))

    #Se rellenan los posibles huecos en los pulmones debido a elementos de menor densidad
    filledLung1 = fillGaps(closedLung1)
    filledLung2 = fillGaps(closedLung2)

    #Obtenemos de nuevo una imagen con los 2 pulmones
    finalLungsMask = filledLung1 + filledLung2

    #Se marcan los bordes de los pulmones sobre la imagen original
    result = markLungs(inImage, finalLungsMask)
    
    #Se muestran resultados
    plt.figure(figsize=(18, 9))

    plt.subplot(3, 3, 1)
    plt.imshow(inImage, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Original')

    plt.subplot(3, 3, 2)
    plt.imshow(adjustedImage, cmap='gray', vmin=0, vmax=1)
    plt.title('Imagen Ajustada')
    
    plt.subplot(3, 3, 3)
    plt.imshow(invertedImage, cmap='gray')
    plt.title('Binaria Invertida')
    
    plt.subplot(3, 3, 4)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title('Etiquetas de Componentes Conexas')
    
    plt.subplot(3, 3, 5)
    plt.imshow(lungsMask, cmap='gray')
    plt.title('Sin objetos pequeños y grandes')

    plt.subplot(3, 3, 6)
    plt.imshow(closedLungsMask, cmap='gray')
    plt.title('Máscara antes de dividir los pulmones')
    
    plt.subplot(3, 3, 7)
    plt.imshow(closedLung1 + closedLung2, cmap='gray')
    plt.title('Máscara después del cierre')

    plt.subplot(3, 3, 8)
    plt.imshow(finalLungsMask, cmap='gray')
    plt.title('Máscara después del relleno')

    plt.subplot(3, 3, 9) 
    plt.imshow(result, cmap='gray')
    plt.title('Resultado final')
    
    plt.tight_layout()
    plt.show()

    return result, finalLungsMask


#Calcula la precisión, la sensibilidad o recall, la similitud y el coeficiente de Similaridad de Dice (DSC)
def calculatePrecRecallSimDSC(resultMask, solMask):
    #Se calculan los verdaderos positivos, falsos positivos y falsos negativos
    tp = np.sum(np.logical_and(resultMask, solMask))
    fp = np.sum(np.logical_and(resultMask, ~solMask))
    fn = np.sum(np.logical_and(~resultMask, solMask))

    #Se calcula la precisión
    precision = 0
    if (tp + fp) > 0:
        precision = tp/(tp + fp)
    
    #Se calcula la sensibilidad
    recall = 0
    if (tp + fn) > 0:
        recall = tp/(tp + fn)

    #Se calcula la similitud
    sim = 1 - np.sqrt((1 - precision)**2 + (1 - recall)**2)/np.sqrt(2)

    #Se calcula el coeficiente de Similaridad de Dice (DSC)
    dsc = (2*tp)/(2*tp + fp + fn)

    return precision, recall, sim, dsc


#Calcula la similitud
def calculateSim(precision, recall):
    return 1 - np.sqrt((1 - precision)**2 + (1 - recall)**2) / np.sqrt(2)


#Calcula la distancia media cuadrática propuesta (FOM)
def calculateFOM(resultMask, solMask, p=1):
    #Se encuentran los contornos de las máscaras
    solContours = measure.find_contours(solMask, 0.5)
    resultContours = measure.find_contours(resultMask, 0.5)

    #Se concatenan todos los contornos en dos listas
    solBorderPixels = np.concatenate(solContours, axis=0)
    resultBorderPixels = np.concatenate(resultContours, axis=0)

    #Se calcula la distancia mínima de cada punto en un contorno al otro contorno
    distances = cdist(solBorderPixels, resultBorderPixels, 'euclidean')
    minDistance = np.min(distances, axis=1)

    #Se calcula N, el número máximo de puntos en los contornos
    N = max(len(solBorderPixels), len(resultBorderPixels))

    #Calculamos FOM según la fórmula
    fom = (1/N) * np.sum(1 / (1 + p * minDistance**2))

    return fom


if __name__ == "__main__":

    #Se crea la carpeta ./resultados si no existe
    resultsFolder = './resultados'
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    for i in range(1, 8):
        imInputPath = f'./imagenes/im{i}.png'
        solInputPath = f'./soluciones/sol{i}.png'

        inImage = io.imread(imInputPath)
        solMask = io.imread(solInputPath)

        outImage, mask = lungsSegmentation(inImage)

        outputPath = os.path.join(resultsFolder, f'res{i}.png')
        io.imsave(outputPath, outImage.astype(np.uint8))
        print(f"{outputPath}' guardada.")

        outputMasksPath = os.path.join(resultsFolder, f'resMask{i}.png')
        io.imsave(outputMasksPath, mask)
        print(f"{outputMasksPath}' guardada.")

        print(f'EVALUACIÓN IMAGEN {i}')
        print("--------------------------------")
        precision, recall, sim, dsc = calculatePrecRecallSimDSC(mask, solMask)
        print(f'Precision = {precision}')
        print(f'Sensibilidad = {recall}')
        print(f'Similitud = {sim}')
        print(f'Coeficiente de Similaridad de Dice = {dsc}')
        FOMresult = calculateFOM(mask, solMask)
        print(f'FOM = {FOMresult}')
        print("--------------------------------")
