#Práctica 1 VA
#Pablo Fernández Pérez (pablo.fperez@udc.es)
#---------------------------------------------------------------

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

background = 0

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):
    if inRange == []:
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin = inRange[0]
        imax = inRange[1]
    
    omin = outRange[0]
    omax = outRange[1]
    
    #Alteración del rango dinámico
    outImage = omin + ((omax - omin)*(inImage - imin))/(imax - imin) 
    
    return outImage


def equalizeIntensity(inImage, nBins=256):
    m, n = inImage.shape

    #Obtención del histograma de la imagen
    hist, _ = np.histogram(inImage, bins=nBins, range=(0, 1))

    #Normalización del histograma acumulado para la obtención de la función de transferencia t
    t = hist.cumsum() / (m*n)

    #Ecualización
    outImage = t[np.round(inImage*(nBins - 1)).astype(int)]
        
    return outImage


def filterImage(inImage, kernel):
    m, n = inImage.shape
    
    #Tamaño del kernel
    if kernel.ndim == 1:
      p = 1
      q = kernel.shape[0]
    elif kernel.ndim == 2:
      p, q = kernel.shape
    else:
      raise ValueError("Kernel inválido.")
    
    #Centro del kernel
    center = [int(np.floor(p/2)), int(np.floor(q/2))]
    
    #Para que la imagen de salida matenga el tamaño de la original necesitamos rellenar esta última
    topRows = center[0]
    bottomRows = p - 1 - center[0]
    leftCols = center[1]
    rightCols = q - 1 - center[1]
    inImage = np.pad(inImage, [(topRows, bottomRows), (leftCols, rightCols)], mode='reflect')

    outImage = np.zeros((m, n), dtype=np.float32)
    
    #Convolución
    for i in range(m):
        for j in range(n):
            window = inImage[i:i + p, j:j + q]
            outImage[i, j] = np.sum(window*kernel)

    return outImage


def gaussKernel1D(sigma):
    n = int(2*np.ceil(3*sigma) + 1)

    center = np.floor(n/2)
    kernel = np.zeros(n)

    for i in range(n):
        x = i - center
        kernel[i] = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x**2)/(2*sigma**2))
        
    return kernel


def gaussianFilter(inImage, sigma):
    kernel = gaussKernel1D(sigma)

    outImage = filterImage(inImage, kernel)

    outImage = filterImage(outImage,  kernel[:, np.newaxis])
        
    return outImage


def medianFilter(inImage, filterSize):
    m, n = inImage.shape
    
    #Centro del filtro
    center = int(np.floor(filterSize/2))
    
    #Para que la imagen de salida matenga el tamaño de la original necesitamos rellenar esta última
    topRows = center
    bottomRows = filterSize - 1- center
    leftCols = center
    rightCols = filterSize - 1 - center
    inImage = np.pad(inImage, [(topRows, bottomRows), (leftCols, rightCols)], mode='reflect')

    outImage = np.zeros((m, n), dtype=np.float32)
    
    for i in range(m):
        for j in range(n):
            window = inImage[i:i + filterSize, j:j + filterSize]
            outImage[i, j] = np.median(window)
    
    return outImage


def erodeAux(a, b):
    #Verifica si todos los elementos iguales a 1 en a también son 1 en b
    return np.all(a[a == 1] == b[a == 1])


def dilateAux(a, b):
    #Verifica si al menos un 1 de a está en la misma posición en b
    return np.any(a[a == 1] == b[a == 1])


def erode(inImage, SE, center=[]):
    m, n = inImage.shape

    if SE.ndim == 1:
      p = 1
      q = SE.shape[0]
      SE = np.atleast_2d(SE)#Para que la operación funcione correctamente convertimos a 2D
    elif SE.ndim == 2:
      p, q = SE.shape
    else:
      raise ValueError("SE inválido.")
    
    outImage = np.zeros((m, n), dtype=np.float32)
    
    if center == []:
        center = [int(np.floor(p/2)), int(np.floor(q/2))]
    
    #Para realizar la erosión necesitamos rellenar la imagen original con el color del fondo en los bordes en función de SE
    topRows = center[0]
    bottomRows = p - 1- center[0]
    leftCols = center[1]
    rightCols = q - 1 - center[1]
    inImage = np.pad(inImage, [(topRows, bottomRows), (leftCols, rightCols)], mode='constant', constant_values=background)

    for i in range(m):
        for j in range(n):
            if erodeAux(SE, inImage[i:i + p, j:j + q]):
                outImage[i, j] = 1
                
    return outImage


def dilate(inImage, SE, center=[]):
    m, n = inImage.shape

    if SE.ndim == 1:
      p = 1
      q = SE.shape[0]
      SE = np.atleast_2d(SE)#Para que la operación funcione correctamente convertimos a 2D
    elif SE.ndim == 2:
      p, q = SE.shape
    else:
      raise ValueError("SE inválido.")
    
    outImage = np.zeros((m, n), dtype=np.float32)
    
    if center == []:
        center = [int(np.floor(p/2)), int(np.floor(q/2))]
    
    #Para realizar la erosión necesitamos rellenar la imagen original con el color del fondo en los bordes en función de SE
    topRows = center[0]
    bottomRows = p - 1 - center[0]
    leftCols = center[1]
    rightCols = q - 1 - center[1]
    inImage = np.pad(inImage, [(topRows, bottomRows), (leftCols, rightCols)], mode='constant', constant_values=background)

    for i in range(m):
        for j in range(n):
            if dilateAux(SE, inImage[i:i + p, j:j + q]):
                outImage[i, j] = 1
                
    return outImage


def opening(inImage, SE, center=[]):
    outImage = erode(inImage, SE, center)
    outImage = dilate(outImage, SE, center)
    return outImage


def closing(inImage, SE, center=[]):
    outImage = dilate(inImage, SE, center)
    outImage = erode(outImage, SE, center)
    return outImage


def invert(matrix):
    m, n = matrix.shape

    invertedMatrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 0:
                invertedMatrix[i, j] = 1

    return invertedMatrix


def intersect(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Las matrices no tienen las mismas dimensiones.")

    m, n = matrix1.shape

    matrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            if matrix1[i, j] == 1 and matrix2[i, j] == 1:
                matrix[i, j] = 1

    return matrix


def hit_or_miss(inImage, objSEj, bgSE, center=[]):
    global background
    #Si para alguna posición de PxQ hay unos simultáneamente en objSE y bgSE se debe indicar error
    if objSEj.shape != bgSE.shape or np.any(objSEj*bgSE):
        raise ValueError("Error: elementos estructurantes incoherentes")

    result1 = erode(inImage, objSEj, center)

    #Para la búsqueda de patrones de fondo debemos invertir tanto la imagen como el fondo con el que se rellenará para que no pierda tamaño
    background = 1
    result2 = erode(invert(inImage), bgSE, center)
    background = 0
    
    outImage = intersect(result1, result2)
    return outImage


def gradientImage(inImage, operator):
    if operator == 'Roberts':
        xMask = np.array([[-1, 0],
                          [0, 1]])
        yMask = np.array([[0, -1],
                          [1, 0]])

    elif operator == 'CentralDiff':
        xMask = np.array([[-1, 0, 1]])
        yMask = xMask.T
        
    elif operator == 'Prewitt':
        xMask = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
        yMask = xMask.T

    elif operator == 'Sobel':
        xMask = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
        yMask = xMask.T

    else:
        raise ValueError("Operador inválido. Indicar 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'.")

    return filterImage(inImage, xMask), filterImage(inImage, yMask)


def zeroCrossing(image, t):
    m, n = image.shape
    result = np.zeros((m, n))
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            neighbors = np.array([image[i-1, j], image[i+1, j], 
                                  image[i, j-1], image[i, j+1],
                                  image[i-1, j-1], image[i+1, j+1],
                                  image[i-1, j+1], image[i+1, j-1]])
            if (image[i,j] < -t and np.any(neighbors > t)):
                result[i,j] = 1

    return result


def LoG(inImage, sigma):
    outImage = gaussianFilter(inImage, sigma)
    laplacianKernel = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])
    outImage = filterImage(outImage, laplacianKernel)
    return outImage


#Supresión no máxima
def cannySupression(magnitude, gradientAngle):
    m, n = magnitude.shape

    result = np.zeros((m, n))

    #Se analiza cada píxel y si es menor que alguno de los píxeles vecinos en la direccion del gradiente se pone a 0, si no, se queda como estaba
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if (0 <= gradientAngle[i, j] < 22.5) or (157.5 <= gradientAngle[i, j] <= 180):
                n1 = magnitude[i, j+1]
                n2 = magnitude[i, j-1]
            elif (22.5 <= gradientAngle[i, j] < 67.5):
                n1 = magnitude[i+1, j+1]
                n2 = magnitude[i-1, j-1]
            elif (67.5 <= gradientAngle[i, j] < 112.5):
                n1 = magnitude[i+1, j]
                n2 = magnitude[i-1, j]
            elif (112.5 <= gradientAngle[i, j] < 157.5):
                n1 = magnitude[i+1, j-1]
                n2 = magnitude[i-1, j+1]
            
            if (magnitude[i, j] < n1) or (magnitude[i, j] < n2):
                result[i, j] = 0
            else:
                result[i, j] = magnitude[i, j]

    return result

#Umbralización con histéresis
def cannyThresholding(suppressed, tlow, thigh, gradientAngle):
    m, n = suppressed.shape

    #Nos quedamos con las posiciones de los píxeles con bordes fuertes y bordes débiles
    strongBorderPixels = (suppressed > thigh)
    weakBorderPixels = (suppressed > tlow) & (suppressed <= thigh)

    #Ponemos los píxeles con bordes fuertes a 1
    suppressed = np.where(strongBorderPixels, 1, suppressed)
    
    #Los píxeles vecinos a los píxeles con bordes fuertes (en la direccion perpendicular a la normal del borde) se ponen a 1 si son > tlow
    #Esto se repite hasta que no hay cambios con respecto a la iteración anterior
    while True:
        previousSupressed = np.copy(suppressed)
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if strongBorderPixels[i, j]:
                    if (0 <= gradientAngle[i, j] < 22.5) or (157.5 <= gradientAngle[i, j] <= 180):
                        if weakBorderPixels[i+1, j]:
                            suppressed[i+1, j] = 1
                        if weakBorderPixels[i-1, j]:
                            suppressed[i-1, j] = 1
                    elif (22.5 <= gradientAngle[i, j] < 67.5):
                        if weakBorderPixels[i+1, j-1]:
                            suppressed[i+1, j-1] = 1
                        if weakBorderPixels[i-1, j+1]:
                            suppressed[i-1, j+1] = 1
                    elif (67.5 <= gradientAngle[i, j] < 112.5):
                        if weakBorderPixels[i, j+1]:
                            suppressed[i, j+1] = 1
                        if weakBorderPixels[i, j-1]:
                            suppressed[i, j-1] = 1
                    elif (112.5 <= gradientAngle[i, j] < 157.5):
                        if weakBorderPixels[i+1, j+1]:
                            suppressed[i+1, j+1] = 1
                        if weakBorderPixels[i-1, j-1]:
                            suppressed[i-1, j-1] = 1

        #Se actualizan las posiciones de los píxeles con bordes fuertes
        strongBorderPixels = (suppressed > thigh)
        if np.array_equal(previousSupressed, suppressed):
            break  
    
    #Los píxeles que no son borde se ponen a 0
    nonBorderPixels = (suppressed < 1)
    suppressed = np.where(nonBorderPixels, 0, suppressed)

    return suppressed


def edgeCanny (inImage, sigma, tlow, thigh):
    outImage = gaussianFilter(inImage, sigma)
    [gx, gy] = gradientImage (outImage, 'Sobel')
    magnitude = np.sqrt(gx**2 + gy**2)
    
    orientation = np.arctan2(gy, gx)
    gradientAngle = orientation * 180 / np.pi
    #Acotamos los angulos al rango [0,180] para simplificar los ifs dado que seguimos teniendo todas las direcciones necesarias
    gradientAngle = np.where((gradientAngle < 0), gradientAngle+180, gradientAngle)

    suppressed = cannySupression(magnitude, gradientAngle)
    
    outImage = cannyThresholding(suppressed, tlow, thigh, gradientAngle)

    return outImage


#Permite obtener una máscara circular (una matriz en la que los elementos que forman el círculo están a True)
def getCircularMask(r):
    mask = np.zeros((r*2+1, r*2+1), dtype=bool)

    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if np.sqrt(i**2 + j**2) <= r:
                mask[i+r, j+r] = True

    return mask


#Esta versión incluye un número mínimo de n para que un punto sea considerado esquina. Esto soluciona ciertos problemas: 
# pixeles negros rodeados de blanco considerados esquinas, bordes considerados esquinas por el gradiente 
# (muy pocos pixeles grises que coinciden con el centro hacen que se detecte una esquina).
def cornerSusan(inImage, r, t):
    p, q = inImage.shape
    outCorners = np.zeros((p, q))
    usanArea = np.zeros((p, q))
    mask = getCircularMask(r)
    #Se calcula el umbral fijo g (umbral geométrico), que está fijado en 3/4 el valor máximo que puede tomar n
    maxn = np.sum(mask)
    g = (3/4)*maxn
    for i in range(r, p-r):
        for j in range(r, q-r):
            centralPixel = inImage[i, j]
            
            circularArea = inImage[i-r : i+r+1, j-r : j+r+1]
            circularArea = circularArea[mask]

            #Se calcula n, que es el número de píxeles cuya diferencia de luminosidad con el núcleo no supera el umbral t
            n = np.sum(np.abs(circularArea - centralPixel) <= t)

            if n < g and n > (1/8)*maxn:
                outCorners[i, j] = g-n

            usanArea[i, j] = n

    return outCorners, usanArea


def getCorners(outCorners, t):
    m, n = outCorners.shape

    onlyCorners = np.zeros((m, n))

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            neighbors = np.array([outCorners[i-1, j], outCorners[i+1, j], 
                                  outCorners[i, j-1], outCorners[i, j+1],
                                  outCorners[i-1, j-1], outCorners[i+1, j+1],
                                  outCorners[i-1, j+1], outCorners[i+1, j-1]])

            if outCorners[i, j] >= np.max(neighbors) and outCorners[i, j] > t:
                onlyCorners[i, j] = 1

    return onlyCorners


def readImage(route):
    inImage = io.imread(route)
    
    if inImage is None:
        raise ValueError("No se pudo cargar la imagen.")
    elif inImage.ndim == 2:
        if (1 > np.min(inImage) > 0) & (1 > np.max(inImage) > 0):
            return inImage
        else:
            return adjustIntensity(inImage)
    else:
        inImage = inImage[:, :, :3]
        #Normalización de la imagen
        inImage = color.rgb2gray(inImage)
        return inImage


def showResult(inImage, outImage, outImageName="outImage"):
    plt.figure(figsize=(12, 8))
    
    #Imagen original
    plt.subplot(2, 3, 1)
    plt.imshow(inImage, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Imagen Original')

    #Histograma de la imagen original
    plt.subplot(2, 3, 2)
    histOriginal, binsOriginal, _ = plt.hist(inImage.ravel(), bins=256, range=(0, 1), color='b', alpha=0.7)
    plt.title('Histograma Original')

    #Histograma acumulado de la imagen original
    plt.subplot(2, 3, 3)
    cumHistOriginal = np.cumsum(histOriginal)
    plt.plot(binsOriginal[:-1], cumHistOriginal, color='b')
    plt.title('Histograma Acumulado Original')

    #Imagen resultante
    plt.subplot(2, 3, 4)
    plt.imshow(outImage, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Imagen Resultante')

    #Histograma de la imagen resultante
    plt.subplot(2, 3, 5)
    histResult, binsResult, _ = plt.hist(outImage.ravel(), bins=256, range=(0, 1), color='r', alpha=0.7)
    plt.title('Histograma Resultante')

    #Histograma acumulado de la imagen resultante
    plt.subplot(2, 3, 6)
    cumHistResult = np.cumsum(histResult)
    plt.plot(binsResult[:-1], cumHistResult, color='r')
    plt.title('Histograma Acumulado Resultante')

    plt.tight_layout()
    plt.show()
        
    #Guardar imagen
    io.imsave(outImageName + '.jpg', (outImage * 255).astype(np.uint8))
    print("'" + outImageName + ".jpg' guardada.")


def showOutImages(inImage, outImage1, outImage2, title, outImage1Name, outImage2Name):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 3, 1)
    plt.imshow(inImage, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Imagen Original')

    plt.subplot(2, 3, 2)
    plt.imshow(outImage1, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title(outImage1Name)

    plt.subplot(2, 3, 3)
    plt.imshow(outImage2, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title(outImage2Name)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title)
    plt.show()
        
    #Guardar imagen
    io.imsave(outImage1Name + '.jpg', (outImage1 * 255).astype(np.uint8), cmap='gray')
    print("'" + outImage1Name + ".jpg' guardada.")
    io.imsave(outImage2Name + '.jpg', (outImage2 * 255).astype(np.uint8), cmap='gray')
    print("'" + outImage2Name + ".jpg' guardada.")


if __name__ == "__main__":
        
        #EJEMPLO adjustIntensity
        #----------------------------------------------------
        """
        inImage = readImage('gato.jpg')
        
        outImage = adjustIntensity(inImage, [], [0.4, 0.6])
        showResult(inImage, outImage, "gatoAjust1")

        outImage = adjustIntensity(inImage)
        showResult(inImage, outImage, "gatoAjust2")
        """
        #----------------------------------------------------

        #EJEMPLO equalizeIntensity
        #----------------------------------------------------
        """
        inImage = readImage('tucan.png')
        outImage = equalizeIntensity(inImage)
        showResult(inImage, outImage, "tucanEcualizado")


        inImage = readImage('gato.jpg')
        outImage = equalizeIntensity(inImage, 128)
        showResult(inImage, outImage, "gatoEcualizado")
        """
        #----------------------------------------------------
    
        #EJEMPLO filterImage
        #----------------------------------------------------
        """
        #Filtro que produce valores fuera de rango
        inImage = readImage('gato.jpg')
        kernel = np.array([[1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1]], dtype=np.float32) / 9.0
        
        outImage = filterImage(inImage, kernel)
        outImage = adjustIntensity(outImage)
        showResult(inImage, outImage, "gatofiltrado")

        #Filtro de medias
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]], dtype=np.float32) / 25.0
        inImage = readImage('cuadradob.jpg')
        outImage = filterImage(inImage, kernel)
        showResult(inImage, outImage, "cuadradofiltrado")

        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=np.float32) / 9.0
        inImage = readImage('saltpepper.png')
        outImage = filterImage(inImage, kernel)
        showResult(inImage, outImage, "saltpepperfiltrado")
        
        inImage = np.zeros([11,11])
        inImage[6,6] = 1
        kernel = readImage('cara.png')
        outImage = filterImage(inImage, kernel)
        showResult(inImage, outImage, "pruebaFilter")
        """
        #----------------------------------------------------

        #EJEMPLO gaussianFilter
        #----------------------------------------------------
        """
        sigma = 15
        kernel = gaussKernel1D(sigma)
        plt.plot(kernel)

        inImage = readImage('saltpepper.png')
        outImage = gaussianFilter(inImage, 2)
        showResult(inImage, outImage, "gaussianFiltered")
        """
        #----------------------------------------------------

        #EJEMPLO medianFilter
        #----------------------------------------------------
        """
        inImage = readImage('saltpepper.png')
        outImage = medianFilter(inImage, 7)
        showResult(inImage, outImage, "medianFiltered1")
        
        inImage = readImage('pruebaDiapos.png')
        outImage = medianFilter(inImage, 3)
        showResult(inImage, outImage, "medianFiltered2")
        """
        #----------------------------------------------------

        #EJEMPLO erode y dilate
        #----------------------------------------------------
        """
        SEDiapos = np.array([1, 1])

        inImageDiapos = np.array([[1, 0, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 1, 1, 0],
                                  [0, 1, 0, 0],
                                  [0, 1, 0, 0]])
        outImage = erode(inImageDiapos, SEDiapos, [0,0])
        showResult(inImageDiapos, outImage, "erode")
        outImage = dilate(inImageDiapos, SEDiapos)
        showResult(inImageDiapos, outImage, "dilate")

        inImage = readImage('image.png')
        SE =  np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
        outImage = invert(erode(invert(inImage), SE))
        showResult(inImage, outImage, "imageEroded")
        outImage = invert(dilate(invert(inImage), SE))
        showResult(inImage, outImage, "imageDilated")
        """
        #----------------------------------------------------

        #EJEMPLO opening y closing
        #----------------------------------------------------
        """
        SE =  np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
        
        openingImage = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                 [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                                 [0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                                 [0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0],
                                 [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0],
                                 [0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0],
                                 [0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0],
                                 [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
                                 [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                                 [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                                 [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
                                 [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                                 [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                                 [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        outImage = opening(openingImage, SE)
        showResult(openingImage, outImage, "opening")

        closingImage = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                 [0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0],
                                 [0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0],
                                 [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0],
                                 [0,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0],
                                 [0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,0],
                                 [0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0],
                                 [0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                                 [0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0],
                                 [0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                                 [0,1,1,0,0,0,0,1,1,1,0,1,0,0,0,0],
                                 [0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0],
                                 [0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        outImage = closing(closingImage, SE)
        showResult(closingImage, outImage, "closing")

        inImage = readImage('image.png')
        outImage = invert(opening(invert(inImage), SE))
        showResult(inImage, outImage, "imageOpened")
        outImage = invert(closing(invert(inImage), SE))
        showResult(inImage, outImage, "imageClosed")
        """
        #----------------------------------------------------

        #EJEMPLO hit or miss
        #----------------------------------------------------
        """
        objSEj = np.array([[0, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0]])

        bgSE = np.array([[0, 1, 1],
                         [0, 0, 1],
                         [0, 0, 0]])

        inImageDiaposHoM = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 0],
                                     [0, 0, 1, 1, 1, 1, 0],
                                     [0, 0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]])
        
        inImage = inImageDiaposHoM
        outImage = hit_or_miss(inImage, objSEj, bgSE)
        showResult(inImage, outImage, "hit_or_miss")
        """
        #----------------------------------------------------

        #EJEMPLO gradientImage
        #----------------------------------------------------
        """
        inImage = readImage('circulob.jpg')
        
        operator = 'Roberts'
        [gx, gy] = gradientImage (inImage, operator)
        #magnitude = np.sqrt(gx**2 + gy**2)
        #showResult(inImage, magnitude, "gradientImage")
        gx = adjustIntensity(gx)
        gy = adjustIntensity(gy)
        showOutImages(inImage, gx, gy, operator, "Gx", "Gy")

        operator = 'CentralDiff'
        [gx, gy] = gradientImage (inImage, operator)
        #magnitude = np.sqrt(gx**2 + gy**2)
        #showResult(inImage, magnitude, "gradientImage")
        gx = adjustIntensity(gx)
        gy = adjustIntensity(gy)
        showOutImages(inImage, gx, gy, operator, "Gx", "Gy")

        operator = 'Prewitt'
        [gx, gy] = gradientImage (inImage, operator)
        #magnitude = np.sqrt(gx**2 + gy**2)
        #showResult(inImage, magnitude, "gradientImage")
        gx = adjustIntensity(gx)
        gy = adjustIntensity(gy)
        showOutImages(inImage, gx, gy, operator, "Gx", "Gy")
        
        operator = 'Sobel'
        [gx, gy] = gradientImage (inImage, operator)
        #magnitude = np.sqrt(gx**2 + gy**2)
        #showResult(inImage, magnitude, "gradientImage")
        gx = adjustIntensity(gx)
        gy = adjustIntensity(gy)
        showOutImages(inImage, gx, gy, operator, "Gx", "Gy")
        """
        #----------------------------------------------------

        #EJEMPLO LoG
        #----------------------------------------------------
        """
        #Figuras básicas
        inImage = readImage('cuadradob.jpg')
        outImage = LoG(inImage, 1.0)
        outImage = zeroCrossing(outImage, 0.015)
        showResult(inImage, outImage, "cuadradolog")

        inImage = readImage('circulob.jpg')
        outImage = LoG(inImage, 1.0)
        outImage = zeroCrossing(outImage, 0.015)
        showResult(inImage, outImage, "circulolog")

        inImage = readImage('triangulob.jpg')
        outImage = LoG(inImage, 1.0)
        outImage = zeroCrossing(outImage, 0.015)
        showResult(inImage, outImage, "triangulolog")

        #Diapositivas
        inImage = readImage('lenna.png')

        outImage = LoG(inImage, 0.5)
        outImage = adjustIntensity(outImage)
        showResult(inImage, outImage)

        outImage = LoG(inImage, 0.5)
        outImage = zeroCrossing(outImage, 0.075)
        showResult(inImage, outImage, "lennalog2")

        outImage = LoG(inImage, 1.5)
        outImage = adjustIntensity(outImage)
        showResult(inImage, outImage)

        outImage = LoG(inImage, 1.5)
        outImage = zeroCrossing(outImage, 0.015)
        showResult(inImage, outImage, "lennalog1")
        """
        #----------------------------------------------------

        #EJEMPLO Operador de Canny
        #----------------------------------------------------
        """
        #Figuras básicas
        inImage = readImage('cuadradob.jpg')
        outImage = edgeCanny(inImage, 0.17, 0.01, 0.9)
        showResult(inImage, outImage, "cuadradocanny")

        inImage = readImage('circulob.jpg')
        outImage = edgeCanny(inImage, 0.17, 0.01, 0.9)
        showResult(inImage, outImage, "circulocanny")

        inImage = readImage('triangulob.jpg')
        outImage = edgeCanny(inImage, 0.17, 0.01, 0.9)
        showResult(inImage, outImage, "triangulocanny")

        #Otras pruebas
        inImage = readImage('circles.png')
        outImage = edgeCanny(inImage, 0.17, 0.01, 0.9)
        showResult(inImage, outImage, "circlescanny")

        inImage = readImage('circles1.png')
        outImage = edgeCanny(inImage, 0.17, 0.01, 0.9)
        showResult(inImage, outImage, "circles1canny")
        """
        #----------------------------------------------------

        #EJEMPLO Detector de esquinas Susan
        #----------------------------------------------------
        """
        #Figuras básicas
        inImage = readImage('cuadradob.jpg')
        outCorners, usanArea = cornerSusan(inImage, 20, 0.5)
        outCorners = adjustIntensity(outCorners)
        usanArea = adjustIntensity(usanArea)
        showOutImages(inImage, outCorners, usanArea, "Detector de esquinas Susan", "outCornersCuadr", "usanAreaCuadr")
        showResult(inImage, getCorners(outCorners, 0.9), "cuadradosusan")
        
        #(El círculo no debería aparecer, solo las esquinas del triángulo)
        inImage = readImage('triang_circ.jpg')
        outCorners, usanArea = cornerSusan(inImage, 20, 0.5)
        outCorners = adjustIntensity(outCorners)
        usanArea = adjustIntensity(usanArea)
        showOutImages(inImage, outCorners, usanArea, "Detector de esquinas Susan", "outCornersTC", "usanAreaTC")
        showResult(inImage, getCorners(outCorners, 0.9), "tcsusan")
        
        inImage = readImage('circles.png')
        outCorners, usanArea = cornerSusan(inImage, 10, 0.075)
        outCorners = adjustIntensity(outCorners)
        usanArea = adjustIntensity(usanArea)
        showOutImages(inImage, outCorners, usanArea, "Detector de esquinas Susan", "outCorners", "usanArea")
        showResult(inImage, getCorners(outCorners, 0.5), "susan")
        """
        #----------------------------------------------------