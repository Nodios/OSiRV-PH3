import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def loadImageGrayscale(path):
    '''
    Loads grayscale image from path.

    Returns image.
    '''
    
    out = cv2.imread(path, 0)
    return out

def cvt2float32(image):
    '''
    Converts given image data type to float32.

    Returns image.
    '''
    
    out = image.astype(np.float32)
    return out

def cvt2uint8(image):
    '''
    Converts given image data type to uint8.

    Returns image.
    '''
    
    out = image.astype(np.uint8)
    return out

def intensityRescaling(imagePath,a,b):
    '''
    Loads grayscale image and preforms intensity rescaling over loaded image.
    Input is image path followed by integers representing range [low,high] of grayscale values of image.

    Output image is saved as "<name>_a-b.bmp". Where 'a' and 'b' are values representing range of grayscale values.
    '''

    #extract name w/o extension from file path
    base = os.path.basename(imagePath)
    baseName = os.path.splitext(base)[0]
    
    #load grayscale
    original = loadImageGrayscale(imagePath)
    cv2.imshow('original',original)

    original = cvt2float32(original)

    h = np.size(original,0)
    v = np.size(original,1)
    
    outputImage = np.zeros((h,v))

    f_max,f_min = np.amax(original),np.amin(original)
    
    for i in range(np.size(original,0)):
        for j in range(np.size(original,1)):
            outputImage[i,j] = ((b-a)/(f_max-f_min)) * (original[i,j] - f_min) + a

    #save output
    saveName = baseName+str(a)+'-'+str(b)+'.bmp'
    cv2.imwrite(saveName, outputImage)
    
    #show output and histogram
    cv2.imshow('output',cvt2uint8(outputImage))
    plt.hist(outputImage.ravel(), 256, [0,256])
    plt.show()

def gammaCorrection(imagePath, y):
    '''
    Preforms Gamma correction over a given image.
    Input is image path followed by gamma value.

    Output image is saved as "<name>_gamma.bmp"
    '''

    #extract name w/o extension from file path
    base = os.path.basename(imagePath)
    baseName = os.path.splitext(base)[0]

    #load grayscale
    original = loadImageGrayscale(imagePath)
    cv2.imshow('original',original)

    original = cvt2float32(original)

    #create hxv zero matrix that will hold output image
    h = np.size(original,0)
    v = np.size(original,1)
    outputImage = np.zeros((h,v))
    
    f_max = np.amax(original)

    for i in range(np.size(original,0)):
        for j in range(np.size(original,1)):
            outputImage[i,j] = f_max*(pow((original[i,j]/f_max),(1/y)))

    #save output
    saveName = baseName+'_'+str(y)+'.bmp'
    cv2.imwrite(saveName, outputImage)

    #show output
    cv2.imshow('output',cvt2uint8(outputImage))

def histogramEqualize(imagePath):
    '''
    Preforms histogram equalization over given image.
    Input is image path.

    Output image is saved as "<name>_equalized.bmp"
    '''
    #extract name w/o extension from file path
    base = os.path.basename(imagePath)
    baseName = os.path.splitext(base)[0]

    #load grayscale
    original = loadImageGrayscale(imagePath)

    #create histogram
    hist, bins = np.histogram(original.flatten(),256,[0,256])

    #Cumulative distribution function - cumulative sum of elements in hist
    cdf = hist.cumsum()

    #normalize
    cdf_normalize = cdf * hist.max() / cdf.max()

    #show original image and plot histogram
    cv2.imshow('original',original)
    plt.plot(cdf_normalize, color = 'b')
    plt.hist(original.flatten(), 256, [0,256], color='r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'),loc='upper left')
    plt.show()

    #find minimum histogram value(w/o zero)
    cdf_min = np.ma.masked_equal(cdf,0)
    cdf_min = (cdf_min - cdf_min.min()) * 255 / (cdf_min.max()-cdf_min.min())
    cdf = np.ma.filled(cdf_min,0).astype('uint8')

    
    outputImage = cdf[original]

    #show output image and plot histogram
    cv2.imshow('output',outputImage)
    plt.plot(cdf_normalize, color = 'b')
    plt.hist(outputImage.flatten(), 256, [0,256], color='r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'),loc='upper left')
    plt.show()

    #save results
    saveName = baseName+'_equalized.bmp'
    cv2.imwrite(saveName, outputImage)

def cheatHistogramEqualization(imagePath):
    '''
    Preforms histogram equalization over given image
    Uses predefined OpenCV method cv2.equalizeHist()
    '''
    #extract name w/o extension from file path
    base = os.path.basename(imagePath)
    baseName = os.path.splitext(base)[0]

    #load grayscale
    original = loadImageGrayscale(imagePath)
    cv2.imshow('original',original)

    #output
    outputImage = cv2.qualizeHist(original)
    cv2.imshow('output', outputImage)

    #save results
    saveName = baseName+'_equalized.bmp'
    cv2.imwrite(saveName, outputImage)
    
