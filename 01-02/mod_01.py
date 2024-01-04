from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
 
imgList = glob(r"C:\Users\bluecom001\Downloads\12-27\12-27\img\*.jpg")

def ImgFileOpen(imgList):
    xData, yData = [],[]
    for file in imgList:
        img = Image.open(file)
        img = np.array(file)
        
        xData.append(img)
        yData.append('flower')
        
    return xData, yData

if __name__ == "main":
    print(ImgFileOpen(imgList))
    
# def ImgPltShow(num=0, **kwargs):
#     x, y = kwargs['x'], kwargs['y']

#     for xD, yD in zip(x, y):
#         plt.imshow(xD)
#         plt.title(yD)
#         plt.show()
# img, label = ImgFileOpen(imgList)
# ImgPltShow(100, x = img, y = label) 