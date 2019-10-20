from architectures import models
import cv2
from matplotlib import pyplot as plt

from train import train



def happy():
    # load parameters and create detection model
    #params = load params from somewhere
    #model = models.Yolo(params)
    
    # train model
    #train(model, params)...
    

    # load and display test image
    image = cv2.imread('img3.jpg')
    img = image[:,:,::-1]
    plt.imshow(img)
    plt.show()
    
    
    
    # give image as input 
    # result = model(input)
    
    
    # display(input, result)
    result = cv2.imread('img3_det.jpg')
    img2 = result[:,:,::-1]
    plt.imshow(img2)
    plt.show()
    