### Data Augmentation - Horizontal Image Flip ###

import cv2
import numpy as np
import os

### Define image dir ###
img_dir = "/content/drive/My Drive/Asgn1/training_data/Melange/"
img_files = os.listdir(img_dir)

for i23 in range(len(img_files)):
    
    IMAGE_PATH = img_dir + img_files[i23]
    img1 = cv2.imread(IMAGE_PATH)
    
	### Horizontal flip image using CV2 flip function ###
    horizontal_img = cv2.flip(img1, 1)
    
    out_name = img_dir + os.path.splitext(os.path.basename(img_files[i23]))[0]  + "_horzflip.jpg"
    #out_name = os.path.split(img_name)[0] + "/" + os.path.splitext(os.path.basename(img_name))[0] + 
    cv2.imwrite(out_name, horizontal_img)
