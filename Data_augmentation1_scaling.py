### Data Augmentation - Center Image Scaling ###

import cv2
import numpy as np
import os

### Define image dir ###
img_dir = "/content/drive/My Drive/Asgn1/training_data/embroidery/"
img_files = os.listdir(img_dir)

### Define the scaling box for image dimensions ###
scales = [0.90, 0.80, 0.70]

boxes = np.zeros((len(scales), 4), dtype = np.float32)
for index, scale in enumerate(scales):
    x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
    x2 = y2 = 0.5 + 0.5 * scale
    boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)

### Read images in training folder and scale & resize them ###

for i in range(len(img_files)):
    
    IMAGE_PATH = img_dir + img_files[i]
    img1 = cv2.imread(IMAGE_PATH)
    
	### Scale image to scales defines above and save each scaled image ###
    x, y, ch = img1.shape
    for i22 in range(len(scales)):
        scl1 = img1[int(x*boxes[i22][1]):int(x*boxes[i22][3]),int(y*boxes[i22][0]):int(y*boxes[i22][2])]
        rsz_img = cv2.resize(scl1,(y,x), interpolation = cv2.INTER_AREA)

        out_name = img_dir + os.path.splitext(os.path.basename(img_files[i]))[0]  + "_crop" + str(i22) + ".jpg"
        #out_name = os.path.split(img_name)[0] + "/" + os.path.splitext(os.path.basename(img_name))[0] + 
        cv2.imwrite(out_name, rsz_img)
