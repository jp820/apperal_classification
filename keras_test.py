
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import os

### Load keras model ###
model = load_model('/content/drive/My Drive/Asgn1/config5_50_epochs_try2.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

### Test dir ###
test_dir = "/content/drive/My Drive/Asgn1/testing_data/"

### List all test images and iterate over them for prediction ###
img_files = os.listdir(test_dir)
cls_predct = []
img_width, img_height = 300, 300

for i in range(len(img_files)):
   
  ### Read current test image file and preprocess for prediction ###
  img = cv2.imread(test_dir + img_files[i])
  img = cv2.resize(img,(img_width,img_height))
  img = img.astype(np.float32)
  img = np.multiply(img, 1.0 / 255.0)
  img = np.reshape(img,[1,img_width,img_height,3])

  classes = model.predict_classes(img)
  cls_predct.append(classes)
  print (img_files[i], classes)

### Save prediction results to CSV ###
result_file = pd.DataFrame({'Image_Name': img_files,
     'Class': cls_predct
    })
result_file.to_csv(os.path.join(SOURCE_FOLDER ,'keras_fifth_model2_result.csv'))
