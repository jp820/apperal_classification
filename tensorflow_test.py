import tensorflow as tf
import cv2
import numpy as np
import pandas as pd 
import os

### Input Image dimesions for network ##
image_size = 227
num_channels=3

### Test dir ###
test_dir = "/content/drive/My Drive/Asgn1/testing_data/"

### List all test images and run predictions one by one ###
img_files = os.listdir(test_dir)
cls_predct = []

for i in range(len(img_files)):
    
	### Read current image file ###
    IMAGE_PATH = test_dir + img_files[i]
    image = cv2.imread(IMAGE_PATH)

    #img_name = "65408.jpg"
    # Reading the image using OpenCV
    #image = cv2.imread(test_dir + img_name)

    print (image.shape)
    images = []
    
	### Preprocess image before feeding to network ###
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 

    x_batch = images.reshape(1, image_size,image_size,num_channels)

`	### Start TF session and load network graph and weights ###
    sess = tf.Session()

    saver = tf.train.import_meta_graph(os.path.join(SOURCE_FOLDER ,'Fynd_data_model1.ckpt-500.meta'))
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

	### Get output tensor from graph ###
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ### Feed the testing images to the input placeholders ###
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, len(os.listdir('training_data')))) 

    ### Create the feed_dict and run session to predict y_pred  ###
    feed_dict_testing = {x: x_batch , y_true: y_test_images, keep_prob : 1.0}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    #print(result)
    
    cls_predct.append(result.argmax(axis = 1))
    
    print (IMAGE_PATH , result.argmax(axis = 1))

### Save prediction results to CSV
result_file = pd.DataFrame({'Image_Name': img_files,
     'Class': cls_predct
    })
result_file.to_csv(os.path.join(SOURCE_FOLDER ,'first_model_result.csv'))

