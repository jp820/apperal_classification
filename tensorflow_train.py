import sys
#sys.path.append('drive/My Drive/Fynd_data')
#import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os

### Set the working dir ###
os.chdir("/content/drive/My Drive/Asgn1/")

### Adding Seed so that random initialization is consistent ###
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32

### Input data preparation ###
classes = os.listdir('training_data')
del(classes[6])
num_classes = len(classes)

img_size = 227
num_channels = 3
train_path='training_data'
print (classes)


### Create placefolders for training ###
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


### --- Define Network function --- ###

### Network graph params ###
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 128

filter_size_conv4 = 3
num_filters_conv4 = 256
    
fc_layer_size = 256


keep_prob = tf.placeholder(tf.float32)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
	### Define weights and biases ###
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    ### Create the convolutional layer ###
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    
    layer += biases

    ### Add Max-pooling layer with filter size and stride as 2 ###
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

	### Use ReLU activation function   
    layer = tf.nn.relu(layer)
        
    return layer

### Convolutional Layer with no max-pooling ###
def create_convolutional_layer_nomax_pool(input,
                                           num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ### Define weights and biases ###
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    ### Create the convolutional layer ###
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    layer += biases

    ### Use ReLU activation function   
    layer = tf.nn.relu(layer)
        
    return layer   

def create_flatten_layer(layer):
    
	### Get the input layer shape and create flatten layer using reshape ###	    
	layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    ### Define weights and biases ###
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

	### Use matmul function, to generate ouptuts using weights
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


### --- Define Network achitecture using above define functions --- ###

### Conv 1 with no max pool ###
layer_conv1 = create_convolutional_layer_nomax_pool(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

### Conv 2 with Max-pool ###
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

### Dropout after conv2 ###
drop_out_layer1 = tf.nn.dropout(layer_conv2, keep_prob)

### Conv 3 with no max pool ###
layer_conv3= create_convolutional_layer_nomax_pool(input=drop_out_layer1,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

### Conv 4 with Max-pool ###
layer_conv4= create_convolutional_layer(input=layer_conv3,
               num_input_channels=num_filters_conv3,
               conv_filter_size=filter_size_conv4,
               num_filters=num_filters_conv4)

### Dropout after conv2 ###
drop_out_layer2 = tf.nn.dropout(layer_conv4, keep_prob)

### Flatten Layer ###
layer_flat = create_flatten_layer(drop_out_layer2)

### FC layer 1 ###
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

### FC layer 2 ###
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

### Softmax layer ###
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)

### Training parameters (loss, LR, optimizer) ###
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 

### Show progress - train and validation accuracy ###
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

SOURCE_FOLDER = '/content/drive/My Drive/Asgn1'

#tf.reset_default_graph()

### Class definitions ###
classdict = {'Checked': 0, 'Colourblock': 1, 'Melange': 2, 'Patterned': 3, 'Printed': 4, 'abstract': 5, 
             'embroidery': 6, 'floral': 7, 'graphic': 8, 'polka dots': 9, 'solid': 10, 'striped': 11, 'typography': 12}
classes = 13
saver = tf.train.Saver()
img_files = []

### Train and Validation dir ###
tf_train_path = "/content/drive/My Drive/Asgn1/training_data/"
tf_val_path = "/content/drive/My Drive/Asgn1/validation_data/"

### Train Function ### 
def train(epoch):
    
    for i in range(epoch):

        for root, dirs, files in os.walk(tf_train_path):
            for file in files:
              img_files.append(os.path.join(root, file))
        random.shuffle(img_files)
        
        for root, dirs, files in os.walk(tf_val_path):
            for file in files:
              val_img_files.append(os.path.join(root, file))
        random.shuffle(val_img_files)
        
        batch_size = 16
        start=0
        valstart=0

		### Set iteration for Current epoch ### 
        for j in range(int(len(img_files)/batch_size)):
            
			batch_files = img_files[start:start+batch_size]
            #print batch_files
            
   			### Get train image tensors for each batch ###        
            images = []
            labels = []
            for i23 in range(len(batch_files)):

                IMAGE_PATH = batch_files[i23]
                image = cv2.imread(IMAGE_PATH)
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                
                dir1 = os.path.dirname(batch_files[1])
                classname = os.path.basename(dir1)
                
                label = np.zeros(len(classes))
                label[classdict[classname]] = 1.0
                labels.append(label)
                #cls.append(fields)
            
            if (valstart>len(val_img_files)):
                valstart = 0

            ### Get validation batch ###  
            valbatch_files = val_img_files[valstart:valstart+batch_size]
            valimages = []
            vallabels = []
            for i23 in range(len(valbatch_files)):

                IMAGE_PATH = valbatch_files[i23]
                image = cv2.imread(IMAGE_PATH)
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                
                dir1 = os.path.dirname(valbatch_files[1])
                classname = os.path.basename(dir1)
                
                vallabel = np.zeros(len(classes))
                vallabel[classdict[classname]] = 1.0
                vallabels.append(vallabel)
                #cls.append(fields)

            x_batch, y_true_batch = images, labels
            x_valid_batch, y_valid_batch = valimages, vallabels


            feed_dict_tr = {x: x_batch,
                               y_true: y_true_batch, keep_prob : 0.85}
            feed_dict_val = {x: x_valid_batch,
                                  y_true: y_valid_batch, keep_prob : 1.0}

            session.run(optimizer, feed_dict=feed_dict_tr)
            
            start = start +batch_size
            valstart = valstart +batch_size
            
         
        val_loss = session.run(cost, feed_dict=feed_dict_val)
        epoch = int(i / int(data.train.num_examples/batch_size))    

        show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
        saver.save(session, os.path.join(SOURCE_FOLDER ,'Fynd_data_model4.ckpt'),global_step=500 )

train(epoch = 20)

