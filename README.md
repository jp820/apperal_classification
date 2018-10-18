# apperal_classification
Apperal classification using tensorflow and keras 

Data augmentation can be done using scripts: Data_augmentation1_scaling for center scaling images at different scales and Data_augmentation1_flip for horizontal flip of images.

A model can be trained in Keras using keras_train.py. Model is defined in model definition section where image size, # layers, strides, normalization, dropout, optimizer can be configured. Training imgages and validatin images directories can be updated in data pipeline section. Model can be saved by model name in training section.

Prediction on new images is done by keras_test.py. Update test_dir with directory of test images and change output csv name at the end. 

Keras model trained on 16k images for 30 epochs can downloaded from model_file.txt (google drive share link). 

Results on test images can be found in Final_keras_fifth_model2_result.xlsx.

Similarly a model can be trained in tensorflow using tensorflow_train.py and prediction using tensorflow_test.py.
