# apperal_classification
Apperal pattern classification using tensorflow and keras 

Data augmentation can be done using scripts: 'Data_augmentation1_scaling.py' for center scaling images at different scales and 'Data_augmentation1_flip.py' for horizontal flip of images.

Model training in Keras is done using keras_train.py. Model is defined in model definition section where we can configure image size, # layers, strides, normalization, dropout, optimizer. Training imgages and validatin images directories should be updated in data pipeline section. To save trained Model, update model name in training section.

Prediction on new images is done by keras_test.py. Update test_dir with directory of test images and change output csv name at the end. 

Keras model trained on 16k images for 30 epochs can be downloaded from model_file.txt (google drive share link). 

Final_keras_fifth_model2_result.xlsx contains class predictions for test images.

Similarly, a model could be trained in tensorflow using tensorflow_train.py and prediction using tensorflow_test.py.

Solution approach is mentioned in Product Pattern Classification Solution Final.docx.
