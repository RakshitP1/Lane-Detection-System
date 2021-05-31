#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Conv2DTranspose, UpSampling2D

import pickle
import numpy as np
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import glob

#from ..ImageProcessing.ImageReader import ImageReader



class Model:

    def __init__(self, layers = [
        # BatchNormalization(input_shape=(480, 640, 3)),
        # #Dense(units=6, activation='relu'),
        # #Dense(units=6, activation='relu'),
        # Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'),

        # # # Conv Layer 2
        # Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'),
        # Dropout(0.2),
        # Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'),
        
        # Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'),
        # Dropout(0.2),

        # # # Deconv 2
        # Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'),
        # Dropout(0.2),

        # # # Upsample 2
        # # UpSampling2D(size=(2,2)),


        # # # Deconv 3
        # Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'sigmoid', name = 'Deconv3'),
        # # Dropout(0.2),

        BatchNormalization(input_shape=(480, 640, 3)),

        Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'),

        
        Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'),
        Dropout(0.2),


        
        Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'),

       
        Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'),
        
        
        #UpSampling2D(size=(2,2)),

        
        Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'),
        

        
        Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'),
        Dropout(0.2),
        
        Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'),

        
        Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4')
        
        
        ]
        #Dense(units=4, activation='sigmoid')]
        ):
        
        self._model = Sequential(layers)





    def get_model(self):
        return self._model

    def train_model(self, batch_size = 7, epochs = 10, input_shape = (480, 640, 3), training_image_path='training_images.p', labeled_image_path='labeled_images.p'):
        x_train, x_val, y_train, y_val = self.get_training_and_validation_data(training_image_path, labeled_image_path)

        self._model.compile(optimizer='Adam', loss='mean_squared_error')
        self._model.fit(
            x_train, y_train, 
            batch_size=batch_size, 
            steps_per_epoch=len(x_train)/batch_size, 
            epochs=epochs, 
            validation_data=(x_val, y_val), 
        )
        self._model.trainable = False
        self._model.save('saved_model.h5')
         # Show summary of model
        self._model.summary()



    def get_training_and_validation_data(self, training_image_path, labeled_image_path):
        training_images = []
        labeled_images = []
        #pickled images
        if str(training_image_path)[-2] == ".p" and str(labeled_image_path)[-2] == ".p":
            pickled_training_data = open(training_image_path, 'rb')
            training_images = pickle.load(pickled_training_data)

            pickled_labeled_data = open(labeled_image_path, 'rb')
            labeled_images = pickle.load(pickled_labeled_data)
            
            training_images = np.array(training_images)
            labeled_images = np.array(labeled_images)

        #Image Folder
        if "." not in str(training_image_path) and "." not in str(labeled_image_path):
            training_image_file_list = glob.glob(training_image_path+'/*.png')
            
            
            
            for image_file in training_image_file_list:
                print(image_file, cv2.imread(image_file).shape)
                training_images.append(np.array(cv2.imread(image_file)))
            
            print("---------------------------------")
            labeled_image_file_list = glob.glob(labeled_image_path+'/*.png')
            for image_file in labeled_image_file_list:
                print(image_file, cv2.imread(image_file).shape)
                labeled_images.append(np.array(cv2.imread(image_file)))

        training_images = np.array(training_images)
        labeled_images = np.array(labeled_images)

        for i in range(len(training_images)):
            training_images[i] = self.resize_image(training_images[i])

        for i in range(len(labeled_images)):
            labeled_images[i] = self.resize_image(labeled_images[i])

        normalized_labeled_images = labeled_images / 255

        # Shuffle images along with their labels, then split into training/validation sets
        training_images, normalized_labeled_images = shuffle(training_images, normalized_labeled_images)
        #print("Train Images/Labels: ", training_images, normalized_labeled_images)
        # Test size may be 10% or 20%
        x_train, x_val, y_train, y_val = train_test_split(training_images, normalized_labeled_images, test_size=0.1)
        #print("x_train, x_val, y_train, y_val: ", x_train, x_val, y_train, y_val)

        return (x_train, x_val, y_train, y_val)
    
        
    def resize_image(self, image):
        # Get image ready for feeding into model
        resized_image = cv2.resize(np.array(image, np.float32), (640, 480), interpolation = cv2.INTER_AREA)
        resized_image = np.array(resized_image)
        #resized_image = resized_image[None,:,:,:]
        return resized_image





def main():

    
    cnn_model = Model()

    cnn_model.train_model(epochs=1,input_shape=(80, 160, 3), training_image_path='training_images/', labeled_image_path='labeled_images/')
    
    # Freeze layers since training is done
    cnn_model.trainable = False
    # cnn_model.compile(optimizer='Adam', loss='mean_squared_error')

    # Save model architecture and weights
    # cnn_model.save('cnn_model.h5')

    # # Show summary of model
    # cnn_model.summary()
    
    # cnn_model.save()

if __name__ == "__main__":
    main()
    