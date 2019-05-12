import os
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2

def extract_samples(samples):
    with open('../data/driving_log_processed.csv') as csvfile: 
        reader = csv.reader(csvfile)
        next(reader, None) #this is necessary to skip the first record as it contains the headings
        for line in reader:
            #line[0]=line[0].split('\\')[-1]
            #line[1]=line[0].split('\\')[-1]
            #line[2]=line[0].split('\\')[-1]
            samples.append(line)
            #print (line)

# function that preprocess the dataset
def preprocess_samples(batch_samples):
    images = []
    angles = []
    for batch_sample in batch_samples:
        for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
            #print(batch_sample[i].split('/')[-1])
            #print(batch_sample[i])
            name = '../data/IMG/'+batch_sample[i].split('/')[-1]
            # converting to RGB where drive.py is using RGB
            #print("######", name)
            center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
            #center_image = cv2.imread(name)
            center_angle = float(batch_sample[3]) #getting the steering angle measurement
            images.append(center_image)

            # adjust angles for left and right images
            if(i==0):
                angles.append(center_angle)
            elif(i==1):
                angles.append(center_angle+0.2)
            elif(i==2):
                angles.append(center_angle-0.2)

            # flip the image for more dataset and measurement
            images.append(cv2.flip(center_image,1))
            if(i==0):
                angles.append(center_angle*-1)
            elif(i==1):
                angles.append((center_angle+0.2)*-1)
            elif(i==2):
                angles.append((center_angle-0.2)*-1) 
    return images, angles

#Generator for batch by batch training/validation
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while True: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images, angles = preprocess_samples(batch_samples)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield X_train, y_train
            
# build and return the NN model
def build_model():   
    from keras.models import Sequential
    from keras.layers.core import Dense, Flatten, Activation, Dropout
    from keras.layers.convolutional import Conv2D
    from keras.layers import Lambda, Cropping2D

    model = Sequential()

    # Normalize the data, 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # reduce the image size
    model.add(Cropping2D(cropping=((60,30),(0,0))))           

    #layer 1- Convolution, no of filters- 24, filter size= 5x5, strides= 2x2
    model.add(Conv2D(24,(5,5),strides=(2,2)))
    model.add(Activation('relu'))

    #layer 2- Convolution, no of filters- 36, filter size= 5x5, strides= 2x2
    model.add(Conv2D(36,(5,5),strides=(2,2)))
    model.add(Activation('relu'))

    #layer 3- Convolution, no of filters- 48, filter size= 5x5, strides= 2x2
    model.add(Conv2D(48,(5,5),strides=(2,2)))
    model.add(Activation('relu'))

    #layer 4- Convolution, no of filters- 64, filter size= 3x3, default stride = 1x1
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))

    #layer 5- Convolution, no of filters- 64, filter size= 3x3, default stride = 1x1
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))

    #flatten image
    model.add(Flatten())

    #layer 6- dense and activation
    model.add(Dense(100))
    model.add(Activation('relu'))

    # dropout to resolve overfitting
    model.add(Dropout(0.25))

    #layer 7- another dense and activation function
    model.add(Dense(50))
    model.add(Activation('relu'))

    #layer 8- fanother dense and activation function
    model.add(Dense(10))
    model.add(Activation('relu'))

    #layer 9- fully connected layer 1
    model.add(Dense(1))

    # the output
    # mean square error is used as loss function
    model.compile(loss='mse',optimizer='adam')

    return model

def main():
    BATCH_SIZE=64
    EPOCH = 5
    
    # data set container
    samples = []
    print("Extract dataset")
    extract_samples(samples)
    
    print("split dataset")
    # split the dataset where test_size is percentage of validation set
    my_train_data, my_validation_data = train_test_split(samples,test_size=0.15) 
    
    model = build_model()
    model.fit_generator(generator(my_train_data, BATCH_SIZE), 
                        samples_per_epoch= len(my_train_data), 
                        validation_data = generator(my_validation_data, BATCH_SIZE),   
                        nb_val_samples=len(my_validation_data),
                        nb_epoch=EPOCH, 
                        verbose=1)

    model.save('model.h5')
    print('Model is saved')
    model.summary()

main()