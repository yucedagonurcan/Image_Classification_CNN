#Building the Convolutional Neural Network

#Importing the libraries that we will use later

#For making the initializing the neural network.
from keras.models import Sequential 
#This is the package for flatten the layers that will come from the Pooling step and sends to the neural networks as input.
from keras.layers import Flatten 
#For pooling step. It will add pooling layers.
from keras.layers import MaxPooling2D 
#For the convolution step. 2D for images.
from keras.layers import Conv2D 
#For add the fully connected layers.
from keras.layers import Dense 
#For using dropout.
from keras.layers import Dropout 


def create_model(drop_out_size,input_shape = (32, 32, 3)):
    
    #Initialising the CNN
    classifier = Sequential()
    
    #Step-1 Convolution 
    
    #Conv2D = (number of feature detector, rows,coloumns)
    #input_shape = Shape of the input images for convolution step as input
    #activation = activation function 
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    
    #Step-2 Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    #We are adding a second convolutional layer and send to pooling
    #We don't need to specify the input_shape because we already defined before in our first convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    
    #Step- 3 Flattening
    classifier.add(Flatten())
    
    #Step - 4 Full Connection
    
    #Fully connected layer
    classifier.add(Dense(units= 128, activation= 'relu'))
    classifier.add(Dropout(p = 0.1))
    #Output layer
    classifier.add(Dense(units= 1, activation= 'sigmoid'))
    
    #Compiling the CNN
    classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
    return classifier


def run_training(batch_sz = 32,epochs = 10):
    #Fitting the CNN to the images
    from keras.preprocessing.image import ImageDataGenerator
    
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
                                        
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=batch_sz,
                                                    class_mode='binary')
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=batch_sz,
                                                class_mode='binary')
    
    import time
    t0 = time.clock()
    classifier = create_model(drop_out_size= 0.6, input_shape=(150, 150, 3))
    classifier.fit_generator(training_set,
                        steps_per_epoch=8000/batch_sz,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=2000/batch_sz)
    
    print ((time.clock()-t0)/60, "minutes process time")

def main():
    run_training(batch_sz=32, epochs=100)
    #Make a new prediction.
    import numpy as np
    from keras.preprocessing import image
    #Load the image
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
    #Add one more coloumn for representing the RGB values as our input shape.
    test_image = image.img_to_array(test_image)
    #We are adding one more dimension for create a batch information for predicting, we have to create a batch even if we have 1 data to predict.
    #Add the next dimension in the the first index = 0.
    test_image = np.expand_dims(test_image, axis = 0)
    #Predicting
    result = classifier.predict(test_image)
    #Class names
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    
""" Main """
if __name__ == "__main__":
    main()
              
    
    
    