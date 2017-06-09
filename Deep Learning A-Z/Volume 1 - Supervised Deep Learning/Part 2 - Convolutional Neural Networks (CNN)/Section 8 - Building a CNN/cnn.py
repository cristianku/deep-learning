from keras.models import  Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Initializing CNN

classifier = Sequential()


# Step 1 - Convolution
classifier.add(Convolution2D(32,3,3))

