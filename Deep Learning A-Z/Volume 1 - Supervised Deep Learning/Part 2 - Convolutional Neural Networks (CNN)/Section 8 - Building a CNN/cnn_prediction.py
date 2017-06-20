
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

## reloading classifier from Json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



#########
test_image = image.load_img('cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image cat_or_dog_1 , result = " + prediction)
#############
#############
#############
#############
test_image = image.load_img('dog2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image dog2 , result = " + prediction)

#############
test_image = image.load_img('dog3.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image dog3 , result = " + prediction)

#############
test_image = image.load_img('dog4.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx = result[0][0]
print(resultx)
if (resultx > 0.8):
    prediction = 'DOG'
else:
    prediction = 'CAT'

print(" for image dog4 , result = " + prediction)


#############
test_image = image.load_img('cat1.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image cat1 , result = " + prediction)


#############
test_image = image.load_img('cat2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image cat2 , result = " + prediction)


#############
test_image = image.load_img('cat3.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches

result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image cat3 , result = " + prediction)