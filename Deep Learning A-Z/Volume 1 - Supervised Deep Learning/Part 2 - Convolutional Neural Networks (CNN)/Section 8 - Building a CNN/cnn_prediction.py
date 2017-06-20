import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
# print (test_image)
#
# print (" **************** ")
# print (" **************** ")
# print (" **************** ")
# print (" **************** ")
test_image = np.expand_dims(test_image, axis = 0) # add new dimension, corresponding to the batches
# print (test_image)


from keras.models import model_from_json

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


result = loaded_model.predict(test_image)
resultx =  result[0][0]
print (resultx)
if (resultx > 0.8 ):
     prediction = 'DOG'
else:
     prediction = 'CAT'

print (" for image cat_or_dog_1 , result = " + prediction)