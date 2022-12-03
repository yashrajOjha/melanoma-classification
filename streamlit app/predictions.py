from numpy import expand_dims
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
model = tf.keras.models.load_model('vggmodel.h5')
# model.summary()

test_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')

def tta_prediction(image,n_examples,model=model,datagen=test_datagen):
    # convert image into dataset
    samples = expand_dims(image, 0)
    # prepare iterator
    it = datagen.flow(samples, batch_size=n_examples)
    # make predictions for each augmented image
    probs = model.predict_generator(it, steps=n_examples, verbose=0)
    #print(len(probs))    
    prob = np.mean(probs, axis=1)    
    return prob

def load_image(file):
    im = Image.open(file)
    img = np.array(im)
    # img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    val = tta_prediction(res,1)
    if val[0]>0.5: return True
    else: return False
