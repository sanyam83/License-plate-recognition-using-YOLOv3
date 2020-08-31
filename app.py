import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from flask import Flask, redirect, url_for, request, render_template
import cv2
import keras
import itertools
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPool2D, Dense,MaxPooling2D,Bidirectional
from keras.layers import AveragePooling2D, Flatten, Activation
from keras.layers import BatchNormalization, Dropout
from keras.layers import Concatenate, Add, Multiply, Lambda
from keras.layers import UpSampling2D, Reshape
from keras.layers.merge import add,concatenate
from keras.layers import Reshape
from keras.models import Model
from keras.layers.recurrent import LSTM,GRU
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


def words_from_labels(labels):
    """
    converts the list of encoded integer labels to word strings like eg. [12,10,29] returns CAT 
    """
    letters= '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    txt=[]
    for ele in labels:
        if ele == len(letters): # CTC blank space
            txt.append("")
        else:
            #print(letters[ele])
            txt.append(letters[ele])
    return "".join(txt)

def decode_label(out):
    """
    Takes the predicted ouput matrix from the Model and returns the output text for the image
    """
    # out : (1, 48, 37)
    out_best = list(np.argmax(out[0,2:], axis=1))

    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

    outstr=words_from_labels(out_best)
    return outstr
  
def test_data_single_image_Prediction(test_img_path):
    """
    Takes the best model, test data image paths, test data groud truth labels and pre-processes the input image to 
    appropriate format for the model prediction, takes the predicted output matrix and uses best path decoding to 
    generate predicted text and prints the Predicted Text Label, Time Taken for Computation
    """
    
    test_img=cv2.imread(test_img_path)
    test_img_resized=cv2.resize(test_img,(170,32))
    test_image=test_img_resized[:,:,1]
    test_image=test_image.T 
    test_image=np.expand_dims(test_image,axis=-1)
    test_image=np.expand_dims(test_image, axis=0)
    test_image=test_image/255
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        model_output=model.predict(test_image)
    return model_output

# define a Flask app
app = Flask(__name__)
# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
# model=model_create()
set_session(sess)
model=load_model('model_weights.h5')
graph = tf.get_default_graph()

print('Successfully loaded weights model...')
print('Visit http://127.0.0.1:5000')

def model_predict(img_path):
    '''
        helper method to process an uploaded image
    '''
    preds=test_data_single_image_Prediction(img_path)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']        
        
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        
        # make prediction about this image's class
        preds = model_predict(file_path)
        
        predicted_output=decode_label(preds)
        result=predicted_output
        # result = str(pred_class[0][0][1])
        # print('[PREDICTED CLASSES]: {}'.format(pred_class))
        print('[RESULT]: {}'.format(result))
        
        return result
    
    return None


if __name__ == '__main__':
    app.run(debug=True)