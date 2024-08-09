import cv2
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
#graph = tf.get_default_graph()

from flask import Flask, redirect, url_for, request
classifier = load_model('model.h5')
labels = ['aga_khan_palace','shaniwar_wada']
UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/getImageLabel',methods = ['POST'])
def getImageLabel():
    reqId = request.form['id']
    imagefile = request.files.get('image')
    path = os.path.join(app.config['UPLOAD_FOLDER'], str(reqId)+'.jpg')
    imagefile.save(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    img = np.reshape(img, [1, 64, 64, 3])
    classes = classifier.predict_classes(img)
    print(classes)
    return {'id':reqId,'label':labels[classes[0]]}

if __name__ == '__main__':
   app.run(host='192.168.1.104', port=5000)

