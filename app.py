import os
import time
import urllib.request
import wget
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from predmodel import *
import tensorflow as tf
import tensorflow.keras.backend as K

def iou_coef(y_true, y_pred, smooth=1e-1):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    smooth = tf.cast(smooth, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

app = Flask(__name__)
app.secret_key= "secret key"
app.config['UPLOAD_FOLDER'] = "static"
ALLOWED_EXTENSIONS = set(['png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image_submit.png'))
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are png')
            return redirect(request.url)
            
@app.route('/verif', methods=['POST'])
def verif():
    # flash(f'répertoire courant : {os.getcwd()}')
    # filepath = os.path.join(os.getcwd(),'static\image_submit.png')
    # listdir = os.listdir(os.path.join(os.getcwd(),'static'))
    # flash(f'dossier racine : {str(listdir)}')
    # isFile = os.path.isfile(filepath)
    # flash(f'Fichier trouvé : {isFile}')
    ##
    ### Loading the model
    model_url = "https://github.com/QuDbo/segmentation-demo/releases/download/demo/variables.data-00000-of-00001?raw=true"
    local_path = "model/unet_vgg16_v3_256_2Aug/variables/"
    wget.download(model_url, local_path)
    
    model = tf.keras.models.load_model("./model/unet_vgg16_v3_256_2Aug/",
                                                      custom_objects={"iou_coef":iou_coef}
                                                     )
    ##
    t0 = time.time()
    ##
    make_predict(model)
    ##
    inference_time = round(time.time() - t0,2)
    return render_template('verif.html', inference_time=inference_time)
        
if __name__ == "__main__":
    app.run()