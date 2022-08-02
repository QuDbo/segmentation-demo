import os
import time
import urllib.request
import wget
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from predmodel import *
import tensorflow as tf
import tensorflow.keras.backend as K
from os.path import exists

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
    local_path = "model/unet_vgg16_v3_256_2Aug/variables/"
    model_url = "https://github.com/QuDbo/segmentation-demo/releases/download/demo/variables.data-00000-of-00001?raw=true"
    
    file_exists = exists(local_path+"variables.data-00000-of-00001")
    if not(file_exists):
        wget.download(model_url, local_path)
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
    ### Loading the model
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

@app.route('/examples', methods=['POST'])
def examples():
    if request.method == 'POST':
        name_list = ['frankfurt_000000_001016','lindau_000000_000019','munster_000046_000019']
        return render_template("examples.html",
                                server_list=name_list)

@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        option = request.form.get('option', '')
        ### Loading the model
        model = tf.keras.models.load_model("./model/unet_vgg16_v3_256_2Aug/",
                                                          custom_objects={"iou_coef":iou_coef}
                                                         )
        ##
        id_originale = "static/examples/"+option+"_leftImg8bit.png"
        id_masks = "static/examples/"+option+"_mask_colors.png"
        id_retrieved = "static/results/"+option+"_retrieved.png"
        ##
        t0 = time.time()
        ##
        make_predict_example(model,id_originale)
        ##
        inference_time = round(time.time() - t0,2)
        return render_template('results.html',
                                id_originale=id_originale,
                                id_masks=id_masks,
                                id_retrieved=id_retrieved,
                                inference_time=inference_time)
        
if __name__ == "__main__":
    app.run()