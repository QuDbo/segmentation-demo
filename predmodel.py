import tensorflow as tf
import tensorflow.keras.preprocessing.image as imgtf
import numpy as np
from postModel_v7 import *

def convert_mask_to_color(masks):
    '''
    Convert a array of mask (height,width,8) in color visualisation array (height,width,3)
    '''
    dic_color = {
        0 : [76,0,153], # flat
        1 : [0,0,204], # vehicle
        2 : [96,96,96], # construction
        3 : [224,224,224], # object
        4 : [0,204,0], # nature
        5 : [255,0,0], # human
        6 : [153,255,255], # sky
        7 : [0,0,0] # void
    }
    dic_r = {k:dic_color[k][0] for k in dic_color.keys()}
    dic_g = {k:dic_color[k][1] for k in dic_color.keys()}
    dic_b = {k:dic_color[k][2] for k in dic_color.keys()}
    
    id_mask = np.argmax(masks,axis=-1)
    color_masks_r = np.vectorize(dic_r.__getitem__)(id_mask)
    color_masks_g = np.vectorize(dic_g.__getitem__)(id_mask)
    color_masks_b= np.vectorize(dic_b.__getitem__)(id_mask)
    color_masks = np.stack((color_masks_r,color_masks_g,color_masks_b),axis=-1)
    return color_masks

def make_predict():
    '''
    Load the image store as static/image_submit.png and convert to an array
    Load the model and make a prediction
    Convert the masks array into a image and store it
    '''
    
    ### Loading the image
    path_img = "static/image_submit.png"
    dimension = (256, 256)
    X = np.empty((1, *dimension, 3), dtype="uint8")
    image = imgtf.load_img(path_img,
    color_mode="rgb",
    target_size=dimension, # Depends of the model trained
    interpolation="nearest",
    )
    X[0,] = image
    
    ### Loading the model
    trained_unet_scratch = tf.keras.models.load_model("model/unet_vgg16_v3_256_2Aug/",
                                                      custom_objects={"iou_coef":iou_coef}
                                                     )
    
    ### Prediction
    y_pred = trained_unet_scratch.predict(X)
    masks = y_pred[0,]
    image_masks = convert_mask_to_color(masks)
    image_to_save = imgtf.array_to_img(image_masks)
    path_to_save = "static/masks_retrieved.png"
    tf.keras.utils.save_img(path_to_save, image_to_save,
                            data_format="channels_last")
    ###
    

def compare_random_image_masks(X,y,y_pred):
    '''
    Select a random image in the generator, predict the masks with the model
    Plot side by side the image, the true masks, the predict masks 
    '''
    
    rng = np.random.default_rng()
    rand_int = rng.integers(0,y.shape[0],size=1,endpoint=False)
    rand_int=rand_int[0]
    
    y_color = convert_mask_to_color(y[rand_int,])
    if y_pred is not None:
        y_test_color = convert_mask_to_color(y_pred[rand_int,])
    
    fig,ax = plt.subplots(1,3,figsize=(3*5,5))
    ax[0].imshow(imgtf.array_to_img(X[rand_int,:,:]))
    ax[0].axis('off')
    ax[0].set_title('Original image')
    ax[1].imshow(imgtf.array_to_img(y_color))
    ax[1].axis('off')
    ax[1].set_title('Original masks')
    if y_pred is not None:
        ax[2].imshow(imgtf.array_to_img(y_test_color))
        ax[2].set_title('Retrieved masks')
    ax[2].axis('off')
    plt.show()