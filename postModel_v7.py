import tensorflow as tf
import tensorflow.keras.preprocessing.image as imgtf
from imgGen_v4 import *
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import IoU
import tensorflow.keras.backend as K
import sys


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
    
def compare_N_example(N,generator,model):
    '''
    Apply N times "compare_random_image_masks"
    '''
    X, y = generator.data_example(display_image=False,display_mask=False)
    if model is None:
        y_pred=None
    else:
        y_pred = model.predict(X)
    
    for i in range(N):
        compare_random_image_masks(X,y,y_pred)

def mask_to_argmax_to_mask(masks):
    '''
    Convert a array of mask probability in an array of mask
    '''
    
    id_mask = np.argmax(masks,axis=-1)
    l_masks = {}
    for k in range(8):
        recreate_mask = lambda x:1 if x==k else 0
        l_masks[k] = np.vectorize(recreate_mask)(id_mask)
    final_masks = np.stack(tuple((l_masks[k] for k in range(8))),axis=-1)
    return final_masks

def dice_coef(y_true, y_pred, smooth=1e-1):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    smooth = tf.cast(smooth, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

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


def compute_mesure_round(generator,model):
    '''
    Compute several metrics on data from generator
    '''
    accu = tf.keras.metrics.CategoricalAccuracy()
    
    mesure_argmax_iou = {k:[] for k in range(8)}
    mesure_argmax_iou['all']=[]
    mesure_accu = {k:[] for k in range(8)}
    mesure_accu['all']=[]
    mesure_argmax_dice = {k:[] for k in range(8)}
    mesure_argmax_dice['all']=[]
    
    e=0
    e_tot = generator.__len__()
    for X,y in generator :
        e_c = int((e+1)/e_tot*50)
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'='*e_c}{'-'*(50-e_c)}]")
        sys.stdout.flush()
        for i in range(X.shape[0]):
            X1 = X[i,]
            X1 = X1[np.newaxis,:]
            y1 = y[i,]
            y1 = y1[np.newaxis,:]
            y_pred = model.predict(X1,verbose=0)
            y_pred_round = mask_to_argmax_to_mask(y_pred)
    
            # IoU mesure
            for k in range(8):
                mesure_argmax_iou[k] += [iou_coef(y1[...,k],y_pred_round[...,k])]
            mesure_argmax_iou['all'] += [iou_coef(y1,y_pred_round)]
            # Accuracy mesure
            for k in range(8):
                accu.update_state(y1[...,k],y_pred_round[...,k])
                mesure_accu[k] += [accu.result().numpy()]
                accu.reset_state()
            accu.update_state(y1,y_pred_round)
            mesure_accu['all'] += [accu.result().numpy()]
            accu.reset_state()
            # Dice mesure
            for k in range(8):
                mesure_argmax_dice[k] += [dice_coef(y1[...,k],y_pred_round[...,k])]
            mesure_argmax_dice['all'] += [dice_coef(y1,y_pred_round)]
        e+=1
    
    print("\nIoU mesure")
    for k in range(8):
        str2print = f"IoU from mask {k} : {np.mean(mesure_argmax_iou[k]):.2f}"
        print(str2print)
    str2print = f"IoU from all masks : {np.mean(mesure_argmax_iou['all']):.2f}"
    print(str2print)

    print("\nAccuracy mesure:")
    for k in range(8):
        str2print = f"Categorical accuracy from mask {k} : {np.mean(mesure_accu[k]):.2f}"
        print(str2print)
    str2print = f"Categorical accuracy from all masks : {np.mean(mesure_accu['all']):.2f}"
    print(str2print)

    print("\nDice mesure:")
    for k in range(8):
        str2print = f"Dice from mask {k} : {np.mean(mesure_argmax_dice[k]):.2f}"
        print(str2print)
    str2print = f"Dice from all masks : {np.mean(mesure_argmax_dice['all']):.2f}"
    print(str2print)

def compute_mesure(generator,model):
    '''
    Compute several metrics on data from generator
    '''
    accu = tf.keras.metrics.CategoricalAccuracy()
    
    mesure_argmax_iou = {k:[] for k in range(8)}
    mesure_argmax_iou['all']=[]
    mesure_accu = {k:[] for k in range(8)}
    mesure_accu['all']=[]
    mesure_argmax_dice = {k:[] for k in range(8)}
    mesure_argmax_dice['all']=[]
    
    e=0
    e_tot = generator.__len__()
    for X,y in generator :
        e_c = int((e+1)/e_tot*50)
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'='*e_c}{'-'*(50-e_c)}]")
        sys.stdout.flush()
        for i in range(X.shape[0]):
            X1 = X[i,]
            X1 = X1[np.newaxis,:]
            y1 = y[i,]
            y1 = y1[np.newaxis,:]
            y_pred = model.predict(X1,verbose=0)
    
            # IoU mesure
            for k in range(8):
                mesure_argmax_iou[k] += [iou_coef(y1[...,k],y_pred[...,k])]
            mesure_argmax_iou['all'] += [iou_coef(y1,y_pred)]
            # Accuracy mesure
            for k in range(8):
                accu.update_state(y1[...,k],y_pred[...,k])
                mesure_accu[k] += [accu.result().numpy()]
                accu.reset_state()
            accu.update_state(y1,y_pred)
            mesure_accu['all'] += [accu.result().numpy()]
            accu.reset_state()
            # Dice mesure
            for k in range(8):
                mesure_argmax_dice[k] += [dice_coef(y1[...,k],y_pred[...,k])]
            mesure_argmax_dice['all'] += [dice_coef(y1,y_pred)]
        e+=1
    
    print("\nIoU mesure")
    for k in range(8):
        str2print = f"IoU from mask {k} : {np.mean(mesure_argmax_iou[k]):.2f}"
        print(str2print)
    str2print = f"IoU from all masks : {np.mean(mesure_argmax_iou['all']):.2f}"
    print(str2print)

    print("\nAccuracy mesure:")
    for k in range(8):
        str2print = f"Categorical accuracy from mask {k} : {np.mean(mesure_accu[k]):.2f}"
        print(str2print)
    str2print = f"Categorical accuracy from all masks : {np.mean(mesure_accu['all']):.2f}"
    print(str2print)

    print("\nDice mesure:")
    for k in range(8):
        str2print = f"Dice from mask {k} : {np.mean(mesure_argmax_dice[k]):.2f}"
        print(str2print)
    str2print = f"Dice from all masks : {np.mean(mesure_argmax_dice['all']):.2f}"
    print(str2print)
    
def compute_mesure_batch(generator,model):
    '''
    Compute several metrics on data from generator
    '''
    accu = tf.keras.metrics.CategoricalAccuracy()
    
    mesure_argmax_iou = {k:[] for k in range(8)}
    mesure_argmax_iou['all']=[]
    mesure_accu = {k:[] for k in range(8)}
    mesure_accu['all']=[]
    mesure_argmax_dice = {k:[] for k in range(8)}
    mesure_argmax_dice['all']=[]
    
    e=0
    e_tot = generator.__len__()
    for X,y in generator :
        e_c = int((e+1)/e_tot*50)
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'='*e_c}{'-'*(50-e_c)}]")
        sys.stdout.flush()
        y_pred = model.predict(X,verbose=0)

        # IoU mesure
        for k in range(8):
            mesure_argmax_iou[k] += [iou_coef(y[...,k],y_pred[...,k])]
        mesure_argmax_iou['all'] += [iou_coef(y,y_pred)]
        # Accuracy mesure
        for k in range(8):
            accu.update_state(y[...,k],y_pred[...,k])
            mesure_accu[k] += [accu.result().numpy()]
            accu.reset_state()
        accu.update_state(y,y_pred)
        mesure_accu['all'] += [accu.result().numpy()]
        accu.reset_state()
        # Dice mesure
        for k in range(8):
            mesure_argmax_dice[k] += [dice_coef(y[...,k],y_pred[...,k])]
        mesure_argmax_dice['all'] += [dice_coef(y,y_pred)]
        e+=1
    
    print("\nIoU mesure")
    for k in range(8):
        str2print = f"IoU from mask {k} : {np.mean(mesure_argmax_iou[k]):.2f}"
        print(str2print)
    str2print = f"IoU from all masks : {np.mean(mesure_argmax_iou['all']):.2f}"
    print(str2print)

    print("\nAccuracy mesure:")
    for k in range(8):
        str2print = f"Categorical accuracy from mask {k} : {np.mean(mesure_accu[k]):.2f}"
        print(str2print)
    str2print = f"Categorical accuracy from all masks : {np.mean(mesure_accu['all']):.2f}"
    print(str2print)

    print("\nDice mesure:")
    for k in range(8):
        str2print = f"Dice from mask {k} : {np.mean(mesure_argmax_dice[k]):.2f}"
        print(str2print)
    str2print = f"Dice from all masks : {np.mean(mesure_argmax_dice['all']):.2f}"
    print(str2print)