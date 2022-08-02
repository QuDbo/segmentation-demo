import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tfImage
import cv2
import os
import re
import json
import albumentations as albu


class ImageGenerator(tf.keras.utils.Sequence):
    '''
    Class to create a image generator :
    - return images from path
    - return images labels from label path (mapping is made to return 8 classes)
    - return images by batch
    - resize image if needed
    - generate (multiple) augmentation(s) of the data if needed with albumentations lib
    
    path_img : path of town directory where images are stored, will detect .png,
    path_label : same but with label, result are available as 
                - png file with color for label
                - json file with polygon
                see mask_method for method selection
    mapping_dic : mapping dictionnary to get 8 classes from subclasses,
                  should be coherent with mask_method
    batch_size : size of batch of image read by the generator 
                (don't include augmented data), default 32,
    inital_dimension : dimension for the images to be read, (1024X2048) default,
    final_dimension : dimension for the images and masks to be return, (256,512) default,
    shuffle : To update index after each epoch,
    augmentation : bool, apply data augmentation if True, default True,
    n_aug : Number of augmentation to be done per image, default 1,
    crop_dimension : dimension for the images and masks to be cropped, (512,1024) default,
    mask_method : To choose between image or json method to read the image masks.
    one_hot : booelan, indicate if the output is in :
                - one hot encoding style (None , height, width, len(mapping_dict))
                - grayscale with level of gray in range(0, len(mapping_dict))
            Mostly used to display example. During training, one_hot should be True
    The number of image return by the generator (augmented data include) is :
        batch_size*n_aug if augmentation=True else batch_size
    
    Inspired by Afshine Amidi and Shervine Amidi (https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
    '''
    
    def __init__(self, path_img, path_label, mapping_dic,
                 initial_dimension=(1024,2048),
                 final_dimension=(256,512),
                 batch_size=32,
                 shuffle=False,
                 mask_method="image",
                 augmentation=True,
                 n_aug=1,
                 crop_dimension=(512,1024),
                 one_hot = True
                ):
        # Variables
        self.initial_dimension = initial_dimension
        self.final_dimension = final_dimension
        self.crop_dimension = crop_dimension
        self.batch_size = batch_size
        self.path_img = path_img
        self.path_label = path_label
        self.mapping_dic = mapping_dic
        self.augmentation = augmentation
        if (self.augmentation):
            self.n_aug=n_aug
        else:
            self.n_aug=0
        self.inverse_map_dic = {
            v:k for k,l_keys in self.mapping_dic.items() for v in l_keys 
        }
        self.shuffle = shuffle
        self.mask_method=mask_method
        self.one_hot = one_hot
        
        # Init treatment
        self.list_IDs = self.get_list_IDs()
        self.on_epoch_end()
    
    def get_list_IDs(self):
        '''
        go through all town directory in path to find the list of png file and 
        return the path of all files as a list
        '''
        town_index = {town:[] for town in os.listdir(self.path_img)}
        for town,town_list in town_index.items() :
            subpath = f"{self.path_img}/{town}/"
            town_rep_list = os.listdir(subpath)
            for name_img in town_rep_list:
                id_img = re.search("\d{6}_\d{6}",name_img)
                try:
                    id_img = id_img.group()
                    town_list+=[id_img]
                except AttributeError as attrib_err:
                    pass
                except:
                    raise
        list_IDs = [f"{self.path_img}/{town}/{town}_{id_img}" 
                  for town,town_list in town_index.items()
                  for id_img in town_list
                 ]
        return list_IDs
                    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data (return image and label on initial_dimension shape)
        X, y = self.__data_generation(list_IDs_temp)

        # Resizing to final dimension
        X_final, y_final = self.__data_resizing(X,y)
        
        # Data augmentation (return cropped image an label on crop_dimension shape) 
        if (self.augmentation):
            for k in range(self.n_aug):
                X_aug, y_aug = self.__data_augmentation(X,y)
                # Resize data (return image to final_dimension shape)
                X_aug, y_aug = self.__data_resizing(X_aug,y_aug) 
                # Concatenate to X and y
                X_final = np.concatenate((X_final,X_aug),axis=0)
                y_final = np.concatenate((y_final,y_aug),axis=0)
        
        # Formatting the y output
        if (self.one_hot):
            return X_final, y_final
        else:
            new_y = np.zeros((self.batch_size*(self.n_aug+1),
                              *self.final_dimension),
                             dtype="uint8")
            for k in range(len(self.mapping_dic)):
                new_y = new_y + k*y_final[:,:,:,k]
            return X_final, new_y
    
    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self,list_IDs_temp):
        '''
        Generates data corresponding to batch samples
        '''
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.initial_dimension,
                      3),
                     dtype="uint8",
                    )
        y = np.zeros((self.batch_size, *self.initial_dimension,
                      len(self.mapping_dic)),
                     dtype="uint8",
                    )

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Read the image
            path_img = f"{ID}_leftImg8bit.png"
            image = tfImage.load_img(path_img,
                                     color_mode="rgb",
                                     target_size=self.initial_dimension,
                                     interpolation="nearest",
                                    )
            image = tfImage.img_to_array(image,
                                         dtype="uint8",
                                        )
            X[i,] = image
            
            # Read mask and create a 8 channel image from it
            mask = y[i,]
            ID = ID.replace('leftImg8bit','gtFine')
            if self.mask_method=="json":
                path_label = f"{ID}_gtFine_polygons.json"
                mask = self.recreate_mask_from_json(path_label,
                                                    mask,
                                                   )
            else :
                path_label = f"{ID}_gtFine_labelIds.png"
                mask = self.recreate_mask_from_image(path_label,
                                                     mask,
                                                    )
            y[i,] = mask
            
        return X, y
    
    def recreate_mask_from_json(self,path,mask):
        json_dict = json.load(open(path,'r'))
        for segm_dict in json_dict["objects"]:
            sub_cat = segm_dict['label']
            vertices = segm_dict['polygon']
            id_mask = self.inverse_map_dic[sub_cat]
            vertices = np.array([[d[0],d[1]] for d in vertices],
                                dtype='int32'
                               )
            sub_mask = np.array(mask[:,:,id_mask])
            cv2.fillConvexPoly(sub_mask,vertices,1)
            mask[:,:,id_mask] = sub_mask
        return mask
    
    def recreate_mask_from_image(self,path,mask):
        image = tfImage.load_img(path,
                                 color_mode="grayscale",
                                 target_size=self.initial_dimension,
                                 interpolation="nearest",
                                )
        image = tfImage.img_to_array(image,
                                     dtype="uint8",
                                    )
        image = image.reshape(self.initial_dimension)
        for id_mask in range(len(self.mapping_dic)):
            bool_mask = np.full(self.initial_dimension, False)
            for sub_id in self.mapping_dic[id_mask]:
                sub_bool_mask = (image==sub_id)
                bool_mask = np.logical_or(bool_mask,sub_bool_mask)
            mask[:,:,id_mask] = bool_mask.astype(int)
        return mask
    
    def __data_augmentation(self,X,y):
        '''
        Generates data augmentation on images and masks labels
        '''
        
        # Initialization
        X_aug = np.empty((self.batch_size,
                          *self.crop_dimension,
                          3),
                         dtype="uint8",                        
                        )
        y_aug = np.zeros((self.batch_size,
                          *self.crop_dimension,
                          len(self.mapping_dic)),
                         dtype="uint8",
                        )
        
        # Data augmentation
        transform = albu.Compose([
            albu.RandomCrop(width=self.crop_dimension[1],
                            height=self.crop_dimension[0],
                           ),
            albu.HorizontalFlip(p=0.8),
            albu.Rotate(limit=22.5,
                        interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_REFLECT,
                        p=0.4),
            albu.RandomBrightnessContrast(p=0.8),
            albu.Blur(blur_limit=[5,5],p=0.7),
        ])
        
        for i,(image,masks) in enumerate(zip(X,y)):
            list_masks = [masks[:,:,i] for i in range(len(self.mapping_dic))]
            transformed = transform(image=image, masks=list_masks)
            image_aug = transformed['image']
            masks_aug = transformed['masks']
            X_aug[i,] = image_aug
            masks_aug = np.asarray(masks_aug)
            masks_aug = np.moveaxis(masks_aug, 0, -1)
            y_aug[i,] = masks_aug
        return X_aug, y_aug

    def __data_resizing(self,X,y):
        '''
        Resize images to final_dimension
        '''
        # Initialization
        X_final = np.empty((self.batch_size,
                            *self.final_dimension,
                            3),
                         dtype="uint8",                        
                        )
        y_final = np.zeros((self.batch_size,
                            *self.final_dimension,
                            len(self.mapping_dic)),
                         dtype="uint8",
                        )
        # Resizing  
        for i,(image,masks) in enumerate(zip(X,y)):
            image_final = cv2.resize(image,
                                     dsize=self.final_dimension[::-1],
                                     interpolation=cv2.INTER_NEAREST,
                                    )
            masks_final = cv2.resize(masks,
                                   dsize=self.final_dimension[::-1],
                                   interpolation=cv2.INTER_NEAREST,
                                  )
            X_final[i,] = image_final
            y_final[i,] = masks_final
            
        return X_final, y_final

    
    ###########################
    ## Fonction for examples ##
    ###########################
    
    def data_example(self,display_image=True,display_mask=True):
        '''
        Generates an example and display it
        '''

        # Generate index
        rng = np.random.default_rng()
        rand_int = rng.integers(0,len(self.list_IDs),size=self.batch_size,endpoint=True)
        list_IDs_temp = [self.list_IDs[k] for k in rand_int]
        
        X, y = self.__data_generation(list_IDs_temp)
        X_final, y_final = self.__data_resizing(X,y)
        
        if (self.augmentation):
            for k in range(self.n_aug):
                X_aug, y_aug = self.__data_augmentation(X,y)
                X_aug, y_aug = self.__data_resizing(X_aug,y_aug) 
                X_final = np.concatenate((X_final,X_aug),axis=0)
                y_final = np.concatenate((y_final,y_aug),axis=0)          
        
        # Formatting the y output
        if (self.one_hot):
            pass
        else:
            new_y = np.zeros((self.batch_size*(self.n_aug+1),
                              *self.final_dimension),
                             dtype="uint8")
            for k in range(len(self.mapping_dic)):
                new_y = new_y + k*y_final[:,:,:,k]
            y_final = new_y
                             
        for i in range(self.batch_size):
            if (display_image):
                image = X_final[i]
                masks = y_final[i]
                cv2.imshow("Resized original image", image[:,:,::-1])
                cv2.waitKey(0)
                for j in range(self.n_aug):
                    image_aug = X_final[i+(j+1)*self.batch_size,]
                    cv2.imshow(f"Augmented image {j+1}", image_aug[:,:,::-1])
                    cv2.waitKey(0)
                    if (display_mask):
                        masks_aug = y_final[i+(j+1)*self.batch_size,] 
                        if (self.one_hot):   
                            for k in range(len(self.mapping_dic)):
                                cv2.imshow(f"resized mask {k}", masks[:,:,k]*255)
                                cv2.imshow(f"resized masks_aug {j+1}-{k}", masks_aug[:,:,k]*255)
                                cv2.waitKey(0)
                                cv2.destroyWindow(f"resized mask {k}")
                                cv2.destroyWindow(f"resized masks_aug {j+1}-{k}")
                        else :
                            multip = 255//len(self.mapping_dic)
                            cv2.imshow(f"resized mask", masks[:,:]*multip)
                            cv2.imshow(f"resized masks_aug {j+1}", masks_aug[:,:]*multip)
                            cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        return X_final, y_final