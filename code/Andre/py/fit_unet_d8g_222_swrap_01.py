"""
Created on Thu Jan 26 17:04:11 2017

@author: Andre Stochniol, andre@stochniol.com

Fit unet style nodule identifier on Luna databaset using 8-grid scheme
Physical resolution 2x2x2mm
Data aggregated, shuffled; wrap augmentation used (swrap)

"""

import numpy as np 

from keras.models import load_model,Model
from keras.layers import MaxPooling3D
from keras.layers import Convolution3D
from keras.layers import Input, merge, UpSampling3D
from keras.optimizers import Adam

from keras import backend as K

#from keras.preprocessing.image import ImageDataGenerator  # Keras original
from image_as_mod3d_2dmask import ImageDataGenerator   # our modified version


K.set_image_dim_ordering('th')   

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

DICE_LOW_LIMIT = 0 ## was 0.001
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    y_pred_f [y_pred_f < DICE_LOW_LIMIT] = 0.
    y_pred_f [y_pred_f > 1- DICE_LOW_LIMIT] = 1.
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
def dice_coef_pos_np(y_true, y_pred, pos = 0):
    y_true_f = y_true[:,pos].flatten()
    y_pred_f = y_pred[:,pos].flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_model_xd3_2_6l_grid(nb_filter=48, dim=5, clen=3 , img_rows=224, img_cols=224 ):   # NOTE that this procedure is/should be used with img_rows & img_cols as None
    
    # aiming for architecture similar to the http://cs231n.stanford.edu/reports2016/317_Report.pdf
    # Our model is six layers deep, consisting  of  a  series  of  three  CONV-RELU-POOL  layyers (with 32, 32, and 64 3x3 filters), a CONV-RELU layer (with 128 3x3 filters), three UPSCALE-CONV-RELU lay- ers (with 64, 32, and 32 3x3 filters), and a final 1x1 CONV- SIGMOID layer to output pixel-level predictions. Its struc- ture resembles Figure 2, though with the number of pixels, filters, and levels as described here

    ## 3D CNN version of a previously developed unet_model_xd_6j 
    zconv = clen
    
    inputs = Input((1, dim, img_rows, img_cols))
    conv1 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)


    conv4 = Convolution3D(4*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(pool2)
    conv4 = Convolution3D(4*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv4)


    up6 = merge([UpSampling3D(size=(2, 2, 2))(conv4), conv2], mode='concat', concat_axis=1)
    conv6 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(up6)
    conv6 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv6)
    
    
    up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv1], mode='concat', concat_axis=1)  # original - only works for even dim 
    conv7 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(up7)
    conv7 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv7)


    pool11 = MaxPooling3D(pool_size=(2, 1, 1))(conv7)

    conv12 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(pool11)
    conv12 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(conv12)
    pool12 = MaxPooling3D(pool_size=(2, 1, 1))(conv12)

    conv13 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(pool12)
    conv13 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(conv13)
    pool13 = MaxPooling3D(pool_size=(2, 1, 1))(conv13)

    if (dim < 16):
        conv8 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(pool13)
    else:   # need one extra layer to get to 1D x 2D mask ...
            conv14 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(pool13)
            conv14 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(conv14)
            pool14 = MaxPooling3D(pool_size=(2, 1, 1))(conv14)
            conv8 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(pool14)        

    model = Model(input=inputs, output=conv8)


    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),  loss=dice_coef_loss, metrics=[dice_coef])

    return model

def grid_data(source, grid=32, crop=16, expand=12):
    height = source.shape[3]  # should be 224 for our data, when used in the initial fix-size mode
    width = source.shape[4]
    
    gridheight = (height - 2 * crop) // grid  # should be 6 for our data
    gridwidth = (width - 2 * crop) // grid
    cells = []
    for j in range(gridheight):
        for i in range (gridwidth):
            cell = source[:,:,:, crop+j*grid-expand:crop+(j+1)*grid+expand, crop+i*grid-expand:crop+(i+1)*grid+expand]
            cells.append(cell)
    
    cells = np.vstack (cells)

    return cells, gridwidth, gridheight


def data_from_grid (cells, gridwidth, gridheight, grid=32):
    
    width = cells.shape[4]
    crop = (width - grid ) // 2 ## for simplicity we are assuming the same crop (and grid) in x & y directions 
       
    if crop > 0:  # do NOT crop with 0 as we get empty cells ...
        cells = cells[:,:,:,crop:-crop,crop:-crop]     
    
    shape = cells.shape
    new_shape_1_dim = shape[0]// (gridwidth * gridheight)  # ws // 36 -- Improved on 20170306
    new_shape = (gridwidth * gridheight, new_shape_1_dim, ) +  tuple([x for x in shape][1:])   # was 36,  Improved on 20170306
    cells = np.reshape(cells, new_shape)  
    cells = np.moveaxis(cells, 0, -3)
    
    shape = cells.shape
    new_shape2 = tuple([x for x in shape[0:3]]) + (gridheight, gridwidth,) + tuple([x for x in shape[4:]])
    cells = np.reshape(cells, new_shape2)
    cells = cells.swapaxes(-2, -3)
    shape = cells.shape
    combine_shape =tuple([x for x in shape[0:3]]) + (shape[-4]*shape[-3], shape[-2]*shape[-1],)
    cells = np.reshape(cells, combine_shape)
    
    return cells
    
    
def data_from_grid_by_proximity (cells, gridwidth, gridheight, grid=32):
    
    # disperse the sequential dats into layers and then use data_from_grid
    shape = cells.shape
    new_shape_1_dim = shape[0]// (gridwidth * gridheight)  # ws // 36 -- Improved on 20170306
    
    
    ### NOTE tha we invert the order of shapes below to get the required proximity type ordering
    new_shape = (new_shape_1_dim, gridwidth * gridheight,  ) +  tuple([x for x in shape][1:])   # was 36,  Improved on 20170306
    #new_shape = (gridwidth * gridheight, new_shape_1_dim, ) +  tuple([x for x in shape][1:])   # was 36,  Improved on 20170306
    
    # swap ordering of axes 
    cells = np.reshape(cells, new_shape) 
    cells = cells.swapaxes(0, 1)
    cells = np.reshape(cells, shape) 
    
    cells = data_from_grid (cells, gridwidth, gridheight, grid)
    
    return cells

def load_aggregate_masks_scans (masks_mnames, grids, upgrid_multis):
    
    scans = []
    masks = []
    
    igrid = 0
    for masks_names in masks_mnames:
        if (len(masks_names) > 0):      
            grid = grids[igrid]
            upgrid_multi = upgrid_multis[igrid]
            upgcount = upgrid_multi * upgrid_multi
            
            scans1 = []
            masks1 = []
            for masks_name in masks_names:
                print ("Loading: ", masks_name)
                masks0 =  np.load(''.join((masks_name, ".npz")))['arr_0']
                scans0 = np.load(''.join((masks_name.replace("masks_", "scans_", 1), ".npz")))['arr_0']
                masks1.append(masks0)
                scans1.append(scans0)
           
            scans1 = np.vstack(scans1)
            masks1 = np.vstack(masks1)
            if len(masks) > 0:
                scans1 = np.vstack([scans1, scans])
                masks1 = np.vstack([masks1, masks])
            
            lm = len(masks1) // upgcount * upgcount  
            scans1 = scans1[0:lm] # cut to multiples of upgcount
            masks1 = masks1[0:lm]
            index_shuf = np.arange(lm)
            np.random.shuffle(index_shuf)
            scans1 = scans1[index_shuf]
            masks1 = masks1[index_shuf]
            
            scans = data_from_grid_by_proximity(scans1, upgrid_multi, upgrid_multi, grid=grid)
            masks = data_from_grid_by_proximity(masks1, upgrid_multi, upgrid_multi, grid=grid)
        
        igrid += 1
        
    return masks, scans    

def load_aggregate_masks_scans_downsample2 (masks_mnames, grids, upgrid_multis, down_base):

    scans = []
    masks = []
    down_size = 50000
    igrid = 0
    for masks_names in masks_mnames:
        if (len(masks_names) > 0):
            grid = grids[igrid]
            upgrid_multi = upgrid_multis[igrid]
            upgcount = upgrid_multi * upgrid_multi

            scans1 = []
            masks1 = []
            for masks_name in masks_names:
                print ("Loading: ", masks_name)
                masks0 =  np.load(''.join((masks_name, ".npz")))['arr_0']
                scans0 = np.load(''.join((masks_name.replace("masks_", "scans_", 1), ".npz")))['arr_0']

                if igrid >= 0:
                     
                    down = down_base * (4 ** igrid)  #  dynamic
                    if len(masks0) > down_size and down > 1:

                        print("Down-sampling masks0/scans0 by: ", masks_name, down)
                        lm = len(masks0)
                        index_shuf = np.arange(lm)
                        np.random.shuffle(index_shuf)
                        scans0 = scans0[index_shuf]
                        masks0 = masks0[index_shuf]
                        masks0 = masks0[0:len(masks0):down]
                        scans0 = scans0[0:len(scans0):down]

                masks1.append(masks0)
                scans1.append(scans0)

            scans1 = np.vstack(scans1)
            masks1 = np.vstack(masks1)
            if len(masks) > 0:
                scans1 = np.vstack([scans1, scans])
                masks1 = np.vstack([masks1, masks])

            lm = len(masks1) // upgcount * upgcount
            scans1 = scans1[0:lm] # cut to multiples of upgcount
            masks1 = masks1[0:lm]
            index_shuf = np.arange(lm)
            np.random.shuffle(index_shuf)
            scans1 = scans1[index_shuf]
            masks1 = masks1[index_shuf]

            scans = data_from_grid_by_proximity(scans1, upgrid_multi, upgrid_multi, grid=grid)
            masks = data_from_grid_by_proximity(masks1, upgrid_multi, upgrid_multi, grid=grid)

        igrid += 1

    return masks, scans



if __name__ == '__main__':
                             
    # Key initial parameters        
    dim = 8  
    start_from_scratch = True
    load_initial_weights = False
    if start_from_scratch and load_initial_weights:
        model_weights_name_to_start_from = "../luna/models/d8_2x2x2_best_weights.h5"  # only used when start_from_scratch is True and load_initial_weights is True
    
    
    ### KEY running parameteres 
    nb_epoch =  25
    
    model_load_name = 'UNUSED../luna/models/d8g4a_model_25.h5'
    model_save_name = '../luna/models/d8g4a_model_25.h5'  ### MUST include "_model" string as we use this for a substituion for weights file
           
    seed = 1000 # should be varied by steps/stages
    downsample = 20
    set_lr_value = False
    new_lr_value = 1e-5  # only used when set_lr_value is True


    use_large_validation = True
    grids = [20, 40]
    upgrid_multis = [2, 2]  # we modify only the last one if/as needed

    batch_size = 7 * int((8 // upgrid_multis[1])**2)  # calculated for a 12GB graphics card (such as Tesla K80/AWS P2 system)
    
    masks_mnames = [
        [     
                    #"../luna/models/masks_d8g1x20ba4a_2x2x2_nodules_0_3_6860",
                    "../luna/models/masks_d8g1x20ba4a_2x2x2_nodules_4_8_8178",

                    #"../luna/models/masks_d8g1x20ba4a_2x2x2_blanks_0_3_68442",
                    "../luna/models/masks_d8g1x20ba4a_2x2x2_blanks_4_8_97406"
        ],
        [     

                    "../luna/models/masks_d8g1x40ba4a_2x2x2_nodules_0_3_5940",
                    #"../luna/models/masks_d8g1x40ba4a_2x2x2_nodules_4_8_6925",

                    #"../luna/models/masks_d8g1x40ba4a_2x2x2_blanks_0_3_52367",   ## unblock this one 
                    #"../luna/models/masks_d8g1x40ba4a_2x2x2_blanks_4_8_74880"
        ]]
        
    masks_val_mnames = [
        [     
                "../luna/models/masks_d8g1x20ba4a_2x2x2_nodules_9_9_1442"
        ],
        [   
                "../luna/models/masks_d8g1x40ba4a_2x2x2_nodules_9_9_1101"        
        ]]


    masks_val_large_mnames = [
        [     
                "../luna/models/masks_d8g1x20ba4a_2x2x2_nodules_9_9_1442",
                "../luna/models/masks_d8g1x20ba4a_2x2x2_blanks_9_9_19861"
        ],
        [     
                "../luna/models/masks_d8g1x40ba4a_2x2x2_nodules_9_9_1101",
                #"../luna/models/masks_d8g1x40ba4a_2x2x2_blanks_9_9_15122"
        ]]

    np.random.seed(seed) 
    masks, scans =  load_aggregate_masks_scans_downsample2 (masks_mnames, grids, upgrid_multis, downsample)
    print ("Masks and Scans shapes: ", masks.shape, scans.shape)
    masks[masks < 0] = 0   # just in case (eliminate the blanks's marking)

    if masks.shape[2] > 1:
        masks = masks[:,:,masks.shape[2] // 2]    ## select the central value as this one contains still all data
        masks = masks[:, np.newaxis]
    print ("Masks shape after 2D mapping: ", masks.shape)

    masks_val, scans_val = load_aggregate_masks_scans (masks_val_mnames, grids, upgrid_multis)
    print ("Val Masks and Scans shapes: ", masks_val.shape, scans_val.shape)
    
    masks_val[masks_val < 0] = 0
    if masks_val.shape[2] > 1:                
        masks_val = masks_val[:,:,masks_val.shape[2] // 2]    ## select the central value as this one contains still all data
        masks_val = masks_val[:, np.newaxis]
    print ("Masks_val shape after 2D mapping: ", masks_val.shape)
    
    masks_val_large, scans_val_large = load_aggregate_masks_scans (masks_val_large_mnames, grids, upgrid_multis)
    print ("Large Val Masks and Scans shapes: ", masks_val_large.shape, scans_val_large.shape)

    masks_val_large[masks_val_large < 0] = 0
    if masks_val_large.shape[2] > 1:                
        masks_val_large = masks_val_large[:,:,masks_val_large.shape[2] // 2]    ## select the central value as this one contains still all data
        masks_val_large = masks_val_large[:, np.newaxis]
    print ("Large Val Masks shape after 2D mapping: ", masks_val_large.shape)

    if start_from_scratch:        
        model = unet_model_xd3_2_6l_grid(nb_filter=20, dim=dim, clen=3, img_rows=None , img_cols=None )  
        print(model.summary())        

        if load_initial_weights:
            model_weights_name = model_weights_name_to_start_from  ### could potentially load best weights
            model.load_weights(model_weights_name)
            print("Weights and output models: ", model_weights_name, model_save_name) 
        else:
            print("Start from scratch (no weights),output models: ", model_save_name) 

    else:
        ## load_previous_model
        print ("Loading model: ", model_load_name)
        model = load_model(model_load_name,          #3                                         
               custom_objects={'dice_coef_loss': dice_coef_loss,
                                   'dice_coef': dice_coef
                                   }
                )
        #print(model.summary())        
        print("Load and output models: ", model_load_name, model_save_name) 
         
    ## set the data ...
    masks = masks.astype(np.int16)
    
    final_couple_of_iterations = False
    if final_couple_of_iterations:
        masks = np.concatenate((masks, masks_val))
        scans = np.concatenate((scans, scans_val))
 
    data_gen_args = dict(featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0,   
        width_shift_range=0.055/2,
        height_shift_range=0.055/2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode=  "wrap",
        zoom_range=0
        )
        
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    shuffle = True # default
    image_datagen.fit(scans, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    
    image_generator = image_datagen.flow(scans,
        batch_size = batch_size,
        #shuffle = shuffle,
        seed=seed)
    
    mask_generator = mask_datagen.flow(masks,
        batch_size = batch_size, 
        #shuffle = shuffle,
        seed=seed)
    
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    
    if set_lr_value:         
        print("Model learning rate (old): ", model.optimizer.lr.get_value())   # was 1e-4
        model.optimizer.lr.set_value(new_lr_value)
        print("Model learning rate(new): ", model.optimizer.lr.get_value())        
    
    samples_per_epoch = masks.shape[0] 
    model.fit_generator(
        train_generator,
        samples_per_epoch= samples_per_epoch,
        nb_epoch = nb_epoch,
        validation_data = ( scans_val, masks_val),
        verbose=1)


    model.save(model_save_name)
    model.save_weights(model_save_name.replace("_model", "_weights", 1))


    masks_pred = model.predict(scans_val, verbose=1)
    
    dice_check = dice_coef_np(masks_val, masks_pred) 
    print ("dice_check: ", dice_check)

    if use_large_validation:

          masks_pred_large = model.predict(scans_val_large, batch_size =1, verbose=1)
          dice_check = dice_coef_np(masks_val_large, masks_pred_large) 
          print ("Full dice_check: ", dice_check)
        
    print("Model learning rate: ", model.optimizer.lr.get_value())

