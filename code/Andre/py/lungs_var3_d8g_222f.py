"""
Created on Thu Jan 26 17:04:11 2017

@author: Andre Stochniol, andre@stochniol.com
"""

import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import os
import time

import pandas as pd

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage # added for scaling\

from keras.models import load_model,Model
from keras.layers import Input, merge, UpSampling3D

from keras.layers import Convolution3D, MaxPooling3D # added for 3D

from keras.optimizers import Adam
from keras import backend as K

from scipy import stats

from scipy.stats import gmean
import gc


RESIZE_SPACING = [2,2,2]
RESOLUTION_STR = "2x2x2"

ALT_WORKSTATION = ""  # "_shared"  # could be _shared on one of our clusters (empty on AWS)

STAGE_DIR_BASE = ''.join(["../input", ALT_WORKSTATION, "/%s/" ])                        # to be used with % stage
LABELS_BASE = ''.join(["../input", ALT_WORKSTATION, "/%s_labels.csv"])                  # to be used with % stage
#SAMPLE_SUBM_BASE = ''.join(["../input", ALT_WORKSTATION, "/%s_sample_submission.csv"]) # to be used with % stage


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    #slices.sort(key = lambda x: int(x.InstanceNumber))
    #slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))  # from v 8 - BUGGY (should be float caused issues with segmenting and rescaling ....
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))  # from v 8
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_3d_data_slices(slices):  # get data in Hunsfield Units
    #slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    #slices.sort(key=lambda x: int(x.InstanceNumber))  # was x.InstanceNumber
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))  # from v 8
    
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)  # ensure int16 (it may be here uint16 for some images )
    image[image == -2000] = 0   #correcting cyindrical bound entrioes to 0
    
    # Convert to Hounsfield units (HU)
    # The intercept is usually -1024
    for slice_number in range(len(slices)):  # from v 8
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:  # added 16 Jan 2016, evening
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
    
    
def get_3d_data_hu(path):  # get data in Hunsfield Units
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    #slices.sort(key=lambda x: int(x.InstanceNumber))  # was x.InstanceNumber
    #slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))  # from v8 - BUGGY 
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))  # from 22.02
    
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)  # ensure int16 (it may be here uint16 for some images )
    image[image == -2000] = 0   #correcting cyindrical bound entrioes to 0
    
    # Convert to Hounsfield units (HU)
    # The intercept is usually -1024
    for slice_number in range(len(slices)):  # from v 8
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:  # added 16 Jan 2016, evening
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
# image values currently range from -1024 to around 2000. Anything above 400 is not interesting to us, as these are simply bones with different radiodensity. A commonly used set of thresholds, as per https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial# , is to normalize between are -1000 and 400. Here's some code you can use:  
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


PIXEL_MEAN = 0.028  # should be from the entire set (approx value used)

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

#image=sample_image
def segment_lung_mask(image, fill_lung_structures=True):
    
    #image = sample_image #TESTING
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1  # was -320
    
    useTestPlot = False
    if useTestPlot:
        plt.hist(image.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

    # Remove the blobs connected to the border of the image
    
    labels = measure.label(binary_image)
    # np.unique(labels)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    
   
    background_label = labels[0,0,0]    
    #background_label2 = largest_label_volume (binary_image, bg = -5000)   

    background_labels_x = np.unique(np.concatenate((
        np.unique(labels[:, 0,0]),
        np.unique(labels[:,image.shape[1]-1,image.shape[2]-1]),
        np.unique(labels[:,0,image.shape[2]-1]),
        np.unique(labels[:,image.shape[1]-1,0])
        )))    
    
    use_all_edges = False
    if use_all_edges:
        
        background_labels_x = np.unique(np.concatenate((
            np.unique(labels[:,image.shape[1]-1,image.shape[2]-1]),
            np.unique(labels[:,0,image.shape[2]-1]),
            np.unique(labels[:,image.shape[1]-1,0]),
            np.unique(labels[:, 0,0]),
                      
            #np.array([background_label2]),
        
            np.unique(labels[image.shape[0]-1,:,image.shape[2]-1]),
            np.unique(labels[0,:,image.shape[2]-1]),
            np.unique(labels[image.shape[0]-1,:,0]),   
            np.unique(labels[0,:,0]),
          # 
            np.unique(labels[image.shape[0]-1,image.shape[1]-1,:]),
            np.unique(labels[0,image.shape[1]-1,:]),
            np.unique(labels[image.shape[0]-1,0,:]),
            np.unique(labels[0,0,:]) )))
    
    for background_label in background_labels_x: 
        binary_image[background_label == labels] = 2
    #binary_image[background_label == labels] = 2
    #binary_image[background_label2 == labels] = 2
    #binary_image[background_label3 == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)  # was 0
            # print ("i, l_max: ", i, l_max)
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1  

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    removeOther = True  # should be TRue
    if removeOther:    
        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        # np.unique(labels)
        l_max = largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0
 
    return binary_image 

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)   # nor mode= "wrap"/xxx, nor cval=-1024 can ensure that the min and max values are unchanged .... # cval added
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')  ### early orig modified 
    #image = scipy.ndimage.zoom(image, real_resize_factor, order=1)    # order=1 bilinear , preserves the min and max of the image -- pronbably better for us (also faster than spkine/order=2)
    
    #image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest', order=1)    # order=1 bilinear , preserves the min and max of the image -- pronbably better for us (also faster than spkine/order=2)
    
    return image, new_spacing
    

def get_segmented_preprocess_bb_xd_file(file_segmented, by_min, by_max, bx_min, bx_max, dim): # image returned in Hounsfield units
    
    
    lung_img = np.load(file_segmented)['arr_0']
    
    scan = lung_img[:, by_min:by_max, bx_min:bx_max]
    
    scan = normalize(scan)
    scan = zero_center(scan)
    
    ver_with_malloc = True  # this halves the time for this procedure on hp old workstation (from 4 to 2 seconds, roughly)
    if not ver_with_malloc:      ### original version    
        tmp = []
        #i = 0
        #for i in range(1, scan.shape[0]-1, 3):  # SKIP EVERY 3
        for i in range(0, scan.shape[0]-dim):  
            #img1 = scan[i-1]
            #img2 = scan[i]
            #img3 = scan[i+1]
            #rgb = np.stack((img1, img2, img3))
            img_2x_span = np.stack(scan[i:i+dim])
            tmp.append(img_2x_span)    
        #scans1 =  np.stack([val for sublist in work for val in sublist ]) # NO skipping as we have already cut the first and the last layer     
        
        scan =  np.stack(tmp) # now rgb set, including r - previous, g - current, b- next layers
    else:
        shape = scan.shape
        scan = scan.astype(np.float32)
        scan_layered = np.zeros((shape[0]-dim, dim, shape[1], shape[2])).astype(np.float32)
        for i in range(0, scan.shape[0]-dim):
            scan_layered[i] = scan[i:i+dim]
        scan = scan_layered
        

    return scan


def find_roi_bb(path_segmented):
    
    restored = np.load(''.join((path_segmented, ""))) # was, ".npz" --> the file already has the .npz extension added - no need to add it here
    #restored.keys()
    mask = restored['arr_0']
    
    regions = measure.regionprops(mask)  # this measures the largest region and is a bug when the mask is not the largest region !!!

    
    bb = regions[0].bbox
    ymargin = 4         ## add extra margin at the bottom and top
    xmargin = 4         ## add some margin horizontally
    #print(bb)
    #zlen = bb[3] - bb[0]
    ylen = bb[4] - bb[1] + 2 * ymargin
    xlen = bb[5] - bb[2] + 2 * xmargin
    
    shape = mask.shape
    
    # on x make the bounding box symmetric - this is both to reflect the lungs symmetry and avoid segmentation bugs
    crop = np.max(np.min((bb[2]-xmargin, shape[2]-bb[5]-xmargin), 0))
    xlen = shape[2] - 2 * crop     # adjust xlen for a symmetry
    divisor = 4  # ensure that the target dimensions would be divisable by the divisor (4), enlarge to ensure we do not cut out beyond border

    xlen = int(np.round(xlen / divisor) * divisor)
    xlen = np.min((xlen, shape[2]))
    crop = (shape[2] - xlen) //2  # finally calculate the symmetric crop by x
    bx_min = crop
    bx_max = np.min((crop + xlen, shape[2]))  ## just inn case , for the symmetric case is not really neeeded
    
    
    by_min = np.max( (bb[1] - ymargin, 0))
    ylen = int(np.round(ylen / divisor) * divisor)
    ylen = np.min((ylen, shape[1]))   # we assume that the mask.shape x and y dimensions are divisable by the divisor (4)
    by_max = np.min((by_min + ylen, shape[1]))

    
    #dx = 0  # could  be reduced
    ## have to reduce dx as for istance at least image the lungs stretch right to the border evebn without cropping 
    ## namely for '../input/stage1/be57c648eb683a31e8499e278a89c5a0'

    #crop_max_ratio_z = 0.6  # 0.8 is to big    make_submit2(45, 1)
    crop_max_ratio_y = 0.3
    crop_max_ratio_x = 0.6
    
    mask_area = xlen*ylen / (shape[1] * shape[2])
    mask_area_thresh = 0.2  # anything below is too small (maybe just one half of the lung or something very small0)
    mask_area_check =   mask_area >   mask_area_thresh
    
    if (not mask_area_check) or ylen/shape[1] < crop_max_ratio_y or xlen/shape[2]  < crop_max_ratio_x:
        # mask likely wrong leave it untouched    
        ### full crop
        print ("Ignoring - Mask seems too small if following ROI boux were used (by_min, by_max, bx_min. bx_max): ", by_min, by_max, bx_min, bx_max)

        #maskOK = False
        #set the bounding boxes to the maximum
        bx_min = 0
        bx_max = shape[2]
        by_min = 0
        by_max = shape[1]
 
    return by_min, by_max, bx_min, bx_max, mask



K.set_image_dim_ordering('th')   

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def hu_describe(data, uid, part=""):
    
    if len(data) == 0:
        nanid = -7777
        d = {   "vol_%s" % part: nanid,
                "min_%s" % part: nanid,
            "max_%s" % part: nanid,
            "mean_%s" % part: nanid,
            "variance_%s" % part: nanid,
            "skewness_%s" % part: nanid,
            "kurtosis_%s" % part:  nanid
            }
    else:
        desc = stats.describe(data, axis=None, nan_policy='omit')  #default policy is 'propagate'
        #names = ["nobs", "min", "max", "mean", "variance", "skewness", "kurtosis"]
    
        d = {   "vol_%s" % part: desc.nobs,
                "min_%s" % part: desc.minmax[0],
            "max_%s" % part: desc.minmax[1],
            "mean_%s" % part: desc.mean,
            "variance_%s" % part: desc.variance,
            "skewness_%s" % part: desc.skewness,
            "kurtosis_%s" % part:  desc.kurtosis
            }
    #columns = ["id",  "n_volume_%s" % part, "hu_min_%s" % part, "hu_nmax_%s" % part, "hu_mean_%s" % part, "hu_variance_%s" % part,"hu_skewness_%s" % part, "hu_kurtosis_%s" % part]
    #d =       [uid, desc.nobs, desc.minmax[0], desc.minmax[1], desc.mean, desc.variance, desc.skewness, desc.kurtosis]
     
    #columns = sorted(d.keys())
    
    df = pd.DataFrame(d, index=[uid]) 
    
    #df = pd.DataFrame.from_records(d, columns=columns, index=["id"])   
    
    #df.reset_index(level=0, inplace=True)
    
    #df.sort_index(axis=1)
    #df.index.name = "id"
    
    
    #df = pd.DataFrame.from_dict(d, orient='index')
    
    return df



def unet_model_xd3_2_6l_grid(nb_filter=48, dim=5, clen=3 , img_rows=None, img_cols=None ):
    
    #aiming for architecture as in http://cs231n.stanford.edu/reports2016/317_Report.pdf
    #The model is eight layers deep, consisting  of  a  series  of  three  CONV-RELU-POOL  lay- ers (with 32, 32, and 64 3x3 filters), a CONV-RELU layer (with 128 3x3 filters), three UPSCALE-CONV-RELU lay- ers (with 64, 32, and 32 3x3 filters), and a final 1x1 CONV- SIGMOID layer to output pixel-level predictions. Its struc- ture resembles Figure 2, though with the number of pixels, filters, and levels as described here

    ## 3D CNN version of undet_model_xd_6j 
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


def calc_features_keras_3dx(stage, dim, run, processors, model_weights_name):
    
    STAGE_DIR = STAGE_DIR_BASE % stage
    LABELS = LABELS_BASE % stage
    
    start_from_model_weights = True
    if start_from_model_weights:
        model = unet_model_xd3_2_6l_grid(nb_filter=20, dim=dim, clen=3, img_rows=None , img_cols=None )   
       
        model.load_weights(model_weights_name)
        print(model.summary())
        print("Using Weights: ", model_weights_name)
   
    else:
       model_name = "SPECIFY_MODE_NAME_IF_WHEN_NEEDED"   #Warning, hard-wired, modify if/as needed
       model = load_model(model_name, 
                custom_objects={'dice_coef_loss': dice_coef_loss,
                                           'dice_coef': dice_coef
                                           }
                       )
       print(model.summary())
       print("Loaded model: ", model_name)
       
    ## obtain cancer labels for quick validation
    labels = pd.read_csv(LABELS) ### was data/
    #print(labels.head())


    source_data_name = ''.join([stage, "_", RESOLUTION_STR ])
    source_data_name_seg = ''.join([stage, "_segmented_", RESOLUTION_STR ])
    
    stage_dir_segmented = STAGE_DIR.replace(stage, source_data_name, 1) 
    
    files = glob.glob(os.path.join(stage_dir_segmented,'*.npz'))
    
    add_missing_entries_only = False
    if add_missing_entries_only:

        colnames = ["fn"]
        files_in = pd.read_csv("./files_in_222.csv", names=colnames)
        files = files_in.fn.tolist()[1:]
        len(files)
        path = files[2]  # 4 is a cancer, 2 is not but may have a lots of false positives 
        start_file = 1300 #0 #2        # was 508 for a restart
        last_file = len(files)  #
    else:
        start_file = 0 #2        # was 508 for a restart
        last_file = len(files)  #

    path = files[2]  # 4 is a cancer, 2 is not but may have a lots of false positives 
    #path = "../input_shared/stage1_2x2x2/21cfe28c4a891795399bb77d63e939d7.npz"
    count = start_file
    cnt = 0
    frames = []
    
    use_single_pass = True
    if not use_single_pass:
        grid=60  # orig 60
        crop=44  # orig 44
        expand=10  # orig 10
    else:
        grid=360/RESIZE_SPACING[1]  
        crop=44/RESIZE_SPACING[1]  # orig 44
        #expand=0  # orig 10  ---> irrelevant for this option    
    
    for path in files[start_file:last_file]:

        uid =  path[path.rindex('/')+1:] 
        uid = uid[:-4]

        if (uid in labels["id"].tolist()):
            cancer = int(labels["cancer"][labels["id"] == uid])
        else:
            cancer = -777
        
        count += 1
    
        
        if count % processors == run:    # do this part in this process, otherwise skip
        #for folder in mask_wrong:
        
  
            
            start_time = time.time()
            
            path_seg = path.replace(source_data_name, source_data_name_seg, 1)
            if RESIZE_SPACING[1] < 2:   # compatibility setting  for the 2 versions supported *8 and 16 layers)
                by_min, by_max, bx_min, bx_max, mask = find_roi_bb(path_seg)
            else:
                mask = np.load(''.join((path_seg, "")))['arr_0']
                by_min = mask.shape[1] % 4 
                bx_min = mask.shape[2] % 4
                bx_max = mask.shape[2]
                by_max = mask.shape[1]
  
            area_reduction = (by_max-by_min)*(bx_max-bx_min)/(mask.shape[1]*mask.shape[2])
            by_len = by_max - by_min
            bx_len = bx_max - bx_min
            print ("count, cancer, uid, by_min, by_max, bx_min, bx_max, height, width, area_reduction: ", count, cancer, uid, by_min, by_max, bx_min, bx_max, by_max-by_min, bx_max-bx_min, area_reduction)
            print ("geom new size y x: ", by_len, bx_len )
            mask = mask[:, by_min:by_max, bx_min:bx_max]  # unused as yet here - just for testing

            
            divisor = 4
            if (by_len % divisor > 0) or (bx_len % divisor > 0):
                print ("WARNING: for uid, by_len or bx_len not multiple of: ", uid, by_len, bx_len, divisor)
                    
           
            testPlot = False
            if testPlot:
                   # Show some slice in the middle
                plt.imshow(np.mean(mask, axis=0), cmap=plt.cm.gray)
                plt.show()
            #start_time = time.time() 
            images3 = get_segmented_preprocess_bb_xd_file(path, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
            images3 = images3[:,np.newaxis]  # add a dimension for the 3 D model
       
            images3_seg = get_segmented_preprocess_bb_xd_file(path_seg, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
            images3_seg = images3_seg[:,np.newaxis]  # add a dimension for the 3 D model

            
            if not use_single_pass:
                scans, gridwidth, gridheight = grid_data(images3, grid=grid, crop=crop, expand=expand)
            else:
                # juts crop the data
                gridwidth = gridheight = 1
                #scans = images3[:,:,:, crop:-crop,crop:-crop]
                scans = images3

                     
            pmasks =  model.predict(scans, batch_size =1, verbose=0)  # batch_size = 1 seems to be 10% faster than any of the 2, 3, 4, 8 (e.g. 66s vs 73-74 sec)
            #pmasks =  model.predict(scans, verbose=1)
               
               
            if not use_single_pass:
                pmasks3 = data_from_grid (pmasks, gridwidth, gridheight, grid=grid)
            else:
                pmasks3 = pmasks
            
            if use_single_pass:
            # now crop the images3 to the size of pmasks3 for simplicity ...
                nothing_to_do = 1  #  # already cropped through the bb    
            else:
                images3 = images3[:,:,:, crop:-crop,crop:-crop]
           
            path_mpred = path.replace(source_data_name, (source_data_name + "_mpred3_%s") % str(dim), 1)
            pmasks3[pmasks3 < 0.5] =0        # zero out eveything below 0.5 -- this should reduce the file size 
            np.savez_compressed (path_mpred, pmasks3)
            
            bb = pd.DataFrame(
                    {"by_min":  by_min,
                     "by_max":  by_max,
                     "bx_min":  bx_min,
                     "bx_max":  bx_max
                     },
                     index=[uid])
            bb.to_csv( path_mpred[:-4] + ".csv", index=True)
            
            
            read_bb_back = False
            if read_bb_back:
                print ("by_min, by_max, bx_min, bx_max: ", by_min, by_max, bx_min, bx_max)
                bb = pd.read_csv(path_mpred[:-4] + ".csv", index_col = 0) #to check
                ### use uid to ensure errot checking and also consolidation of the bb's if need to
                by_min = bb.loc[uid].by_min
                by_max = bb.loc[uid].by_max
                bx_min = bb.loc[uid].bx_min
                bx_max = bb.loc[uid].bx_max
                                
            
            dim_out = pmasks3.shape[2]
            
            #        reduce the images and pmasks to 2 dimension sets
    
            testPlot = False
            j_pred = dim // 2
            if testPlot:
                i=31  # for tests
                for i in range(images3.shape[0]):
                    j = 0
                    for j in range(j_pred, j_pred+1):  # we use the dim cut for results
                        img = images3[i, 0, j]
                        #pmsk = pmasks3[i,0,j]  # for multi-plane output
                        pmsk = pmasks3[i,0,0]
                        pmsk_max = np.max(pmsk)
                        #print ('scan '+str(i))
                        print ('scan & max_value: ', i, pmsk_max)
                        if pmsk_max > 0.99:
                            f, ax = plt.subplots(1, 2, figsize=(10,5))
                            ax[0].imshow(img,cmap=plt.cm.gray)
                            ax[1].imshow(pmsk,cmap=plt.cm.gray)
                            ####ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                            plt.show()
                            #print ("max =", np.max(pmsk))
            
            #### list solution for the several rolls that also reduce the memory requirements as we only store the relevant cut ...
            
            if dim_out > 1:
                pmaskl = [np.roll(pmasks3, shift, axis =0)[dim:,0,shift] for shift in range(dim)]  # pmask list (skipping the first dim elements)
                pmav = np.min(pmaskl, axis = 0)
                ### also adjust the images3 -- skipp the first dim layers 
                images3 = images3[dim:0]
            else:
                pmav = pmasks3[:,0,0]
   
            
            if testPlot:
                for i in range(pmasks3.shape[0]):
                    j = 0
                    for j in range(j_pred, j_pred+1):
                        #img = images3[i, 0, j]
                        
                        print ('scan '+str(i))
                        f, ax = plt.subplots(1, 2, figsize=(10,5))
                        ax[0].imshow(images3[i, 0, j],cmap=plt.cm.gray)
                        ax[1].imshow(pmav[i],cmap=plt.cm.gray)

                dist = pmav.flatten()
                thresh = 0.005
                dist = dist[(dist > thresh) & (dist < 1-thresh)]
                plt.hist(dist, bins=80, color='c')
                plt.xlabel("Nodules prob")
                plt.ylabel("Frequency")
                plt.show()
                print(len(dist))  #2144 for pmav dist , 3748 for pm , 1038 for pm[:,0,0], 627 for pm[:,0,1], 770 for pm[:,0, 2], 1313 for for pm[:,0, 3]; 1129 for pmav2 - peak around 0.5 where it was not there before
                
            part=0
            frames0 = []
            frames1 = []
            zsel = dim // 2  # use the mid channel/cut (could use any other for the stats and trhey are repeated)
            segment = 2
            for segment in range(3):
                # 0 = top part
                # 1 bottom half
                # 2 all
                if segment == 0:
                    sstart = 0
                    send = images3.shape[0] // 2
                elif segment == 1:
                    sstart = images3.shape[0] // 2
                    send = images3.shape[0]
                else:                   ### the last one must be the entire set
                    sstart = 0
                    send = images3.shape[0]
        

                ims = images3[sstart:send,0,zsel]      # selecting the zsel cut for nodules calc ...
                ims_seg = images3_seg[sstart:send,0,zsel] 
                #pms = pmasks3[sstart:send,0,0]
                pms = pmav[sstart:send]
                
                # threshold the precited nasks ...
                
                #for thresh in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
                for thresh in [0.75, 0.9999999, 0.99999]:
 
                    
                    idx = pms > thresh
                    nodls = np.zeros(pms.shape).astype(np.int16)
                    nodls[idx] = 1
                    
                    nx =  nodls[idx]
                    # volume = np.sum(nodls)  # counted as a count within hu_describe ...
                    nodules_pixels = ims[idx]   # flat
                    nodules_hu = pix_to_hu(nodules_pixels)
                    part_name = ''.join([str(segment), '_', str(thresh)])
                    #df = hu_describe(nodules_hu, uid, 1)
                    df = hu_describe(nodules_hu, uid=uid, part=part_name)
                    
                    ### add any additional params to df ...
                    #sxm = ims * nodls
                    add_projections = False
                    axis = 1
                    nodules_projections = []
                    for axis in range(3):
                         #sxm_projection = np.max(sxm, axis = axis)
                         nodls_projection = np.max(nodls, axis=axis)
                         naxis_name = ''.join(["naxis_", str(axis),"_", part_name])
                         if add_projections:   
                             df[naxis_name] = np.sum(nodls_projection)
                         nodules_projections.append(nodls_projection)
                    
                    ## find the individual nodules ... as per the specified probabilities 
                    labs, labs_num = measure.label(idx, return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
                    if labs_num > 0:                  
                        regprop = measure.regionprops(labs, intensity_image=ims)
                        areas = [rp.area for rp in regprop]
                        #ls = [rp.label for rp in regprop]
                        max_val = np.max(areas)
                        max_index = areas.index(max_val)
                        max_label = regprop[max_index].label
                        #max_ls = ls[max_index]
                        zcenters = [(r.bbox[0]+r.bbox[3])/2  for r in regprop]
                        zweighted = sum(areas[i] * zcenters[i] for i in range(len(areas))) / sum(areas)
                        zrel = zweighted / ims.shape[0]
                        
                        idl = labs ==  regprop[max_index].label   
                        nodules_pixels = ims[idl]
                        nodules_hu = pix_to_hu(nodules_pixels)
                    else:
                        nodules_hu = []
                        zrel = -777
                    part_name = ''.join([str(segment), '_', str(thresh),'_n1'])
                    df2 = hu_describe(nodules_hu, uid=uid, part=part_name) 
                    zrel_name = ''.join(["zrel_", part_name])
                    df2[zrel_name] = zrel
                    count_name = ''.join(["ncount_", str(segment), '_', str(thresh)])
                    df2[count_name] = labs_num
                       
                    df3 = pd.concat( [df, df2], axis =1)
                    
                    frames0.append(df3)
                    part += 1
                
                dfseg = pd.concat(frames0, axis=1, join_axes=[frames0[0].index])  # combine rows
                
                 ##### add any section features independent of nodules ....
                # estimate lung volume by counting all relevant pixels using the segmented lungs data
                # for d16g do it for the 3 different areas, including the emphysema calcs
                
                
                HU_LUNGS_MIN0 = -990  # includes emphysema 
                HP_LUNGS_EMPHYSEMA_THRESH = -950
                HP_LUNGS_EMPHYSEMA_THRESH2 = -970
                HU_LUNGS_MAX = -400
    
                pix_lungs_min = hu_to_pix(HU_LUNGS_MIN0)
                pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)

                idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                idv_emphysema  =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH))  & (ims_seg < pix_lungs_max)
                idv_emphysema2 =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH2)) & (ims_seg < pix_lungs_max)
                
                test_emphysema = False
                if test_emphysema:
                    
                    f, ax = plt.subplots(1, 4, figsize=(20,5))
                    ax[0].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                    ax[1].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                    #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                    ax[2].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                    ax[3].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                    
                    #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                    plt.show()
      
                    
                    idv_all = []
                    hu_lungs_min_all=[]
                    for HU_LUNGS_MIN in range(-1000, -800, 10):
                        pix_lungs_min = hu_to_pix(HU_LUNGS_MIN)
                        pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)

                        idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                        idv = (ims >pix_lungs_min) & (ims < pix_lungs_max)
                        idv_all.append(np.sum(idv))
                        hu_lungs_min_all.append(HU_LUNGS_MIN)
                    e_results = pd.DataFrame(
                        {"HU min": hu_lungs_min_all,
                         "volume approx": np.round(idv_all, 4)
                         })
                    plt.figure()
                    e_results.plot()
                    e_results.plot(kind='barh', x='HU min', y='volume approx', legend=False, figsize=(6, 10))
                        
                        
                lvol = np.sum(idv)
                       
                dfseg[''.join(['lvol', '_', str(segment)])] =  lvol   # 'lvol_2' etc.
                dfseg[''.join(['emphy', '_', str(segment)])] = (lvol - np.sum(idv_emphysema))/(lvol+1)
                dfseg[''.join(['emphy2', '_', str(segment)])] = (lvol - np.sum(idv_emphysema2))/(lvol+1)
        
                frames1.append(dfseg)
                
        
            df = pd.concat(frames1, axis=1, join_axes=[frames1[0].index])  # combine rows
            
            ## use the most recent threshold as a test ...
           
            testPlot2 = True
            #if testPlot2 and ims.shape[0] > ims.shape[1]:        
            if testPlot2:        
                print ("Summary, Count, Cancer, labs_num (nodules count 0), uid, process time: ", count, cancer, labs_num, uid, time.time()-start_time)
 
                if ims.shape[0] > ims.shape[1]:
                    print ("Suspicious hight, shape: ", images3.shape, uid)
                f, ax = plt.subplots(1, 4, figsize=(20,5))
                ax[0].imshow(nodules_projections[0],cmap=plt.cm.gray)
                ax[1].imshow(nodules_projections[1],cmap=plt.cm.gray)
                #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                ax[2].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                ax[3].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                
                #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                plt.show()
  
                
            ### calculate the volume in the central sections (say 60% - 5 sectors)  .-- may be capturing circa 90% of total volume for some (trying to reduce impact of outside/incorect segmetations??)..
            sections = 5
            for sections in range(3,7):       
                zstart = ims_seg.shape[0] // sections
                zstop = (sections -1) * ims_seg.shape[0] // sections
                ims_part = ims_seg[zstart:zstop]
                
                #idv = (ims_part > 0.9 * np.min(ims_part)) & (ims_part < 0.9 * np.max(ims_part))
                idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
          
                #df["lvol_sct"] =  np.sum(idv) 
                df["lvol_sct%s" % sections] =  np.sum(idv) 
                
                #df["lvol"]
                #df["lvol_sct"] /  df["lvol"]
            
            
            df["cancer"] = cancer
    
            
            testPrint = False
            if testPrint:
                dfeasy = np.round(df, 2)
                if cnt == 0:
                    print (dfeasy)
                    #print(dfeasy.to_string(header=True))
                else:
                    print(dfeasy.to_string(header=False))
         
            cnt += 1
            frames.append(df)
            del(images3)
            del(images3_seg)
            del(mask)
            del(nodls)
            del(pmasks3)
            if (cnt % 10 == 0):
                print ("Scans processed: ", cnt)
                gc.collect()  # looks that memory goes up /leakage somwehere
           
          
    result = pd.concat(frames)
    
    return result


def find_lungs_range(y, noise):
    n = len(y)
    mid = n // 2
    
    new_start = 0
    for i in range(mid, 0, -1):
        if y[i] < noise:
            new_start = i
            break
    new_end = n
    for i in range(mid, n, 1):
        if y[i] < noise:
            new_end = i
            break
    return new_start, new_end
  
def cut_out_non_lungs_z (images3, pmasks3, images3_seg, uid, dim):
    HU_LUNGS_MIN = -900  # the algo is sensitive to this value -- keep it 900 unless retested
    HU_LUNGS_MAX = -400
    
    pix_lungs_min = hu_to_pix(HU_LUNGS_MIN)
    pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)

    mid = dim // 2    
    
    ymin = int(0.4 * images3.shape[3])  ## BUG was 4
    ymax = int(0.6 * images3.shape[3])  ## # waut it failed for tne one following 4b351d0c19be183cc880f5af3fe5abee ( index 240 is out of bounds for axis 3 with size 240)
    zmin_new = images3.shape[0] // 2
    zmax_new = images3.shape[0] // 2
    j = ymin
    for j in range(ymin, ymax+1):   
         img_cut = images3[:,0,mid, j]
         img_cut_lungs = (img_cut > pix_lungs_min) & (img_cut < pix_lungs_max)
         lungs_across = np.sum(img_cut_lungs, axis = 1)
         noise_bottom_some = np.mean(lungs_across[0:10])  # increase by 2
         noise = np.max([3*np.min(lungs_across), 0.05 * np.max(lungs_across), noise_bottom_some])  # experimanetal -- could fail is scan has only central part of lungs and no borders at all -- CHECK
         zmin, zmax = find_lungs_range(lungs_across, noise)
         if zmin < zmin_new:
             zmin_new = zmin
         if zmax > zmax_new:
             #print ("j, zmax: ", j, zmax)
             zmax_new = zmax

    ### do not cut it to fine (add few pixels on each side ...)
    zmin_new = np.max([0, zmin_new-mid])
    zmax_new = np.min([images3.shape[0], zmax_new+mid])
    print("cut_out_non_lungs_z from to:", images3.shape[0], zmin_new, zmax_new, uid )
    if ((zmax_new-zmin_new)/images3.shape[0] < 0.5):
            print ("SUSPICSIOUS large cut of > 50%, NOT executing ...")
    else:
        images3 = images3[zmin_new:zmax_new]
        pmasks3 = pmasks3[zmin_new:zmax_new]
        images3_seg = images3_seg[zmin_new:zmax_new]

    return images3, pmasks3, images3_seg
  

def recalc_features_keras_3dx_0313(stage, dim, run, processors, withinsegonly= True, valonly = False):
    STAGE_DIR = STAGE_DIR_BASE % stage
    LABELS = LABELS_BASE % stage

    labels = pd.read_csv(LABELS) ### was data/

    source_data_name = ''.join([stage, "_", RESOLUTION_STR ])
    source_data_name_seg = ''.join([stage, "_segmented_", RESOLUTION_STR ])

    stage_dir_lungs = STAGE_DIR.replace(stage, source_data_name, 1)  # was stage1_segmented_2x2x2
    
    files = glob.glob(os.path.join(stage_dir_lungs,'*.npz'))
    
    limit_to_ids = []            # results of grep Suspicious *.txt    (in the results directory:w)

    # cancer with several nodules: ed0f3c1619b2becec76ba5df66e1ea56 
    # caner with emphysema & ver large nodule (from video): 437e42695e7ad0a1cb834e1b3e780516
    selected = 'ed0f3c1619b2becec76ba5df66e1ea56' # cancer = 1 (huge emphysema ?? 0.26, 0.089 (2)
    
    
    path_template_for_test = '../input_shared/stage1_2x2x2/_to_replace_.npz'
    path = path_template_for_test.replace("_to_replace_", selected, 1)

    if valonly:
        vsdf = pd.read_csv("valset_stratified_1542_0.csv", index_col = 0)
        limit_to_ids = list(vsdf["uid"])
        
        varstr = str(5)
        stratstr = "str"
        val =    np.load(''.join(("../models/val_16g6_th0999_279_%s%s" % (varstr, stratstr), ".npz")))['arr_0']   
        val_uids =    np.load(''.join(("../models/val_uids_16g6_th0999_279_%s%s" % (varstr, stratstr), ".npz")))['arr_0'] 
        val_labels =    np.load(''.join(("../models/val_labels_16g6_th0999_279_%s%s" % (varstr, stratstr), ".npz")))['arr_0'] 
  
    path = files[6]
    start_file = 0      # was 508 for a restart
    count = start_file
 
    last_file = len(files)
    cnt = 0
    frames = []
    
    feats_bb_shape = []
    calculate_only_feats_bb_shape = False  # quick manual hack
    
    for path in files[start_file:last_file]:
        
        uid =  path[path.rindex('/')+1:] 
        uid = uid[:-4]   # cut the .npz suffix
        
  
        if (uid in labels["id"].tolist()):
            cancer = int(labels["cancer"][labels["id"] == uid])  # so we know during testing and also later store it
        else:
            cancer = -777
            
        if uid in limit_to_ids or len(limit_to_ids) == 0:   # either use the limit_to list or use all
            count += 1
            if count % processors == run:    # do this part in this process, otherwise skip
            #for folder in mask_wrong:
            
                start_time = time.time() 
##########################################

                #path_mpred = path.replace("stage1_segmented_2x2x2","stage1_segmented_2x2x2_mpred3_%s" % str(dim), 1)
                path_mpred = path.replace(source_data_name, (source_data_name + "_mpred3_%s") % str(dim), 1)
                path_seg = path.replace(source_data_name, source_data_name_seg, 1)
     
                # read the bouding box saved when mpred was genearted  
                bb = pd.read_csv(path_mpred[:-4] + ".csv", index_col = 0) #to check
                ### use uid to ensure errot checking and also consolidation of the bb's if need to
                by_min = bb.loc[uid].by_min
                by_max = bb.loc[uid].by_max
                bx_min = bb.loc[uid].bx_min
                bx_max = bb.loc[uid].bx_max
                #print ("by_min, by_max, bx_min, bx_max: ", by_min, by_max, bx_min, bx_max)
    
    
                pmasks3 = np.load(''.join((path_mpred, "")))['arr_0']   # the suffix is laready there noe need tp add".npz"
                
                images3 = get_segmented_preprocess_bb_xd_file(path, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
                images3 = images3[:,np.newaxis]  # add a dimension for the 3 D model
           
                images3_seg = get_segmented_preprocess_bb_xd_file(path_seg, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
                images3_seg = images3_seg[:,np.newaxis]  # add a dimension for the 3 D model


               
                dim_out = pmasks3.shape[2]
                #        reduce the images and pmasks to 2 dimension sets
        
                #### NOW cut off the non-lungs areas by the z axis 
 
                bb["zshape"] = images3.shape[0]
                bb["yshape"] = images3.shape[3]
                bb["xshape"] = images3.shape[4]
                do_cut_out_non_lungs_z4 = False  ## manual hack for a while -- needs to be aligned wiht the final code FINAL CODE
                if do_cut_out_non_lungs_z4:
                    images3, pmasks3, images3_seg = cut_out_non_lungs_z (images3, pmasks3, images3_seg, uid, dim)
                    
                if calculate_only_feats_bb_shape:
                    feats_bb_shape.append(bb)
                    continue
                #feats_bb_shape = pd.concat(feats_bb_shape)    
        
                # further constrain within segmented lungs only
                pix_min_001 =    hu_to_pix(MIN_BOUND + 0.001)
                pmasks3_0 = np.array(pmasks3)
                if withinsegonly:    
                    pmasks3[images3_seg[:,:,dim//2:(dim//2+1)] < pix_min_001] = 0    
             
                testPlot = False
                j_pred = dim // 2
                if testPlot:
                    i=31  # for tests
                    for i in range(images3.shape[0]):
                        j = 0
                        for j in range(j_pred, j_pred+1):  # we use the dim cut for results
                            img = images3[i, 0, j]
                            img_seg =  images3_seg[i, 0, j]
                            #pmsk = pmasks3[i,0,j]  # for multi-plane output
                            pmsk = pmasks3[i,0,0]
                            pmsk_max = np.max(pmsk)
                            #print ('scan '+str(i))
                            print ('scan & probability max_value: ', i, pmsk_max)
                            if pmsk_max > 0.9:
                                f, ax = plt.subplots(1, 3, figsize=(12,4))
                                ax[0].imshow(img,cmap=plt.cm.gray)
                                ax[1].imshow(pmsk,cmap=plt.cm.gray)
                                ax[2].imshow(img_seg,cmap=plt.cm.gray)
                                ####ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                                plt.show()
                                #print ("max =", np.max(pmsk))
            
                #### list solution for the several rolls that also reduce the memory requirements as we only store the relevant cut ...
                
                if dim_out > 1:
                    pmaskl = [np.roll(pmasks3, shift, axis =0)[dim:,0,shift] for shift in range(dim)]  # pmask list (skipping the first dim elements)
                    pmav = np.min(pmaskl, axis = 0)
                    
                    pmaskl_0 = [np.roll(pmasks3_0, shift, axis =0)[dim:,0,shift] for shift in range(dim)]  # pmask list (skipping the first dim elements)
                    pmav_0 = np.min(pmaskl_0, axis = 0)
                    ### also adjust the images3 -- skipp the first dim layers 
                    images3 = images3[dim:0]
                else:
                    pmav = pmasks3[:,0,0]
                    pmav_0 = pmasks3_0[:,0,0]
       
                
                #testPlot=False
                if testPlot:
                    for i in range(pmasks3.shape[0]):
                        j = 0
                        for j in range(j_pred, j_pred+1):
                            #img = images3[i, 0, j]
                            
                            print ('scan '+str(i))
                            f, ax = plt.subplots(1, 2, figsize=(10,5))
                            ax[0].imshow(images3[i, 0, j],cmap=plt.cm.gray)
                            ax[1].imshow(pmav[i],cmap=plt.cm.gray)
            
                    dist = pmav.flatten()
                    thresh = 0.005
                    dist = dist[(dist > thresh) & (dist < 1-thresh)]
                    plt.hist(dist, bins=80, color='c')
                    plt.xlabel("Nodules prob")
                    plt.ylabel("Frequency")
                    plt.show()
                    print(len(dist)) 
                
                               
                HU_LUNGS_MIN0 = -990  # includes emphysema 
                HP_LUNGS_EMPHYSEMA_THRESH = -950
                HP_LUNGS_EMPHYSEMA_THRESH2 = -970
                HU_LUNGS_MAX = -400
    
                pix_lungs_min = hu_to_pix(HU_LUNGS_MIN0)
                pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)
                
                ims_seg = images3_seg[:,0,dim // 2]
                idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                
                idv_by_z = np.sum(idv, axis=(1,2))
                idv_by_y = np.sum(idv, axis=(0,2))
                idv_by_x = np.sum(idv, axis=(0,1))
                
              
                lvol_by_z = idv_by_z * range(0,len(idv_by_z))
                lvol_by_y = idv_by_y * range(0,len(idv_by_y))
                lvol_by_x = idv_by_x * range(0,len(idv_by_x))
                
                center_z = np.sum(lvol_by_z)/np.sum(idv_by_z)
                center_y = np.sum(lvol_by_y)/np.sum(idv_by_y)
                center_x = np.sum(lvol_by_x)/np.sum(idv_by_x)
                
                if testPlot:
                    plt.plot(idv_by_z)
                    plt.show()
                    plt.plot(idv_by_z)
                    plt.show()
                    print("center_z, center_y, center_x: ",  center_z, center_y, center_x)   # centre by volume of lungs by z, y, and x
                 
                partn=0
                frames1 = []
                zsel = dim // 2  # use the mid channel/cut (could use any other for the stats and trhey are repeated)
                segment = 2
                for segment in range(3):
                    frames0 = []
                    # 0 = top partn
                    # 1 bottom half
                    # 2 all
                    if segment == 0:
                        sstart = 0
                        send = images3.shape[0] // 2
                    elif segment == 1:
                        sstart = images3.shape[0] // 2
                        send = images3.shape[0]
                    else:                   ### the last one must be the entire set
                        sstart = 0
                        send = images3.shape[0]

                    ims = images3[sstart:send,0,zsel]      # selecting the zsel cut for nodules calc ...
                    ims_seg = images3_seg[sstart:send,0,zsel] 
                    #pms = pmasks3[sstart:send,0,0]
                    pms = pmav[sstart:send]
                    pms_0 = pmav_0[sstart:send]
                    #for thresh in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]: # full set
                    for thresh in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:  # keras 0311
                        
                        idx = pms > thresh
                        nodls = np.zeros(pms.shape).astype(np.int16)
                        nodls[idx] = 1
                        
                        idx_0 = pms_0 > thresh
                        nodls_0 = np.zeros(pms_0.shape).astype(np.int16)
                        nodls_0[idx_0] = 1
                        
                        nx =  nodls[idx]
                        # volume = np.sum(nodls)  # counted as a count within hu_describe ...
                        nodules_pixels = ims[idx]   # flat
                        nodules_hu = pix_to_hu(nodules_pixels)
                        part_name = ''.join([str(segment), '_', str(thresh)])
                        #print ("part_name: ", part_name)
                        #df = hu_describe(nodules_hu, uid, 1)
                        
                        ### DO NOT do them here      
                        use_corrected_nodules = True  # do it below from 20170311
                        if not use_corrected_nodules:
                            df = hu_describe(nodules_hu, uid=uid, part=part_name)
                        
                        ### add any additional params to df ...
                        #sxm = ims * nodls
                        add_projections = False
                        axis = 1
                        nodules_projections = []
                        nodules_projections_0 = []
                        for axis in range(3):
                             #sxm_projection = np.max(sxm, axis = axis)
                             nodls_projection = np.max(nodls, axis=axis)
                             nodls_projection_0 = np.max(nodls_0, axis=axis)
                             naxis_name = ''.join(["naxis_", str(axis),"_", part_name])
                             if add_projections:   
                                 df[naxis_name] = np.sum(nodls_projection)
                             nodules_projections.append(nodls_projection)
                             nodules_projections_0.append(nodls_projection_0)
                        
                        
                        ## find the individual nodules ... as per the specified probabilities 
                        labs, labs_num = measure.label(idx, return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
                        if labs_num > 0:                  
                            #labs_num_to_store = 5                            
                            regprop = measure.regionprops(labs, intensity_image=ims)
                            voxel_volume = np.product(RESIZE_SPACING) 
                            areas = [rp.area for rp in regprop] # this is in cubic mm now (i.e. should really be called volume)                       
                            volumes = [rp.area * voxel_volume for rp in regprop] 
                            diameters = [2 * (3* volume / (4 * np.pi ))**0.3333 for volume in volumes]
 
 
                            labs_ids =  [rp.label for rp in regprop]
                            #ls = [rp.label for rp in regprop]
                            max_val = np.max(areas)
                            max_index = areas.index(max_val)
                            max_label = regprop[max_index].label
                            #max_ls = ls[max_index]
                            zcenters = [(r.bbox[0]+r.bbox[3])/2 - center_z for r in regprop]
                            ycenters = [(r.bbox[1]+r.bbox[4])/2 - center_y for r in regprop]
                            xcenters = [(r.bbox[2]+r.bbox[5])/2 - center_x for r in regprop]
        
                            bbgmeans = [ gmean([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]
                            irreg_vol = [bbgmeans[i]/diameters[i] for i in range(len(diameters))]
                            irreg_shape = [ np.var([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]
                            bb_min =    [ np.min([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]    
                            bb_max =    [ np.max([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]  
                            aspect = [bb_min[i]/bb_max[i] for i in range(len(diameters))]
                            
                            bb_min_xy =    [ np.min([(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]    
                            bb_max_xy =    [ np.max([(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]  
                            aspect_xy = [bb_min_xy[i]/bb_max_xy[i] for i in range(len(diameters))]
                            extent = [rp.extent for rp in regprop]
                            dia_aspect = [diameters[i] * aspect[i] for i in range(len(diameters))]
                            dia_extent = [diameters[i] * extent[i] for i in range(len(diameters))]
                            
                            nodules_min_diameter = 2 if RESIZE_SPACING[1] < 2 else 3 # dybamic subject to resolution used; i.e 2mm or 3mm
                            
                            voxels_min_count = int(((nodules_min_diameter/2)**3) *4*np.pi/3 / (np.product(RESIZE_SPACING)))  # 7

                            idl = labs ==  regprop[max_index].label   #  400
                            nodules_pixels = ims[idl]
                            nodules_hu = pix_to_hu(nodules_pixels)

                            nodules_hu_reg = []
                            for rp in regprop:
                                idl = labs == rp.label
                                nodules_pixels = ims[idl]
                                nodules_hu = pix_to_hu(nodules_pixels)
                                nodules_hu_reg.append(nodules_hu)           # NOTE some are out of interest, i.e. are equal all (or near all) to  MAX_BOUND (400)
                                
                            dfn = pd.DataFrame(
                                {"zcenter": zcenters,
                                 "ycenter": ycenters, 
                                 "xcenter": xcenters, 
                                 "area":    areas,
                                 "diameter":   diameters,
                                 "irreg_vol":  irreg_vol,
                                 "irreg_shape": irreg_shape,
                                 "aspect": aspect,
                                 "aspect_xy": aspect_xy,
                                 "extent": extent,
                                 "dia_extent": dia_extent,
                                 "dia_aspect": dia_aspect,
                                 "nodules_hu": nodules_hu_reg
                                 },
                                 index=labs_ids)
                            
                            nodules_count_0 = len(dfn)
                            of_interest_content = [np.mean(nhu) < MAX_BOUND * 0.99 for nhu in dfn["nodules_hu"]] # CUT OUT at least the ones obvioiusly out of interest
                            dfn = dfn[of_interest_content]
                            nodules_count_1 = len(dfn)
            
                            dfn = dfn[dfn["diameter"] > nodules_min_diameter]  # CUT OUT anything that is less than 2 mm (essentially less than 7 voxels for 2x2x2)
                       
                            nodules_count_2 = len(dfn)
                            if len(dfn) >0:
                                
                                zweighted = np.sum(dfn["area"] * dfn["zcenter"])/np.sum(dfn["area"])
                                yweighted = np.sum(dfn["area"] * dfn["ycenter"])/np.sum(dfn["area"])
                                xweighted = np.sum(dfn["area"] * dfn["xcenter"])/np.sum(dfn["area"])
                                zrel = zweighted / ims.shape[0]
                                yrel = yweighted / ims.shape[1]                                                        
                                xrel = yweighted / ims.shape[2]
                            else:
                                zrel = -7777
                                yrel = -7777
                                xrel = -7777
                            
                            
                        else:
                            nodules_hu = []
                            dfn = []
                            zrel = -7777
                            yrel = -7777
                            xrel = -7777
                            nodules_count_0 = 0
                            nodules_count_1 = 0
                            nodules_count_2 = 0
                        
                        # calculate features for a set number of nodules sorting them by:
                        sort_types_count = 1 # keras 0311, in April 3
                        max_nodules_count = 5 # keras 0311, later 10
                        frame_dfn2 = []
                        for sort in range(sort_types_count):
                            if sort == 0 and len(dfn) > 0:
                                dfn = dfn.sort_values(["diameter"], ascending = [False])
                            elif sort == 1 and len(dfn) > 0:
                                dfn = dfn.sort_values(["dia_aspect"], ascending = [False])
                            elif sort == 2 and len(dfn) > 0:
                                dfn = dfn.sort_values(["dia_extent"], ascending = [False])
                                
                            for n in range(max_nodules_count):
                                part_name = ''.join([str(segment), '_', str(thresh),'_n', str(n), '_', str(sort)])
                                #print (part_name)
                                #columns_to_use = ["zcenter", "ycenter", "xcenter", "diameter", "irreg_vol", "irreg_shape",
                                #                  "aspect", "aspect_xy", "extent", "dia_extent", "dia_aspect"]
                                columns_to_use = ["zcenter", "ycenter", "xcenter", "diameter", "irreg_vol", "irreg_shape"] #  keras 0311,
                                                  #"aspect", "aspect_xy", "extent", "dia_extent", "dia_aspect"]


                                if n < len(dfn): # use the nodule data, otheriwse empty
                                    dfn2 = hu_describe(dfn.iloc[n]["nodules_hu"], uid=uid, part=part_name) 
                                    for col in columns_to_use:
                                        dfn2[''.join([col, '_', part_name])] = dfn.iloc[n][col]
                                else:
                                    nodule_hu = []
                                    dfn2 = hu_describe(nodule_hu, uid=uid, part=part_name) 
                                    for col in columns_to_use:
                                        dfn2[''.join([col, '_', part_name])] = -7777
                                frame_dfn2.append(dfn2)
                                
                        df2 = pd.concat(frame_dfn2, axis=1, join_axes=[frame_dfn2[0].index])
                        
                        part_name = ''.join([str(segment), '_', str(thresh),'_ns'])
                        #df2 = hu_describe(nodules_hu, uid=uid, part=part_name) 
                        df2[ ''.join(["zrel_", part_name])] = zrel
                        df2[ ''.join(["yrel_", part_name])] = yrel
                        df2[ ''.join(["xrel_", part_name])] = xrel
                        count_name = ''.join(["ncount_", str(segment), '_', str(thresh)])
                        #print ("count_name: ", count_name)
                        df2[count_name] = len(dfn)
                        
                        
                        if use_corrected_nodules: # now calculated the overall stats based on corrected info
                            part_name = ''.join([str(segment), '_', str(thresh)])
                            if len(dfn) > 0:
                                nodules_hu = [dfn.iloc[n]["nodules_hu"] for n in range(len(dfn))]
                                nodules_hu = np.concatenate(nodules_hu)  
                            else:
                                nodules_hu = []    
                            df = hu_describe(nodules_hu, uid=uid, part=part_name)
                      
                           
                        df3 = pd.concat( [df, df2], axis =1)
                        
                        
                        frames0.append(df3)
                        partn += 1
                                  
                    dfseg = pd.concat(frames0, axis=1, join_axes=[frames0[0].index])  # combine rows
                    
                    ##### add any section features independent of nodules ....
                    # estimate lung volume by counting all relevant pixels using the segmented lungs data
                    # for d16g do it for the 3 different areas, including the emphysema calcs
                    
                    idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                    idv_emphysema  =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH))  & (ims_seg < pix_lungs_max)
                    idv_emphysema2 =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH2)) & (ims_seg < pix_lungs_max)
                    
                    
                    test_emphysema = False
                    if test_emphysema:
                        
                        f, ax = plt.subplots(1, 4, figsize=(20,5))
                        ax[0].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                        ax[1].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                        #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                        ax[2].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                        ax[3].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                        
                        #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                        plt.show()
          
                        idv_all = []
                        hu_lungs_min_all=[]
                        for HU_LUNGS_MIN in range(-1000, -800, 10):
                            pix_lungs_min = hu_to_pix(HU_LUNGS_MIN)
                            pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)
    
                            idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                            idv = (ims >pix_lungs_min) & (ims < pix_lungs_max)
                            idv_all.append(np.sum(idv))
                            hu_lungs_min_all.append(HU_LUNGS_MIN)
                        e_results = pd.DataFrame(
                            {"HU min": hu_lungs_min_all,
                             "volume approx": np.round(idv_all, 4)
                             })
                        plt.figure()
                        e_results.plot()
                        e_results.plot(kind='barh', x='HU min', y='volume approx', legend=False, figsize=(6, 10))
                            
                            
                    lvol = np.sum(idv)
                           
                    dfseg[''.join(['lvol', '_', str(segment)])] =  lvol   # 'lvol_2' etc.
                    dfseg[''.join(['emphy', '_', str(segment)])] = (lvol - np.sum(idv_emphysema))/(lvol+1)
                    dfseg[''.join(['emphy2', '_', str(segment)])] = (lvol - np.sum(idv_emphysema2))/(lvol+1)
            
                    frames1.append(dfseg)
                    
                df = pd.concat(frames1, axis=1, join_axes=[frames1[0].index])  # combine rows
                
                ## use the most recent threshold as a test ...
               
                testPlot2 = False
                if testPlot2:        
                    print ("Summary, Count, Cancer, nodules_count_0, 1 amd 2, uid, process time: ", count, cancer, nodules_count_0,nodules_count_1 , nodules_count_2, uid, time.time()-start_time)
 
                    if ims.shape[0] > ims.shape[1]:
                        print ("Suspicious hight, shape: ", images3.shape, uid)
                    f, ax = plt.subplots(1, 6, figsize=(24,4))
                    ax[0].imshow(nodules_projections_0[0],cmap=plt.cm.gray)
                    ax[1].imshow(nodules_projections_0[1],cmap=plt.cm.gray)
                    ax[2].imshow(nodules_projections[0],cmap=plt.cm.gray)
                    ax[3].imshow(nodules_projections[1],cmap=plt.cm.gray)
                    #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                    ax[4].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                    if not valonly:
                        ax[5].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                    else:
                        val_voxel = val[val_uids == uid][0]
                        ax[5].imshow(val_voxel[0, dim//2],cmap=plt.cm.gray)
                    
                        #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                    plt.show()
      
                    
                ### calculate the volume in the central sections (say 60% - 5 sectors)  .-- may be capturing circa 90% of total volume for some (trying to reduce impact of outside/incorect segmetations??)..
                sections = 5
                for sections in range(3,7):       
                    zstart = ims_seg.shape[0] // sections
                    zstop = (sections -1) * ims_seg.shape[0] // sections
                    ims_part = ims_seg[zstart:zstop]
                    
                    #idv = (ims_part > 0.9 * np.min(ims_part)) & (ims_part < 0.9 * np.max(ims_part))
                    idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
              
                    #df["lvol_sct"] =  np.sum(idv) 
                    df["lvol_sct%s" % sections] =  np.sum(idv) 
                    
                    #df["lvol"]
                    #df["lvol_sct"] /  df["lvol"]
                
                
                df["cancer"] = cancer
                df["zshape"] = images3.shape[0] 
                df["yshape"] = images3.shape[3]
                df["xshape"] = images3.shape[4]
        
                
                testPrint = False
                if testPrint:
                    dfeasy = np.round(df, 2)
                    if cnt == 0:
                        print (dfeasy)
                        #print(dfeasy.to_string(header=True))
                    else:
                        print(dfeasy.to_string(header=False))
             
                cnt += 1
                if (cnt % 10 == 0):
                    print ("Scans processed: ", cnt)
                
                frames.append(df)
                gc.collect()
                
          
    result = pd.concat(frames)
    return result

 
def recalc_features_keras_3dx(stage, dim, run, processors, withinsegonly= True, valonly = False):

    STAGE_DIR = STAGE_DIR_BASE % stage
    LABELS = LABELS_BASE % stage

    labels = pd.read_csv(LABELS) ### was data/
    #print(labels.head())


    source_data_name = ''.join([stage, "_", RESOLUTION_STR ])
    source_data_name_seg = ''.join([stage, "_segmented_", RESOLUTION_STR ])

    stage_dir_lungs = STAGE_DIR.replace(stage, source_data_name, 1)  # was stage1_segmented_2x2x2
    
    files = glob.glob(os.path.join(stage_dir_lungs,'*.npz'))
    
    limit_to_ids = []            # results of grep Suspicious *.txt    (in the results directory:w)

    if valonly:
        vsdf = pd.read_csv("valset_stratified_1542_0.csv", index_col = 0)
        limit_to_ids = list(vsdf["uid"])
        
        varstr = str(5)
        stratstr = "str"
        val =    np.load(''.join(("../models/val_16g6_th0999_279_%s%s" % (varstr, stratstr), ".npz")))['arr_0']   
        val_uids =    np.load(''.join(("../models/val_uids_16g6_th0999_279_%s%s" % (varstr, stratstr), ".npz")))['arr_0'] 
        val_labels =    np.load(''.join(("../models/val_labels_16g6_th0999_279_%s%s" % (varstr, stratstr), ".npz")))['arr_0'] 
  
    path = files[6]
    start_file = 0      # was 508 for a restart
    count = start_file
 
    last_file = len(files)
    cnt = 0
    frames = []
    
    feats_bb_shape = []
    calculate_only_feats_bb_shape = False  # quick manual hack
    
    for path in files[start_file:last_file]:
        
        uid =  path[path.rindex('/')+1:] 
        uid = uid[:-4]   # cut the .npz suffix
        
  
        if (uid in labels["id"].tolist()):
            cancer = int(labels["cancer"][labels["id"] == uid])  # so we know during testing and also later store it
        else:
            cancer = -777
            
        if uid in limit_to_ids or len(limit_to_ids) == 0:   # either use the limit_to list or use all
            count += 1
            if count % processors == run:    # do this part in this process, otherwise skip
            #for folder in mask_wrong:
            
                start_time = time.time() 
##########################################

                path_mpred = path.replace(source_data_name, (source_data_name + "_mpred3_%s") % str(dim), 1)
                path_seg = path.replace(source_data_name, source_data_name_seg, 1)
     
                # read the bouding box saved when mpred was genearted  
                bb = pd.read_csv(path_mpred[:-4] + ".csv", index_col = 0) #to check
                ### use uid to ensure errot checking and also consolidation of the bb's if need to
                by_min = bb.loc[uid].by_min
                by_max = bb.loc[uid].by_max
                bx_min = bb.loc[uid].bx_min
                bx_max = bb.loc[uid].bx_max
                print ("by_min, by_max, bx_min, bx_max: ", by_min, by_max, bx_min, bx_max)
    
    
                pmasks3 = np.load(''.join((path_mpred, "")))['arr_0']   # the suffix is laready there noe need tp add".npz"
                
                images3 = get_segmented_preprocess_bb_xd_file(path, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
                images3 = images3[:,np.newaxis]  # add a dimension for the 3 D model
           
                images3_seg = get_segmented_preprocess_bb_xd_file(path_seg, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
                images3_seg = images3_seg[:,np.newaxis]  # add a dimension for the 3 D model


               
                dim_out = pmasks3.shape[2]
                #        reduce the images and pmasks to 2 dimension sets
        
                #### NOW cut off the non-lungs areas by the z axis 
 
                bb["zshape"] = images3.shape[0]
                bb["yshape"] = images3.shape[3]
                bb["xshape"] = images3.shape[4]
                do_cut_out_non_lungs_z4 = False  ## manual hack for a while -- needs to be aligned wiht the final code FINAL CODE
                if do_cut_out_non_lungs_z4:
                    images3, pmasks3, images3_seg = cut_out_non_lungs_z (images3, pmasks3, images3_seg, uid, dim)
                    
                if calculate_only_feats_bb_shape:
                    feats_bb_shape.append(bb)
                    continue
                #feats_bb_shape = pd.concat(feats_bb_shape)    
        
                # further constrain within segmented lungs only
                pix_min_001 =    hu_to_pix(MIN_BOUND + 0.001)
                pmasks3_0 = np.array(pmasks3)
                if withinsegonly:    
                    pmasks3[images3_seg[:,:,dim//2:(dim//2+1)] < pix_min_001] = 0    
             
                testPlot = False
                j_pred = dim // 2
                if testPlot:
                    i=31  # for tests
                    for i in range(images3.shape[0]):
                        j = 0
                        for j in range(j_pred, j_pred+1):  # we use the dim cut for results
                            img = images3[i, 0, j]
                            img_seg =  images3_seg[i, 0, j]
                            #pmsk = pmasks3[i,0,j]  # for multi-plane output
                            pmsk = pmasks3[i,0,0]
                            pmsk_max = np.max(pmsk)
                            #print ('scan '+str(i))
                            print ('scan & probability max_value: ', i, pmsk_max)
                            if pmsk_max > 0.9:
                                f, ax = plt.subplots(1, 3, figsize=(12,4))
                                ax[0].imshow(img,cmap=plt.cm.gray)
                                ax[1].imshow(pmsk,cmap=plt.cm.gray)
                                ax[2].imshow(img_seg,cmap=plt.cm.gray)
                                ####ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                                plt.show()
                                #print ("max =", np.max(pmsk))
                    
        
                #### list solution for the several rolls that also reduce the memory requirements as we only store the relevant cut ...
                
                if dim_out > 1:
                    pmaskl = [np.roll(pmasks3, shift, axis =0)[dim:,0,shift] for shift in range(dim)]  # pmask list (skipping the first dim elements)
                    pmav = np.min(pmaskl, axis = 0)
                    
                    pmaskl_0 = [np.roll(pmasks3_0, shift, axis =0)[dim:,0,shift] for shift in range(dim)]  # pmask list (skipping the first dim elements)
                    pmav_0 = np.min(pmaskl_0, axis = 0)
                    ### also adjust the images3 -- skipp the first dim layers 
                    images3 = images3[dim:0]
                else:
                    pmav = pmasks3[:,0,0]
                    pmav_0 = pmasks3_0[:,0,0]
       
                
                
                #testPlot=False
                if testPlot:
                    for i in range(pmasks3.shape[0]):
                        j = 0
                        for j in range(j_pred, j_pred+1):
                            #img = images3[i, 0, j]
                            
                            print ('scan '+str(i))
                            f, ax = plt.subplots(1, 2, figsize=(10,5))
                            ax[0].imshow(images3[i, 0, j],cmap=plt.cm.gray)
                            ax[1].imshow(pmav[i],cmap=plt.cm.gray)
    
                
                
                
                    dist = pmav.flatten()
                    thresh = 0.005
                    dist = dist[(dist > thresh) & (dist < 1-thresh)]
                    plt.hist(dist, bins=80, color='c')
                    plt.xlabel("Nodules prob")
                    plt.ylabel("Frequency")
                    plt.show()
                    print(len(dist))  #2144 for pmav dist , 3748 for pm , 1038 for pm[:,0,0], 627 for pm[:,0,1], 770 for pm[:,0, 2], 1313 for for pm[:,0, 3]; 1129 for pmav2 - peak around 0.5 where it was not there before
                
                
                
                               
                HU_LUNGS_MIN0 = -990  # includes emphysema 
                HP_LUNGS_EMPHYSEMA_THRESH = -950
                HP_LUNGS_EMPHYSEMA_THRESH2 = -970
                HU_LUNGS_MAX = -400
    
                pix_lungs_min = hu_to_pix(HU_LUNGS_MIN0)
                pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)
                
                ims_seg = images3_seg[:,0,dim // 2]
                idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                
                idv_by_z = np.sum(idv, axis=(1,2))
                idv_by_y = np.sum(idv, axis=(0,2))
                idv_by_x = np.sum(idv, axis=(0,1))
                
              
                lvol_by_z = idv_by_z * range(0,len(idv_by_z))
                lvol_by_y = idv_by_y * range(0,len(idv_by_y))
                lvol_by_x = idv_by_x * range(0,len(idv_by_x))
                
                center_z = np.sum(lvol_by_z)/np.sum(idv_by_z)
                center_y = np.sum(lvol_by_y)/np.sum(idv_by_y)
                center_x = np.sum(lvol_by_x)/np.sum(idv_by_x)
                
                if testPlot:
                    plt.plot(idv_by_z)
                    plt.show()
                    plt.plot(idv_by_z)
                    plt.show()
                    print("center_z, center_y, center_x: ",  center_z, center_y, center_x)   # centre by volume of lungs by z, y, and x
                 
                
                 
                partn=0
                frames1 = []
                zsel = dim // 2  # use the mid channel/cut (could use any other for the stats and trhey are repeated)
                segment = 2
                for segment in range(3):
                    frames0 = []
                    # 0 = top partn
                    # 1 bottom half
                    # 2 all
                    if segment == 0:
                        sstart = 0
                        send = images3.shape[0] // 2
                    elif segment == 1:
                        sstart = images3.shape[0] // 2
                        send = images3.shape[0]
                    else:                   ### the last one must be the entire set
                        sstart = 0
                        send = images3.shape[0]

                    ims = images3[sstart:send,0,zsel]      # selecting the zsel cut for nodules calc ...
                    ims_seg = images3_seg[sstart:send,0,zsel] 
                    #pms = pmasks3[sstart:send,0,0]
                    pms = pmav[sstart:send]
                    pms_0 = pmav_0[sstart:send]
                    
                    # threshold the precited nasks ...
                    #for thresh in [0.5, 0.9, 0.9999]:
                    #for thresh in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
                    for thresh in [0.75, 0.9, 0.999, 0.999999, 0.9999999]:
                    #for thresh in [0.99999]:
                        #thresh = 0.9  #0.5 0.75 0.9 0.95 0.98
                        
                        idx = pms > thresh
                        nodls = np.zeros(pms.shape).astype(np.int16)
                        nodls[idx] = 1
                        
                        idx_0 = pms_0 > thresh
                        nodls_0 = np.zeros(pms_0.shape).astype(np.int16)
                        nodls_0[idx_0] = 1
                        
                        nx =  nodls[idx]
                        # volume = np.sum(nodls)  # counted as a count within hu_describe ...
                        nodules_pixels = ims[idx]   # flat
                        nodules_hu = pix_to_hu(nodules_pixels)
                        part_name = ''.join([str(segment), '_', str(thresh)])
                        #print ("part_name: ", part_name)
                        #df = hu_describe(nodules_hu, uid, 1)
                        
                        ### DO NOT do them here      
                        use_corrected_nodules = True  # do it below from 20170311
                        if not use_corrected_nodules:
                            df = hu_describe(nodules_hu, uid=uid, part=part_name)
                        
                        ### add any additional params to df ...
                        #sxm = ims * nodls
                        add_projections = False
                        axis = 1
                        nodules_projections = []
                        nodules_projections_0 = []
                        for axis in range(3):
                             #sxm_projection = np.max(sxm, axis = axis)
                             nodls_projection = np.max(nodls, axis=axis)
                             nodls_projection_0 = np.max(nodls_0, axis=axis)
                             naxis_name = ''.join(["naxis_", str(axis),"_", part_name])
                             if add_projections:   
                                 df[naxis_name] = np.sum(nodls_projection)
                             nodules_projections.append(nodls_projection)
                             nodules_projections_0.append(nodls_projection_0)
                        
                        
                        ## find the individual nodules ... as per the specified probabilities 
                        labs, labs_num = measure.label(idx, return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
                        if labs_num > 0:                  
                            #labs_num_to_store = 5                            
                            regprop = measure.regionprops(labs, intensity_image=ims)
                            voxel_volume = np.product(RESIZE_SPACING) 
                            areas = [rp.area for rp in regprop] # this is in cubic mm now (i.e. should really be called volume)                       
                            volumes = [rp.area * voxel_volume for rp in regprop] 
                            diameters = [2 * (3* volume / (4 * np.pi ))**0.3333 for volume in volumes]
 
 
                            labs_ids =  [rp.label for rp in regprop]
                            #ls = [rp.label for rp in regprop]
                            max_val = np.max(areas)
                            max_index = areas.index(max_val)
                            max_label = regprop[max_index].label
                            #max_ls = ls[max_index]
                        
                            
                            zcenters = [(r.bbox[0]+r.bbox[3])/2 - center_z for r in regprop]
                            ycenters = [(r.bbox[1]+r.bbox[4])/2 - center_y for r in regprop]
                            xcenters = [(r.bbox[2]+r.bbox[5])/2 - center_x for r in regprop]
        
                            bbgmeans = [ gmean([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]
                            irreg_vol = [bbgmeans[i]/diameters[i] for i in range(len(diameters))]
                            irreg_shape = [ np.var([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]
                            bb_min =    [ np.min([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]    
                            bb_max =    [ np.max([(r.bbox[3]-r.bbox[0])*RESIZE_SPACING[0],(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]  
                            aspect = [bb_min[i]/bb_max[i] for i in range(len(diameters))]
                            
                            bb_min_xy =    [ np.min([(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]    
                            bb_max_xy =    [ np.max([(r.bbox[4]-r.bbox[1])*RESIZE_SPACING[1], (r.bbox[5]-r.bbox[2])*RESIZE_SPACING[2] ]) for r in regprop]  
                            aspect_xy = [bb_min_xy[i]/bb_max_xy[i] for i in range(len(diameters))]
                            extent = [rp.extent for rp in regprop]
                            dia_aspect = [diameters[i] * aspect[i] for i in range(len(diameters))]
                            dia_extent = [diameters[i] * extent[i] for i in range(len(diameters))]
                            
                            nodules_min_diameter = 2 if RESIZE_SPACING[1] < 2 else 3 # dybamic subject to resolution used; i.e 2mm or 3mm
                            
                            voxels_min_count = int(((nodules_min_diameter/2)**3) *4*np.pi/3 / (np.product(RESIZE_SPACING)))  # 7

                            idl = labs ==  regprop[max_index].label   #  400
                            nodules_pixels = ims[idl]
                            nodules_hu = pix_to_hu(nodules_pixels)

                            nodules_hu_reg = []
                            for rp in regprop:
                                idl = labs == rp.label
                                nodules_pixels = ims[idl]
                                nodules_hu = pix_to_hu(nodules_pixels)
                                nodules_hu_reg.append(nodules_hu)           # NOTE some are out of interest, i.e. are equal all (or near all) to  MAX_BOUND (400)
                                
                            dfn = pd.DataFrame(
                                {"zcenter": zcenters,
                                 "ycenter": ycenters, 
                                 "xcenter": xcenters, 
                                 "area":    areas,
                                 "diameter":   diameters,
                                 "irreg_vol":  irreg_vol,
                                 "irreg_shape": irreg_shape,
                                 "aspect": aspect,
                                 "aspect_xy": aspect_xy,
                                 "extent": extent,
                                 "dia_extent": dia_extent,
                                 "dia_aspect": dia_aspect,
                                 "nodules_hu": nodules_hu_reg
                                 },
                                 index=labs_ids)
                            
                            nodules_count_0 = len(dfn)
                            of_interest_content = [np.mean(nhu) < MAX_BOUND * 0.99 for nhu in dfn["nodules_hu"]] # CUT OUT at least the ones obvioiusly out of interest
                            dfn = dfn[of_interest_content]
                            nodules_count_1 = len(dfn)
            
                            #dfn = dfn[dfn["diameter"] > nodules_min_diameter]  # CUT OUT anything that is less than 2 mm (essentially less than 7 voxels for 2x2x2)
                       
                            nodules_count_2 = len(dfn)
                            if len(dfn) >0:
                                
                                zweighted = np.sum(dfn["area"] * dfn["zcenter"])/np.sum(dfn["area"])
                                yweighted = np.sum(dfn["area"] * dfn["ycenter"])/np.sum(dfn["area"])
                                xweighted = np.sum(dfn["area"] * dfn["xcenter"])/np.sum(dfn["area"])
                                zrel = zweighted / ims.shape[0]
                                yrel = yweighted / ims.shape[1]                                                        
                                xrel = yweighted / ims.shape[2]
                            else:
                                zrel = -7777
                                yrel = -7777
                                xrel = -7777
                            
                            
                        else:
                            nodules_hu = []
                            dfn = []
                            zrel = -7777
                            yrel = -7777
                            xrel = -7777
                            nodules_count_0 = 0
                            nodules_count_1 = 0
                            nodules_count_2 = 0
                        
                        # calculate features for a set number of nodules sorting them by:
                        sort_types_count = 3
                        max_nodules_count = 10
                        frame_dfn2 = []
                        for sort in range(sort_types_count):
                            if sort == 0 and len(dfn) > 0:
                                dfn = dfn.sort_values(["diameter"], ascending = [False])
                            elif sort == 1 and len(dfn) > 0:
                                dfn = dfn.sort_values(["dia_aspect"], ascending = [False])
                            elif sort == 2 and len(dfn) > 0:
                                dfn = dfn.sort_values(["dia_extent"], ascending = [False])
                                
                            for n in range(max_nodules_count):
                                part_name = ''.join([str(segment), '_', str(thresh),'_n', str(n), '_', str(sort)])
                                #print (part_name)
                                columns_to_use = ["zcenter", "ycenter", "xcenter", "diameter", "irreg_vol", "irreg_shape",
                                                  "aspect", "aspect_xy", "extent", "dia_extent", "dia_aspect"]

                                if n < len(dfn): # use the nodule data, otheriwse empty
                                    dfn2 = hu_describe(dfn.iloc[n]["nodules_hu"], uid=uid, part=part_name) 
                                    for col in columns_to_use:
                                        dfn2[''.join([col, '_', part_name])] = dfn.iloc[n][col]
                                else:
                                    nodule_hu = []
                                    dfn2 = hu_describe(nodule_hu, uid=uid, part=part_name) 
                                    for col in columns_to_use:
                                        dfn2[''.join([col, '_', part_name])] = -7777
                                frame_dfn2.append(dfn2)
                                
                        df2 = pd.concat(frame_dfn2, axis=1, join_axes=[frame_dfn2[0].index])
                        
                        part_name = ''.join([str(segment), '_', str(thresh),'_ns'])
                        #df2 = hu_describe(nodules_hu, uid=uid, part=part_name) 
                        df2[ ''.join(["zrel_", part_name])] = zrel
                        df2[ ''.join(["yrel_", part_name])] = yrel
                        df2[ ''.join(["xrel_", part_name])] = xrel
                        count_name = ''.join(["ncount_", str(segment), '_', str(thresh)])
                        #print ("count_name: ", count_name)
                        df2[count_name] = len(dfn)
                        
                        
                        if use_corrected_nodules: # now calculated the overall stats based on corrected info
                            part_name = ''.join([str(segment), '_', str(thresh)])
                            if len(dfn) > 0:
                                nodules_hu = [dfn.iloc[n]["nodules_hu"] for n in range(len(dfn))]
                                nodules_hu = np.concatenate(nodules_hu)  
                            else:
                                nodules_hu = []    
                            df = hu_describe(nodules_hu, uid=uid, part=part_name)
                      
                           
                        df3 = pd.concat( [df, df2], axis =1)
                        
                        
                        frames0.append(df3)
                        partn += 1
                                  
                    dfseg = pd.concat(frames0, axis=1, join_axes=[frames0[0].index])  # combine rows
                    
                     ##### add any section features independent of nodules ....
                    # estimate lung volume by counting all relevant pixels using the segmented lungs data
                    # for d16g do it for the 3 different areas, including the emphysema calcs
                    
     
                    idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                    idv_emphysema  =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH))  & (ims_seg < pix_lungs_max)
                    idv_emphysema2 =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH2)) & (ims_seg < pix_lungs_max)
                    
                    
                    test_emphysema = False
                    if test_emphysema:
                        
                        f, ax = plt.subplots(1, 4, figsize=(20,5))
                        ax[0].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                        ax[1].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                        #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                        ax[2].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                        ax[3].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                        
                        #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                        plt.show()
          
                        idv_all = []
                        hu_lungs_min_all=[]
                        for HU_LUNGS_MIN in range(-1000, -800, 10):
                            pix_lungs_min = hu_to_pix(HU_LUNGS_MIN)
                            pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)
    
                            idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                            idv = (ims >pix_lungs_min) & (ims < pix_lungs_max)
                            idv_all.append(np.sum(idv))
                            hu_lungs_min_all.append(HU_LUNGS_MIN)
                        e_results = pd.DataFrame(
                            {"HU min": hu_lungs_min_all,
                             "volume approx": np.round(idv_all, 4)
                             })
                        plt.figure()
                        e_results.plot()
                        e_results.plot(kind='barh', x='HU min', y='volume approx', legend=False, figsize=(6, 10))
                            
                            
                    lvol = np.sum(idv)
                           
                    dfseg[''.join(['lvol', '_', str(segment)])] =  lvol   # 'lvol_2' etc.
                    dfseg[''.join(['emphy', '_', str(segment)])] = (lvol - np.sum(idv_emphysema))/(lvol+1)
                    dfseg[''.join(['emphy2', '_', str(segment)])] = (lvol - np.sum(idv_emphysema2))/(lvol+1)
            
                    frames1.append(dfseg)
                    
            
            
                df = pd.concat(frames1, axis=1, join_axes=[frames1[0].index])  # combine rows
                
          
                ## use the most recent threshold as a test ...
               
                
                testPlot2 = True
                #if testPlot2 and ims.shape[0] > ims.shape[1]:        
                if testPlot2:        
                    print ("Summary, Count, Cancer, nodules_count_0, 1 amd 2, uid, process time: ", count, cancer, nodules_count_0,nodules_count_1 , nodules_count_2, uid, time.time()-start_time)
 
                    if ims.shape[0] > ims.shape[1]:
                        print ("Suspicious hight, shape: ", images3.shape, uid)
                    f, ax = plt.subplots(1, 6, figsize=(24,4))
                    ax[0].imshow(nodules_projections_0[0],cmap=plt.cm.gray)
                    ax[1].imshow(nodules_projections_0[1],cmap=plt.cm.gray)
                    ax[2].imshow(nodules_projections[0],cmap=plt.cm.gray)
                    ax[3].imshow(nodules_projections[1],cmap=plt.cm.gray)
                    #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                    ax[4].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                    if not valonly:
                        ax[5].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                    else:
                        val_voxel = val[val_uids == uid][0]
                        ax[5].imshow(val_voxel[0, dim//2],cmap=plt.cm.gray)
                    
                    #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                    plt.show()
      
                
                ### calculate the volume in the central sections (say 60% - 5 sectors)  .-- may be capturing circa 90% of total volume for some (trying to reduce impact of outside/incorect segmetations??)..
                sections = 5
                for sections in range(3,7):       
                    zstart = ims_seg.shape[0] // sections
                    zstop = (sections -1) * ims_seg.shape[0] // sections
                    ims_part = ims_seg[zstart:zstop]
                    
                    #idv = (ims_part > 0.9 * np.min(ims_part)) & (ims_part < 0.9 * np.max(ims_part))
                    idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
              
                    #df["lvol_sct"] =  np.sum(idv) 
                    df["lvol_sct%s" % sections] =  np.sum(idv) 
                    
                    #df["lvol"]
                    #df["lvol_sct"] /  df["lvol"]
                
                
                df["cancer"] = cancer
        
                
                testPrint = False
                if testPrint:
                    dfeasy = np.round(df, 2)
                    if cnt == 0:
                        print (dfeasy)
                        #print(dfeasy.to_string(header=True))
                    else:
                        print(dfeasy.to_string(header=False))
             
                cnt += 1
                if (cnt % 10 == 0):
                    print ("Scans processed: ", cnt)
                
                frames.append(df)
                gc.collect()
                
          
    result = pd.concat(frames)
    
    return result  

def hu_to_pix (hu):
    return (hu - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN

def pix_to_hu (pix):
    return (pix + PIXEL_MEAN)  * (MAX_BOUND - MIN_BOUND) + MIN_BOUND



def grid_data(source, grid=32, crop=16, expand=12):
    #gridsize = grid + 2 * expand
    #stacksize = source.shape[0]
    height = source.shape[3]  # should be 224 for our data
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
    
    #height = cells.shape[3]  # should be 224 for our data
    width = cells.shape[4]
    crop = (width - grid ) // 2 ## for simplicity we are assuming the same crop (and grid) vertically and horizontally
    #dspacing = gridwidth * gridheight
    #layers = cells.shape[0] // dspacing
    
    cells = cells[:,:,:,crop:-crop,crop:-crop]     
    
    shape = cells.shape
    new_shape_1_dim = shape[0]//36
    new_shape = (36, new_shape_1_dim, ) +  tuple([x for x in shape][1:]) 
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


img_rows = 448      ## global value
img_cols = 448      ## global value



if __name__ == '__main__':
    
    
    stage = "stage1"
    feats = []
    feats1 = []
    feats2 = []


    dim = 8
    run = 0
    processors = 1
    
    
    batch_size = 1 # was 36
    #count, cnt = calc_keras_3d_preds(dim, run, processors, batch_size)
    
    date_version = "0411x"       # set as per the final submission 
    
    make_predictions = True  # you may wish to set it False once the nodules have beein identified, as this step is time consuming
    if make_predictions:
      for stage in ["stage1", "stage2"]:
        start_time = time.time()
        
        ### REDEFINE the nodule identifier to be used if/as needed, as per the ReadMe.txt description  (by commenting out only the relevant option)
        model_weights_name = "../luna/models/d8_2x2x2_best_weights.h5"  #  Option 1

        #model_weights_name = "../luna/models/d8g_bre_weights_50.h5"    #  Option 2
        #model_weights_name = "../luna/models/d8g4a_weights_74.h5"      #  Option 3
        
        feats = calc_features_keras_3dx(stage, dim, run, processors, model_weights_name)
        
        fname = 'feats_base_8_%s_%s_%s.csv'% (stage, len(feats), date_version)
        print ("OVERALL Process time, predictions & base features: ", stage, fname, time.time()-start_time)
        feats.to_csv(fname, index=True)
        

    # Create 3 features files, starting from the most recent one, and 2 compatible with the March calculations
    stage = "stage2"
    for stage in ["stage1", "stage2"]:
        start_time = time.time()
        
        feats3 = recalc_features_keras_3dx(stage, dim, run, processors, withinsegonly= False, valonly=False)
        
        fname3 = 'feats_keras8_0313_%s_%s_%s.csv'% (stage, len(feats1), date_version)
        feats3.to_csv(fname3, index=True)
        print ("OVERALL feature file and process: ", stage, fname3, time.time()-start_time)
        print ("Validation: Any NaNs in features? ", feats3.isnull().values.any())
        

        
        start_time = time.time()
        
        feats1 = recalc_features_keras_3dx_0313(stage, dim, run, processors, withinsegonly= False, valonly=False)
        
        fname1 = 'feats_keras8_0313_%s_%s_%s.csv'% (stage, len(feats1), date_version)
        feats1.to_csv(fname1, index=True)
        print ("OVERALL feature file and process: ", stage, fname1, time.time()-start_time)
        print ("Validation: Any NaNs in features? ", feats1.isnull().values.any())

        feats_not_in_0311_version = {'xcenter_0_0.5_n0_0', 'xcenter_0_0.5_n1_0', 'xcenter_0_0.5_n2_0', 'xcenter_0_0.5_n3_0', 'xcenter_0_0.5_n4_0', 'xcenter_0_0.75_n0_0', 'xcenter_0_0.75_n1_0', 'xcenter_0_0.75_n2_0',
             'xcenter_0_0.75_n3_0', 'xcenter_0_0.75_n4_0', 'xcenter_0_0.95_n0_0', 'xcenter_0_0.95_n1_0', 'xcenter_0_0.95_n2_0', 'xcenter_0_0.95_n3_0', 'xcenter_0_0.95_n4_0', 'xcenter_0_0.98_n0_0', 'xcenter_0_0.98_n1_0',
             'xcenter_0_0.98_n2_0', 'xcenter_0_0.98_n3_0', 'xcenter_0_0.98_n4_0', 'xcenter_0_0.9999999_n0_0', 'xcenter_0_0.9999999_n1_0', 'xcenter_0_0.9999999_n2_0', 'xcenter_0_0.9999999_n3_0', 'xcenter_0_0.9999999_n4_0',
             'xcenter_0_0.999999_n0_0', 'xcenter_0_0.999999_n1_0', 'xcenter_0_0.999999_n2_0', 'xcenter_0_0.999999_n3_0', 'xcenter_0_0.999999_n4_0', 'xcenter_0_0.99999_n0_0', 'xcenter_0_0.99999_n1_0', 'xcenter_0_0.99999_n2_0',
             'xcenter_0_0.99999_n3_0', 'xcenter_0_0.99999_n4_0', 'xcenter_0_0.9999_n0_0', 'xcenter_0_0.9999_n1_0', 'xcenter_0_0.9999_n2_0', 'xcenter_0_0.9999_n3_0', 'xcenter_0_0.9999_n4_0', 'xcenter_0_0.999_n0_0',
             'xcenter_0_0.999_n1_0', 'xcenter_0_0.999_n2_0', 'xcenter_0_0.999_n3_0', 'xcenter_0_0.999_n4_0', 'xcenter_0_0.99_n0_0', 'xcenter_0_0.99_n1_0', 'xcenter_0_0.99_n2_0', 'xcenter_0_0.99_n3_0', 'xcenter_0_0.99_n4_0',
             'xcenter_0_0.9_n0_0', 'xcenter_0_0.9_n1_0', 'xcenter_0_0.9_n2_0', 'xcenter_0_0.9_n3_0', 'xcenter_0_0.9_n4_0', 'xcenter_1_0.5_n0_0', 'xcenter_1_0.5_n1_0', 'xcenter_1_0.5_n2_0', 'xcenter_1_0.5_n3_0',
             'xcenter_1_0.5_n4_0', 'xcenter_1_0.75_n0_0', 'xcenter_1_0.75_n1_0', 'xcenter_1_0.75_n2_0', 'xcenter_1_0.75_n3_0', 'xcenter_1_0.75_n4_0', 'xcenter_1_0.95_n0_0', 'xcenter_1_0.95_n1_0', 'xcenter_1_0.95_n2_0',
             'xcenter_1_0.95_n3_0', 'xcenter_1_0.95_n4_0', 'xcenter_1_0.98_n0_0', 'xcenter_1_0.98_n1_0', 'xcenter_1_0.98_n2_0', 'xcenter_1_0.98_n3_0', 'xcenter_1_0.98_n4_0', 'xcenter_1_0.9999999_n0_0', 'xcenter_1_0.9999999_n1_0',
             'xcenter_1_0.9999999_n2_0', 'xcenter_1_0.9999999_n3_0', 'xcenter_1_0.9999999_n4_0', 'xcenter_1_0.999999_n0_0', 'xcenter_1_0.999999_n1_0', 'xcenter_1_0.999999_n2_0', 'xcenter_1_0.999999_n3_0', 'xcenter_1_0.999999_n4_0',
             'xcenter_1_0.99999_n0_0', 'xcenter_1_0.99999_n1_0', 'xcenter_1_0.99999_n2_0', 'xcenter_1_0.99999_n3_0', 'xcenter_1_0.99999_n4_0', 'xcenter_1_0.9999_n0_0', 'xcenter_1_0.9999_n1_0', 'xcenter_1_0.9999_n2_0',
             'xcenter_1_0.9999_n3_0', 'xcenter_1_0.9999_n4_0', 'xcenter_1_0.999_n0_0', 'xcenter_1_0.999_n1_0', 'xcenter_1_0.999_n2_0', 'xcenter_1_0.999_n3_0', 'xcenter_1_0.999_n4_0', 'xcenter_1_0.99_n0_0', 'xcenter_1_0.99_n1_0',
             'xcenter_1_0.99_n2_0', 'xcenter_1_0.99_n3_0', 'xcenter_1_0.99_n4_0', 'xcenter_1_0.9_n0_0', 'xcenter_1_0.9_n1_0', 'xcenter_1_0.9_n2_0', 'xcenter_1_0.9_n3_0', 'xcenter_1_0.9_n4_0', 'xcenter_2_0.5_n0_0',
             'xcenter_2_0.5_n1_0', 'xcenter_2_0.5_n2_0', 'xcenter_2_0.5_n3_0', 'xcenter_2_0.5_n4_0', 'xcenter_2_0.75_n0_0', 'xcenter_2_0.75_n1_0', 'xcenter_2_0.75_n2_0', 'xcenter_2_0.75_n3_0', 'xcenter_2_0.75_n4_0',
             'xcenter_2_0.95_n0_0', 'xcenter_2_0.95_n1_0', 'xcenter_2_0.95_n2_0', 'xcenter_2_0.95_n3_0', 'xcenter_2_0.95_n4_0', 'xcenter_2_0.98_n0_0', 'xcenter_2_0.98_n1_0', 'xcenter_2_0.98_n2_0', 'xcenter_2_0.98_n3_0',
             'xcenter_2_0.98_n4_0', 'xcenter_2_0.9999999_n0_0', 'xcenter_2_0.9999999_n1_0', 'xcenter_2_0.9999999_n2_0', 'xcenter_2_0.9999999_n3_0', 'xcenter_2_0.9999999_n4_0', 'xcenter_2_0.999999_n0_0', 'xcenter_2_0.999999_n1_0',
             'xcenter_2_0.999999_n2_0', 'xcenter_2_0.999999_n3_0', 'xcenter_2_0.999999_n4_0', 'xcenter_2_0.99999_n0_0', 'xcenter_2_0.99999_n1_0', 'xcenter_2_0.99999_n2_0', 'xcenter_2_0.99999_n3_0', 'xcenter_2_0.99999_n4_0',
             'xcenter_2_0.9999_n0_0', 'xcenter_2_0.9999_n1_0', 'xcenter_2_0.9999_n2_0', 'xcenter_2_0.9999_n3_0', 'xcenter_2_0.9999_n4_0', 'xcenter_2_0.999_n0_0', 'xcenter_2_0.999_n1_0', 'xcenter_2_0.999_n2_0',
             'xcenter_2_0.999_n3_0', 'xcenter_2_0.999_n4_0', 'xcenter_2_0.99_n0_0', 'xcenter_2_0.99_n1_0', 'xcenter_2_0.99_n2_0', 'xcenter_2_0.99_n3_0', 'xcenter_2_0.99_n4_0', 'xcenter_2_0.9_n0_0', 'xcenter_2_0.9_n1_0',
             'xcenter_2_0.9_n2_0', 'xcenter_2_0.9_n3_0', 'xcenter_2_0.9_n4_0', 'xshape', 'yshape', 'zshape'}

        feats2 = feats1.drop(feats_not_in_0311_version, 1)
        fname2 ='feats_keras_0311_%s_%s_%s.csv'% (stage, len(feats2), date_version) 
        feats2.to_csv(fname2, index=True)
        print ("OVERALL feature file and process: ", stage, fname2, time.time()-start_time)
        
