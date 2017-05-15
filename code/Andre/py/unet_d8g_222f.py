"""
Created on Thu Jan 26 17:04:11 2017

@author: Andre Stochniol, andre@stochniol.com
"""

#%matplotlib inline

import numpy as np 
import pandas as pd
import dicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import scipy.ndimage # added for scaling

import cv2
import time
import glob


from skimage import measure, morphology, segmentation
import SimpleITK as sitk


RESIZE_SPACING = [2,2,2]  # z, y, x  (x & y MUST be the same)
RESOLUTION_STR = "2x2x2"

img_rows = 448 
img_cols = 448 # global values

DO_NOT_USE_SEGMENTED = True

#STAGE = "stage1"

STAGE_DIR_BASE = "../input/%s/"  # on one cluster we had input_shared

LUNA_MASKS_DIR = "../luna/data/original_lung_masks/"
luna_subset = 0       # initial 
LUNA_BASE_DIR = "../luna/data/original_lungs/subset%s/"  # added on AWS; data as well 
LUNA_DIR = LUNA_BASE_DIR % luna_subset

CSVFILES = "../luna/data/original_lungs/CSVFILES/%s"
LUNA_ANNOTATIONS = CSVFILES % "annotations.csv"
LUNA_CANDIDATES =  CSVFILES % "candidates.csv"



# Load the scans in given folder path (loads the most recent acquisition)
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    #slices.sort(key = lambda x: int(x.InstanceNumber))
       
    acquisitions = [x.AcquisitionNumber for x in slices]
    
    vals, counts = np.unique(acquisitions, return_counts=True)
    vals = vals[::-1]  # reverse order so the later acquisitions are first (the np.uniques seems to always return the ordered 1 2 etc.
    counts = counts[::-1]
    
    ## take the acquistions that has more entries; if these are identical take the later  entrye
    acq_val_sel = vals[np.argmax(counts)]
  

    ##acquisitions = sorted(np.unique(acquisitions), reverse=True)
    
    if len(vals) > 1:
        print ("WARNING ##########: MULTIPLE acquisitions & counts, acq_val_sel, path: ", vals, counts, acq_val_sel, path)
    slices2= [x for x in slices if x.AcquisitionNumber == acq_val_sel]
    
    slices = slices2
    
   
    ## ONE path includes 2 acquisitions (2 sets), take the latter acquiisiton only whihch cyupically is better than the first/previous ones.
    ## example of the     '../input/stage1/b8bb02d229361a623a4dc57aa0e5c485'
    
    #slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))  # from v 8, BUG should be float
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))  # from v 9
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_3d_data_slices(slices):  # get data in Hunsfield Units
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))  # from v 9
    
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
    
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    ### slope can differ per slice -- so do it individually (case in point black_tset, slices 95 vs 96)
    ### Changes/correction - 31.01.2017
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)    
    
    return np.array(image, dtype=np.int16)

MARKER_INTERNAL_THRESH = -400  
MARKER_FRAME_WIDTH = 9      # 9 seems OK for the half special case ...
def generate_markers(image):
    #Creation of the internal Marker
    
    useTestPlot = False
    if useTestPlot:
        timg = image
        plt.imshow(timg, cmap='gray')
        plt.show()

    add_frame_vertical = True  
    if add_frame_vertical:   # add frame for potentially closing the lungs that touch the edge, but only vertically
                        
        fw = MARKER_FRAME_WIDTH  # frame width (it looks that 2 is the minimum width for the algorithms implemented here, namely the first 2 operations for the marker_internal)
   
        xdim = image.shape[1]
        #ydim = image.shape[0]
        img2 = np.copy(image)
        #y3 = ydim // 3
        
        img2 [:, 0]    = -1024
        img2 [:, 1:fw] = 0

        img2 [:, xdim-1:xdim]    = -1024
        img2 [:, xdim-fw:xdim-1] = 0
               
        marker_internal = img2 < MARKER_INTERNAL_THRESH  
    else:
        marker_internal = image < MARKER_INTERNAL_THRESH  # was -400
        
    
    useTestPlot = False
    if useTestPlot:
        timg = marker_internal
        plt.imshow(timg, cmap='gray')
        plt.show()
       
    
    correct_edges2 = False  ## NOT a good idea - no added value
    if correct_edges2:   
        marker_internal[0,:]   = 0
        marker_internal[:,0]   = 0
        #marker_internal[:,1]   = True
        #marker_internal[:,2]   = True
        marker_internal[511,:]   = 0
        marker_internal[:,511]   = 0
    
    marker_internal = segmentation.clear_border(marker_internal, buffer_size=0)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)  # was 10
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)   # was 55 
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    #marker_watershed = np.zeros((512, 512), dtype=np.int)  # origi
    marker_watershed = np.zeros((marker_external.shape), dtype=np.int)
    
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed

# Some of the starting Code is taken from ArnavJain, since it's more readable then my own

def generate_markers_3d(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal_labels = np.zeros(image.shape).astype(np.int16)
    for i in range(marker_internal.shape[0]):
        marker_internal[i] = segmentation.clear_border(marker_internal[i])
        marker_internal_labels[i] = measure.label(marker_internal[i])
    #areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas = [r.area for i in range(marker_internal.shape[0]) for r in measure.regionprops(marker_internal_labels[i])]
    for i in range(marker_internal.shape[0]):
        areas = [r.area for r in measure.regionprops(marker_internal_labels[i])]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels[i]):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                           marker_internal_labels[i, coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    
    # 3x3 structuring element with connectivity 1, used by default
    struct1 = ndimage.generate_binary_structure(2, 1)
    struct1 = struct1[np.newaxis,:,:]  # expand by z axis .
    
    external_a = ndimage.binary_dilation(marker_internal, structure=struct1, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, structure=struct1, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    #marker_watershed = np.zeros((512, 512), dtype=np.int)  # origi
    marker_watershed = np.zeros((marker_external.shape), dtype=np.int)
    
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed


BINARY_CLOSING_SIZE = 7  #was 7 before final;   5 for disk seems sufficient - for safety let's go with 6 or even 7
def seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    ##structure = np.ones((BINARY_CLOSING_SIZE,BINARY_CLOSING_SIZE)) # 5 is not enough, 7 is
    structure = morphology.disk(BINARY_CLOSING_SIZE) # better , 5 seems sufficient, we use 7 for safety/just in case
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=structure, iterations=3) #, iterations=3)  # was structure=np.ones((5,5))
    ### NOTE if no iterattions, i.e. default 1 we get holes within lungs for the disk(5) and perhaps more
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))  ### was -2000
    
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

def rescale_n(n,reduce_factor):
    return max( 1, int(round(n / reduce_factor)))

def seperate_lungs_cv2(image):      # for increased speed
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #image_size = image.shape[0]
    reduce_factor = 512 / image.shape[0]
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    useTestPlot = False
    if useTestPlot:
        timg = sobel_gradient
        plt.imshow(timg, cmap='gray')
        plt.show()
        
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    if useTestPlot:
        timg = marker_external
        plt.imshow(timg, cmap='gray')
        plt.show()    
    
    #Reducing the image created by the Watershed algorithm to its outline
    #wsize = rescale_n(3,reduce_factor)  # THIS IS TOO SMALL, dynamically adjusting the size for the watersehed algorithm
    outline = ndimage.morphological_gradient(watershed, size=(3,3))   # original (3,3), (wsize, wsize) is too small to create an outline
    outline = outline.astype(bool)
    outline_u = outline.astype(np.uint8)  #added
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    

    use_reduce_factor = True
    if use_reduce_factor:
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, rescale_n(8,reduce_factor))  # dyanmically adjust the number of iterattions; original was 8
    else:
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8) 
    
    blackhat_struct_cv2 = blackhat_struct.astype(np.uint8)
    #Perform the Black-Hat
    
    #outline += ndimage.black_tophat(outline, structure=blackhat_struct)  # original slow 

    #outline1 = outline + (cv2.morphologyEx(outline_u, cv2.MORPH_BLACKHAT, kernel=blackhat_struct_cv2)).astype(np.bool)
    #outline2 = outline + ndimage.black_tophat(outline, structure=blackhat_struct)
    #np.array_equal(outline1,outline2)  # True

    outline += (cv2.morphologyEx(outline_u, cv2.MORPH_BLACKHAT, kernel=blackhat_struct_cv2)).astype(np.bool)  # fats


    if useTestPlot:
        timg = outline
        plt.imshow(timg, cmap='gray')
        plt.show()
        

    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    
    if useTestPlot:
        timg = lungfilter
        plt.imshow(timg, cmap='gray')
        plt.show()
    
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    ##structure = np.ones((BINARY_CLOSING_SIZE,BINARY_CLOSING_SIZE)) # 5 is not enough, 7 is
    structure2 = morphology.disk(2)  # used to fill the gaos/holes close to the border (otherwise the large sttructure would create a gap by the edge)
    if use_reduce_factor:
        structure3 = morphology.disk(rescale_n(BINARY_CLOSING_SIZE,reduce_factor)) # dynanically adjust; better , 5 seems sufficient, we use 7 for safety/just in case
    else:
        structure3 = morphology.disk(BINARY_CLOSING_SIZE) # dynanically adjust; better , 5 seems sufficient, we use 7 for safety/just in case
    
    
    ##lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=structure, iterations=3) #, ORIGINAL iterations=3)  # was structure=np.ones((5,5))
    lungfilter2 = ndimage.morphology.binary_closing(lungfilter, structure=structure2, iterations=3)  # ADDED
    lungfilter3 = ndimage.morphology.binary_closing(lungfilter, structure=structure3, iterations=3)
    lungfilter = np.bitwise_or(lungfilter2, lungfilter3)
    
    ### NOTE if no iterattions, i.e. default 1 we get holes within lungs for the disk(5) and perhaps more
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    #image.shape
    #segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)).astype(np.int16)) # was -2000 someone suggested 30
    segmented = np.where(lungfilter == 1, image, -2000*np.ones(image.shape).astype(np.int16)) # was -2000 someone suggested 30
    
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

def seperate_lungs_3d(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers_3d(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, axis=2)
    sobel_filtered_dy = ndimage.sobel(image, axis=1)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(1,3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    
    blackhat_struct = blackhat_struct[np.newaxis,:,:]
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)   # very long time
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    ##structure = np.ones((BINARY_CLOSING_SIZE,BINARY_CLOSING_SIZE)) # 5 is not enough, 7 is
    structure = morphology.disk(BINARY_CLOSING_SIZE) # better , 5 seems sufficient, we use 7 for safety/just in case
    structure = structure[np.newaxis,:,:]
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=structure, iterations=3) #, iterations=3)  # was structure=np.ones((5,5))
    ### NOTE if no iterattions, i.e. default 1 we get holes within lungs for the disk(5) and perhaps more
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones(marker_internal.shape))
    
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed
    

def get_slice_location(dcm):
    return float(dcm[0x0020, 0x1041].value)

def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos
  
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    
    #scan[2].SliceThickness


    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')  ### early orig modified 
    
    return image, new_spacing


def segment_all(stage, part=0, processors=1, showSummaryPlot=True):  # stage added to simplify the stage1 and stage2 calculations

    count = 0
    STAGE_DIR = STAGE_DIR_BASE % stage
    folders = glob.glob(''.join([STAGE_DIR,'*']))
    if len(folders) == 0:
        print ("ERROR, check directory, no folders found in: ", STAGE_DIR )
        
    for folder in folders:
        
        count += 1
        if count % processors == part:    # do this part in this process, otherwise skip
            path = folder
            
            slices = load_scan(path)
            image_slices = get_3d_data_slices(slices)
            #mid = len(image_slices) // 2
            #img_sel = mid
            
            useTestPlot = False
            if useTestPlot:
                print("Shape before segmenting\t", image_slices.shape)
                plt.hist(image_slices.flatten(), bins=80, color='c')
                plt.xlabel("Hounsfield Units (HU)")
                plt.ylabel("Frequency")
                plt.show()


            start = time.time()
            resampleImages = True
            if resampleImages:
                image_resampled, spacing = resample(image_slices, slices, RESIZE_SPACING)   # let's start wkith this small resolutuion for workign our the system (then perhaps 2, 0.667, 0.667)
                print("Shape_before_&_after_resampling\t", image_slices.shape,image_resampled.shape)
                if useTestPlot:
                    plt.imshow(image_slices[image_slices.shape[0]//2], cmap=plt.cm.bone)
                    plt.show() 
                    plt.imshow(image_resampled[image_resampled.shape[0]//2], cmap=plt.cm.bone)
                    np.max(image_slices)     
                    np.max(image_resampled)
                    
                    np.min(image_slices)     
                    np.min(image_resampled)   
                    plt.show() 
                image_slices = image_resampled
                        
            shape = image_slices.shape
            l_segmented = np.zeros(shape).astype(np.int16)
            l_lungfilter = np.zeros(shape).astype(np.bool)
            l_outline = np.zeros(shape).astype(np.bool)
            l_watershed = np.zeros(shape).astype(np.int16)
            l_sobel_gradient = np.zeros(shape).astype(np.float32)
            l_marker_internal = np.zeros(shape).astype(np.bool)
            l_marker_external = np.zeros(shape).astype(np.bool)
            l_marker_watershed = np.zeros(shape).astype(np.int16)  
            
            # start = time.time()
            i=0
            for i in range(shape[0]):
                l_segmented[i], l_lungfilter[i], l_outline[i], l_watershed[i], l_sobel_gradient[i], l_marker_internal[i], l_marker_external[i], l_marker_watershed[i] = seperate_lungs_cv2(image_slices[i])
            print("Rescale & Seg time, and path: ", ((time.time() - start)), path )
                
            
            if useTestPlot:
                plt.hist(image_slices.flatten(), bins=80, color='c')
                plt.xlabel("Hounsfield Units (HU)")
                plt.ylabel("Frequency")
                plt.show()
            
            
                plt.hist(l_segmented.flatten(), bins=80, color='c')
                plt.xlabel("Hounsfield Units (HU)")
                plt.ylabel("Frequency")
                plt.show()
                
                img_sel_i = shape[0] // 2
                # Show some slice in the middle
                plt.imshow(image_slices[img_sel_i], cmap=plt.cm.gray)
                plt.show()
                
                # Show some slice in the middle
                plt.imshow(l_segmented[img_sel_i], cmap='gray')
                plt.show()
            
            path_rescaled = path.replace(stage, ''.join([stage, "_", RESOLUTION_STR]), 1)
            path_segmented = path.replace(stage, ''.join([stage, "_segmented_", RESOLUTION_STR]), 1)
            path_segmented_crop = path.replace(stage, ''.join([stage, "_segmented_", RESOLUTION_STR, "_crop"]), 1)
         
            np.savez_compressed (path_rescaled, image_slices)
            np.savez_compressed (path_segmented, l_segmented)
            
            mask = l_lungfilter.astype(np.int8)
            
            regions = measure.regionprops(mask)  # this measures the largest region and is a bug when the mask is not the largest region !!!

     
            bb = regions[0].bbox
            #print(bb)
            zlen = bb[3] - bb[0]
            ylen = bb[4] - bb[1]
            xlen = bb[5] - bb[2]
            
            dx = 0  # could  be reduced
            ## have to reduce dx as for istance at least image the lungs stretch right to the border evebn without cropping 
            ## namely for '../input/stage1/be57c648eb683a31e8499e278a89c5a0'
            
            crop_max_ratio_z = 0.6  # 0.8 is to big    make_submit2(45, 1)
            crop_max_ratio_y = 0.4
            crop_max_ratio_x = 0.6
            
            bxy_min = np.min(bb[1:3]) 
            bxy_max = np.max(bb[4:6])
            mask_shape= mask.shape
            image_shape = l_segmented.shape
            
            mask_volume = zlen*ylen*zlen /(mask_shape[0] * mask_shape[1] * mask_shape[2])
            mask_volume_thresh = 0.08  # anything below is too small (maybe just one half of the lung or something very small0)
            mask_volume_check =   mask_volume >   mask_volume_thresh
            # print ("Mask Volume: ", mask_volume )    
            
            ### DO NOT allow the mask to touch x & y ---> if it does it is likely a wrong one as for:
            ## folders[3] , path = '../input/stage1/9ba5fbcccfbc9e08edcfe2258ddf7
            
            maskOK = False
            if bxy_min >0 and bxy_max < 512 and mask_volume_check and zlen/mask_shape[0] > crop_max_ratio_z and ylen/mask_shape[1] > crop_max_ratio_y and xlen/mask_shape[2]  > crop_max_ratio_x:
                
                ## square crop and at least dx elements on both sides on x & y
                bxy_min = np.min(bb[1:3]) 
                bxy_max = np.max(bb[4:6])
                
                if bxy_min == 0 or bxy_max == 512:
                    # Mask to bigg, auto-correct
                    print("The following mask likely too big, autoreducing by:", dx)
                    
                    bxy_min = np.max((bxy_min, dx)) 
                    bxy_max = np.min ((bxy_max, mask_shape[1] - dx))
                
                image = l_segmented[bb[0]:bb[3], bxy_min:bxy_max, bxy_min:bxy_max]
                mask =   mask[bb[0]:bb[3], bxy_min:bxy_max, bxy_min:bxy_max]
                #maskOK = True
         
                print ("Shape, cropped, bbox ", mask_shape, mask.shape, bb)
        
            elif bxy_min> 0 and bxy_max < 512 and mask_volume_check and zlen/mask.shape[0] > crop_max_ratio_z:
                ## cut on z at least
            
                image = l_segmented[bb[0]:bb[3], dx: image_shape[1] - dx, dx: image_shape[2] - dx]
                #mask =   mask[bb[0]:bb[3], dx: mask_shape[1] - dx, dx: mask_shape[2] - dx]
                print("Mask too small, NOT auto-cropping x-y: shape, cropped, bbox, ratios, violume:", mask_shape, image.shape, bb, path, zlen/mask_shape[0], ylen/mask_shape[1], xlen/mask_shape[2], mask_volume)
        
        
            else:
                image = l_segmented[0:mask_shape[0], dx: image_shape[1] - dx, dx: image_shape[2] - dx]
                #mask =   mask[0:mask_shape[0], dx: mask_shape[1] - dx, dx: mask_shape[2] - dx]
                print("Mask wrong, NOT auto-cropping: shape, cropped, bbox, ratios, volume:", mask_shape, image.shape, bb, path, zlen/mask_shape[0], ylen/mask_shape[1], xlen/mask_shape[2], mask_volume)
                
            
            if showSummaryPlot:
                img_sel_i = shape[0] // 2
                # Show some slice in the middle
                useSeparatePlots = False
                if useSeparatePlots:
                    plt.imshow(image_slices[img_sel_i], cmap=plt.cm.gray)
                    plt.show()
                    
                    # Show some slice in the middle
                    plt.imshow(l_segmented[img_sel_i], cmap='gray')
                    plt.show()
                else:                     
                    f, ax = plt.subplots(1, 2, figsize=(6,3))
                    ax[0].imshow(image_slices[img_sel_i],cmap=plt.cm.bone)
                    ax[1].imshow(l_segmented[img_sel_i],cmap=plt.cm.bone)
                    plt.show()   
                # Show some slice in the middle
                #plt.imshow(image[image.shape[0] // 2], cmap='gray')  # don't show it for simpler review 
                #plt.show()
                
            
            np.savez_compressed(path_segmented_crop, image)
            #print("Mask count: ", count)
            #print ("Shape: ", image.shape)
    return part, processors, count


# the following 3 functions to read LUNA files are from: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def seq(start, stop, step=1):
	n = int(round((stop - start)/float(step)))
	if n > 1:
		return([start + step*i for i in range(n+1)])
	else:
		return([])

'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''

#image = lung_img
#spacing = new_spacing
def draw_circles(image,cands,origin,spacing):
	#make empty matrix, which will be filled with the mask
	image_mask = np.zeros(image.shape, dtype=np.int16)

	#run over all the nodules in the lungs
	for ca in cands.values:
		#get middel x-,y-, and z-worldcoordinate of the nodule
		#radius = np.ceil(ca[4])/2     ## original:  replaced the ceil with a very minor increase of 1% ....
		radius = (ca[4])/2 + 0.51 * spacing[0]  # increasing by circa half of distance in z direction .... (trying to capture wider region/border for learning ... and adress the rough net .
    
		coord_x = ca[1]
		coord_y = ca[2]
		coord_z = ca[3]
		image_coord = np.array((coord_z,coord_y,coord_x))

		#determine voxel coordinate given the worldcoordinate
		image_coord = world_2_voxel(image_coord,origin,spacing)

		#determine the range of the nodule
		#noduleRange = seq(-radius, radius, RESIZE_SPACING[0])  # original, uniform spacing 
		noduleRange_z = seq(-radius, radius, spacing[0])
		noduleRange_y = seq(-radius, radius, spacing[1])
		noduleRange_x = seq(-radius, radius, spacing[2])

          #x = y = z = -2
		#create the mask
		for x in noduleRange_x:
			for y in noduleRange_y:
				for z in noduleRange_z:
					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
					#if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:  ### original (contrained to a uniofrm RESIZE)
					if (np.linalg.norm((image_coord-coords) * spacing)) < radius:
						image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
	

	return image_mask

'''
This function takes the path to a '.mhd' file as input and 
is used to create the nodule masks and segmented lungs after 
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as 
input.
'''
   
def load_scans_masks(luna_subset, useAll, use_unsegmented=True):
    
    #luna_subset = "[0-6]"
    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))

    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()

    sids = []
    scans = []
    masks = []
    cnt = 0
    skipped = 0
    for file in files:
        imagePath = file
        seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
        seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid
        
        path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
        if use_unsegmented:
            path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)
        else:
            path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)

        cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
        
        #useAll = True
        if (len(cands) > 0 or useAll):
            sids.append(seriesuid)  
            
            if use_unsegmented:
                scan_z = np.load(''.join((path_segmented  + '_lung' + '.npz'))) 
            else:
                scan_z = np.load(''.join((path_segmented  + '_lung_seg' + '.npz'))) 
            #scan_z.keys()
            scan = scan_z['arr_0']
            mask_z = np.load(''.join((path_segmented  + '_nodule_mask' + '.npz'))) 
            mask = mask_z['arr_0']    
            
            scans.append(scan)
            masks.append(mask)       
            cnt += 1
        else:
            print("Skipping non-nodules entry ", seriesuid)
            skipped += 1
            
            
    print ("Summary: cnt & skipped: ", cnt, skipped)
    
    return scans, masks, sids

def load_scans_masks_or_blanks(luna_subset, useAll, use_unsegmented=True):
    
    #luna_subset = "[0-6]"
    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))

    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()

    candidates =    pd.read_csv(LUNA_CANDIDATES)
    candidates_false = candidates[candidates["class"] == 0]  # only select the false candidates
    candidates_true = candidates[candidates["class"] == 1]  # only select the false candidates



    sids = []
    scans = []
    masks = []
    blankids = []  # class/id whether scan is with nodule or without, 0 - with, 1 - without 
    cnt = 0
    skipped = 0
    #file=files[7]
    for file in files:    
        imagePath = file
        seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
        seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid
        
        path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
        if use_unsegmented:
            path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)
        else:
            path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)

        cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
        ctrue = candidates_true[seriesuid == candidates_true.seriesuid] 
        cfalse = candidates_false[seriesuid == candidates_false.seriesuid] 
        
        #useAll = True
        

        blankid = 1 if (len(cands) == 0 and len(ctrue) == 0 and len(cfalse) > 0) else 0
        
        skip_nodules_entirely = False  # was False
        use_only_nodules = False  # was True
        if skip_nodules_entirely and blankid ==0:
            ## manual switch to generate extra data for the corrupted set
            print("Skipping nodules  (skip_nodules_entirely) ", seriesuid)
            skipped += 1
            
        elif use_only_nodules and (len(cands) == 0):
            ## manual switch to generate only nodules data due lack of time and repeat etc time pressures
            print("Skipping blanks  (use_only_nodules) ", seriesuid)
            skipped += 1
        else:  # NORMAL operations
             if (len(cands) > 0 or 
                    (blankid >0) or
                    useAll):
                sids.append(seriesuid)  
                blankids.append(blankid)  
                
                
                if use_unsegmented:
                    scan_z = np.load(''.join((path_segmented  + '_lung' + '.npz'))) 
                else:
                    scan_z = np.load(''.join((path_segmented  + '_lung_seg' + '.npz'))) 
                #scan_z.keys()
                scan = scan_z['arr_0']
                #mask_z = np.load(''.join((path_segmented  + '_nodule_mask' + '.npz'))) 
                mask_z = np.load(''.join((path_segmented  + '_nodule_mask_wblanks' + '.npz'))) 
                mask = mask_z['arr_0']   
                
                testPlot = False
                if testPlot:
                    maskcheck_z = np.load(''.join((path_segmented  + '_nodule_mask' + '.npz'))) 
                    maskcheck = maskcheck_z['arr_0']
                    
                    f, ax = plt.subplots(1, 2, figsize=(10,5))
                    ax[0].imshow(np.sum(np.abs(maskcheck), axis=0),cmap=plt.cm.gray)
                    ax[1].imshow(np.sum(np.abs(mask), axis=0),cmap=plt.cm.gray)
                    #ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                    plt.show()
                    
                scans.append(scan)
                masks.append(mask)       
                cnt += 1
             else:
                print("Skipping non-nodules and non-blank entry ", seriesuid)
                skipped += 1
            
            
    print ("Summary: cnt & skipped: ", cnt, skipped)
    
    return scans, masks, sids, blankids
    #return scans, masks, sids   # not yet, old style


def load_scans_masks_no_nodules(luna_subset, use_unsegmented=True):  # load only the ones that do not contain nodules
    
    #luna_subset = "[0-6]"
    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))

    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()

    sids = []
    scans = []
    masks = []
    cnt = 0
    skipped = 0
    for file in files:
        imagePath = file
        seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
        seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid
        
        path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
        if use_unsegmented:
            path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)
        else:
            path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)

        cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
        
        #useAll = True
        if (len(cands)):
            print("Skipping entry with nodules ", seriesuid)
            skipped += 1
        else:
            sids.append(seriesuid)  
            if use_unsegmented:
                scan_z = np.load(''.join((path_segmented  + '_lung' + '.npz'))) 
            else:
                scan_z = np.load(''.join((path_segmented  + '_lung_seg' + '.npz'))) 

            #scan_z.keys()
            scan = scan_z['arr_0']
            mask_z = np.load(''.join((path_segmented  + '_nodule_mask' + '.npz'))) 
            mask = mask_z['arr_0']    
            
            scans.append(scan)
            masks.append(mask)       
            cnt += 1
            
    print ("Summary: cnt & skipped: ", cnt, skipped)
    
    return scans, masks, sids


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.028  ## for LUNA subset 0 and our preprocessing, only with nudels was 0.028, all was 0.020421744071562546 (in the tutorial they used 0.25)

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def load_scans(path):  # function used for testing
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_scans(df,scans_list):
    scans=np.stack([load_scans(scan_folder+df.id[i_scan[0]])[i_scan[1]] for i_scan in scans_list])
    scans=process_scans(scans)
    view_scans(scans)
    return(scans)

def process_scans(scans):  # used for tesing
    scans1=np.zeros((scans.shape[0],1,img_rows,img_cols))
    for i in range(scans.shape[0]):
        img=scans[i,:,:]
        img = 255.0 / np.amax(img) * img
        img =img.astype(np.uint8)
        img =cv2.resize(img, (img_rows, img_cols))
        scans1[i,0,:,:]=img
    return (scans1)


only_with_nudels = True
def convert_scans_and_masks(scans, masks, only_with_nudels):
    
    flattened1 = [val for sublist in scans for val in sublist[1:-1]]  # skip one element at the beginning and at the end
    scans1 = np.stack(flattened1)
    
    flattened1 = [val for sublist in masks for val in sublist[1:-1]]  # skip one element at the beginning and at the end
    masks1 = np.stack(flattened1)  # 10187
    
    
    #only_with_nudels = True
    if only_with_nudels:    
        nudels_pix_count = np.sum(masks1, axis = (1,2))  
        scans1 = scans1[nudels_pix_count>0]  
        masks1 = masks1[nudels_pix_count>0] # 493 -- circa 5 % with nudeles oters without
    
    
    #nudels2 =  np.where(masks1 == 1, scans1, -4000*np.ones(( masks1.shape[1], masks1.shape[2)))  ### was -2000
    #nudels1 =  np.where(masks1 == 1, scans1, masks1 - 4000)  ### was -2000

    #nudles1_rf = nudels1.flatten()
    #nudles1_rf = nudles1_rf[nudles1_rf > -4000]
    
    scans = normalize(scans1)
    
    useTestPlot = False
    if useTestPlot:
        
        plt.hist(scans1.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        #for i in range(scans.shape[0]):
        for i in range(20):
            print ('scan '+str(i))
            f, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(scans1[i,:,:],cmap=plt.cm.gray)
            ax[1].imshow(scans[i,:,:],cmap=plt.cm.gray)
            ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
            plt.show()
        

    #np.mean(scans)  # 0.028367 / 0.0204    
    #np.min(scans)   # 0
    #np.max(scans)    # 

    scans = zero_center(scans)
    masks = np.copy(masks1)
       
       
    ## if needed do the resize here ....
    img_rows = scans.shape[1]  ### redefine img_rows/ cols and add resize if needed
    img_cols = scans.shape[2]
    
    scans1=np.zeros((scans.shape[0],1,img_rows,img_cols))
    for i in range(scans.shape[0]):
        img=scans[i,:,:]
        ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
        scans1[i,0,:,:]=img
        
    masks1=np.zeros((masks.shape[0],1,img_rows,img_cols))
    for i in range(masks.shape[0]):
        img=masks[i,:,:]
        ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
        masks1[i,0,:,:]=img
        
    return scans1, masks1




#scans = [scans[i]]
#masks = [masks[i]]



def convert_scans_and_masks_xd_ablanks(scans, masks, blankids, only_with_nudels, dim=3):
    
    

    # reuse scan to reduce memory footprint
    dim_orig = dim    
    add_blank_spacing_size = dim * 8     #### use 4 for [0 - 3] and 8 for [4 - 7] ???initial trial (should perhaps be just dim ....)
    #skip = dim // 2    # old 
    skip_low = dim // 2  # dim shoudl be uneven -- it is recalculated anyway to this end
    skip_high = dim -skip_low - 1
    do_not_allow_even_dim = False ## now we allow odd numbers ...
    if do_not_allow_even_dim:
        dim = 2 * skip_low + 1
        skip_low = dim // 2
        skip_high = dim -skip_low - 1
        
        if dim != dim_orig:
            print ("convert_scans_and_masks_x: Dim must be uneven, corrected from .. to:", dim_orig, dim)
    work = []     # 3 layers
    #scan = scans[0]
    for scan in scans:  ##TEMP
        tmp = []
        #i = 1
        #for i in range(1, scan.shape[0]-1, 3):  # SKIP EVERY 3
        for i in range(skip_low, scan.shape[0]-skip_high):  
            #img1 = scan[i-1]
            #img2 = scan[i]
            #img3 = scan[i+1]
            #rgb = np.stack((img1, img2, img3))
            rgb = np.stack(scan[i-skip_low:i+skip_high+1])
            tmp.append(rgb)
        work.append(np.array(tmp))      

    #flattened1 = [val for sublist in work for val in sublist ] # NO skipping as we have already cut the first and the last layer    
    #scans1 = np.stack(flattened1)
    scans1 =  np.stack([val for sublist in work for val in sublist ]) # NO skipping as we have already cut the first and the last layer     
    work = []
    
    ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
    # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
    blanks_per_axis = 4 # skip border
    crop = 16
    dx = (img_cols - 2 * crop) // (blanks_per_axis + 2)
    dy =  (img_rows - 2 * crop) // (blanks_per_axis + 2)
    
    for mask in masks:
        if (np.sum(mask) < 0):
            ## we have a blank 
            ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
            # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
            for i in range(skip_low, mask.shape[0]-skip_high, add_blank_spacing_size):  
                for ix in range(blanks_per_axis):
                    xpos = crop + (ix+1)*dx + dx //2
                    for iy in range(blanks_per_axis):
                        ypos = crop + (iy+1)*dy + dy //2
                        #print (xpos, ypos)                    
                        mask[skip_low, ypos, xpos] = -1  # negative pixel to be picked up below and corrected back to none
    


    #for k in range(len(blankids)):
    #    if blankids[k] > 0:
    #        mask = masks[k]
    #        ## add the blanls
    #        for i in range(skip_low, mask.shape[0]-skip_high, add_blank_spacing_size):  
    #            mask[skip_low, 0, 0] = -1  # negative pixel to be picked up below and corrected back to none

    
    use_3d_mask = True  ## 
    if use_3d_mask:
        work = []     # 3 layers
        #mask = masks[0]
        
        for mask in masks:
            tmp = []
            #i = 0
            for i in range(skip_low, mask.shape[0]-skip_high):  
                #img1 = mask[i-1]
                #img2 = mask[i]
                #img3 = mask[i+1]
                #rgb = np.stack((img1, img2, img3))
                rgb = np.stack(mask[i-skip_low:i+skip_high+1])
                tmp.append(rgb)
            work.append(np.array(tmp))      
    
        masks1 = np.stack([val for sublist in work for val in sublist ] )# NO skipping as we have already cut the first and the last layer    
    else:
        masks1 = np.stack([val for sublist in masks for val in sublist[skip_low:-skip_high]] ) # skip one element at the beginning and at the end
        #masks1 = np.stack(flattened1)  # 10187
        
    
    #only_with_nudels = True
    if only_with_nudels:  
        if use_3d_mask:
            nudels_pix_count = np.sum(masks1[:,skip_low], axis = (1,2))  ## abd added for the potential blanks; modified that the centre mask be mask!
        else:
            nudels_pix_count = np.sum(masks1, axis = (1,2)) 
        scans1 = scans1[nudels_pix_count != 0]  
        masks1 = masks1[nudels_pix_count != 0]  
        
        #blank_mask_factor = np.sign(nudels_pix_count)[nudels_pix_count != 0]
        #sum(blank_mask_factor)
        #blank_mask_factor[blank_mask_factor <0] = 0
        #mask1_orig = masks1
        #np.sum(mask1_orig)
        #np.min(masks1)
        #masks1 = masks1[nudels_pix_count != 0] * blank_mask_factor # 493 -- circa 5 % with nudeles oters without; 232 if we skip over every 3 layers and use a 3d mask
        masks1[masks1 < 0]  = 0 # 493 -- circa 5 % with nudeles oters without; 232 if we skip over every 3 layers and use a 3d mask
        
        #masks1[nudels_pix_count < 0] = 0  # making empty mask for balancing training set
    
    #nudels2 =  np.where(masks1 == 1, scans1, -4000*np.ones(( masks1.shape[1], masks1.shape[2)))  ### was -2000
    #nudels1 =  np.where(masks1 == 1, scans1, masks1 - 4000)  ### was -2000

    #nudles1_rf = nudels1.flatten()
    #nudles1_rf = nudles1_rf[nudles1_rf > -4000]
    
    scans1 = normalize(scans1)
    
    useTestPlot = False
    if useTestPlot:
        
        plt.hist(scans1.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        #for i in range(scans.shape[0]):
        for i in range(20):
            print ('scan '+str(i))
            f, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(scans1[i,:,:],cmap=plt.cm.gray)
            ax[1].imshow(scans[i,:,:],cmap=plt.cm.gray)
            ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
            plt.show()
        

    #np.mean(scans)  # 0.028367 / 0.0204    
    #np.min(scans)   # 0
    #np.max(scans)    # 

    scans1 = zero_center(scans1)
    #masks = np.copy(masks1)
       
       
    ## if needed do the resize here .... (img_rows and img_cols are global values defined externally)
    #img_rows = scans.shape[1]  ### redefine img_rows/ cols and add resize if needed
    #img_cols = scans.shape[2]
    
    
    # scans already are in the tensor mode  with 3 rgb elements ....
    
    #scans1 = scans   ## no change
    
    #scans1=np.zeros((scans.shape[0],3,img_rows,img_cols))
    #for i in range(scans.shape[0]):
    #    img=scans[i,:,:]
    #    ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
    #    scans1[i,0,:,:]=img
    
    if use_3d_mask:
        done = 1  # nothing to do
    else:      
        masks = np.copy(masks1)        
        masks1=np.zeros((masks.shape[0],1,img_rows,img_cols))
        for i in range(masks.shape[0]):
            img=masks[i,:,:]
            ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
            masks1[i,0,:,:]=img
        
    return scans1, masks1

#scans = [scans[j]]
#masks = [masks[j]]
def convert_scans_and_masks_xd3(scans, masks, only_with_nudels, dim=3, crop=16, blanks_per_axis = 4, add_blank_spacing_size=0, add_blank_layers = 0):
    
    
    # reuse scan to reduce memory footprint
    dim_orig = dim    
    #add_blank_spacing_size = 0 # dim *4  # dim  # was dim  ### set to 0 for version_16  #### initial trial (should perhaps be just dim ....), if 0 - do not add ...
    #add_blank_layers = 0  # was 4
    
    #skip = dim // 2    # old 
    skip_low = dim // 2  # dim shoudl be uneven -- it is recalculated anyway to this end
    skip_high = dim -skip_low - 1
    do_not_allow_even_dim = False ## now we allow odd numbers ...
    if do_not_allow_even_dim:
        dim = 2 * skip_low + 1
        skip_low = dim // 2
        skip_high = dim -skip_low - 1
        
        if dim != dim_orig:
            print ("convert_scans_and_masks_x: Dim must be uneven, corrected from .. to:", dim_orig, dim)
    work = []     # 3 layers
    #scan = scans[0]
    for scan in scans:  ##TEMP
        tmp = []
        #i = 1
        #for i in range(1, scan.shape[0]-1, 3):  # SKIP EVERY 3
        for i in range(skip_low, scan.shape[0]-skip_high):  
            #img1 = scan[i-1]
            #img2 = scan[i]
            #img3 = scan[i+1]
            #rgb = np.stack((img1, img2, img3))
            rgb = np.stack(scan[i-skip_low:i+skip_high+1])
            tmp.append(rgb)
        work.append(np.array(tmp))      

    #flattened1 = [val for sublist in work for val in sublist ] # NO skipping as we have already cut the first and the last layer    
    #scans1 = np.stack(flattened1)
    scans1 =  np.stack([val for sublist in work for val in sublist ]) # NO skipping as we have already cut the first and the last layer     
    work = []
    
    
    ##blanks_per_axis = 6 # cover all slice
    ##crop = 44
    dxrange = scans[0].shape[-1] - 2 * crop
    dyrange = scans[0].shape[-2] - 2 * crop
    #dx = (img_cols - 2 * crop) // (blanks_per_axis)
    #dy =  (img_rows - 2 * crop) // (blanks_per_axis)
    #dx = dxrange // (blanks_per_axis+1)
    #dy = dyrange // (blanks_per_axis+1)
       ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
    # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
    if add_blank_spacing_size > 0:
        for mask in masks:
            if (np.min(mask) < 0):
                ## we have a blank 
                ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
                # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
                for i in range(skip_low+(add_blank_spacing_size//2), mask.shape[0]-skip_high, add_blank_spacing_size):  
                    
                    mask[i, np.random.randint(0,dyrange), np.random.randint(0,dxrange)] = -1  # negative pixel to be picked up below and corrected back to none
    
    
    if add_blank_layers > 0:
        for mask in masks:
            if (np.min(mask) < 0):
                dzrange = mask.shape[0]-dim
                ## we have a blank 
                ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
                # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..      
                for k in range(add_blank_layers):
                    i = np.random.randint(0, dzrange) + skip_low    
                    #print ("dz position, random, mask.shape ", i, mask.shape)
                    mask[i, np.random.randint(0,dyrange), np.random.randint(0,dxrange)] = -1  # negative pixel to be picked up below and corrected back to none
    

  
    #mask = masks[0]
    add_random_blanks_in_blanks = False  ## NO need for the extra random blank pixels now, 20170327
    if add_random_blanks_in_blanks:
      for mask in masks:
        if (np.min(mask) < 0):
            ## we have a blank 
            ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
            # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
            #zlow = skip_low
            #zhigh = mask.shape[0]-skip_high
            pix_sum = np.sum(mask, axis=(1,2))
            idx_blanks =   np.min(mask, axis=(1,2)) < 0   ## don't use it - let's vary the position across the space
            for iz in range(mask.shape[0]):
                if (np.min(mask[iz])) < 0:  
                    for ix in range(blanks_per_axis):
                        #xpos = crop + (ix)*dx + dx //2  
                        for iy in range(blanks_per_axis):
                            #ypos = crop + (iy)*dy + dy //2
                            xpos = crop + np.random.randint(0,dxrange)
                            ypos = crop + np.random.randint(0,dyrange)
                            #print (iz, xpos, ypos)                    
                            #mask[idx_blanks, ypos, xpos] = -1  # negative pixel to be picked up below and corrected back to none
                            mask[iz, ypos, xpos] = -1  


    
    use_3d_mask = True  ## 
    if use_3d_mask:
        work = []     # 3 layers
        #mask = masks[0]
        for mask in masks:
            tmp = []
            #i = 0
            for i in range(skip_low, mask.shape[0]-skip_high):  
                #img1 = mask[i-1]
                #img2 = mask[i]
                #img3 = mask[i+1]
                #rgb = np.stack((img1, img2, img3))
                rgb = np.stack(mask[i-skip_low:i+skip_high+1])
                tmp.append(rgb)
            work.append(np.array(tmp))      
    
        masks1 = np.stack([val for sublist in work for val in sublist ] )# NO skipping as we have already cut the first and the last layer    
    else:
        masks1 = np.stack([val for sublist in masks for val in sublist[skip_low:-skip_high]] ) # skip one element at the beginning and at the end
        #masks1 = np.stack(flattened1)  # 10187
        
    
    #only_with_nudels = True
    if only_with_nudels:  
        if use_3d_mask:
            #nudels_pix_count = np.sum(np.abs(masks1[:,skip_low]), axis = (1,2))  ## CHANGE IT WED - use ANY i.e. remove skip_low abd added for the potential blanks; modified that the centre mask be mask!
            nudels_pix_count = np.sum(np.abs(masks1), axis = (1,2,3))  ## USE ANY March 1; CHANGE IT WED - use ANY i.e. remove skip_low abd added for the potential blanks; modified that the centre mask be mask!

        else:
            nudels_pix_count = np.sum(np.abs(masks1), axis = (1,2)) 
        scans1 = scans1[nudels_pix_count != 0]  
        masks1 = masks1[nudels_pix_count != 0]  
        
        #blank_mask_factor = np.sign(nudels_pix_count)[nudels_pix_count != 0]
        #sum(blank_mask_factor)
        #blank_mask_factor[blank_mask_factor <0] = 0
        #mask1_orig = masks1
        #np.sum(mask1_orig)
        #np.min(masks1)
        #masks1 = masks1[nudels_pix_count != 0] * blank_mask_factor # 493 -- circa 5 % with nudeles oters without; 232 if we skip over every 3 layers and use a 3d mask
        
        
        #masks1[masks1 < 0]  = 0 #  !!!!!!!!!!!!!!  in GRID version do NOT do that - do it in the key version 493 -- circa 5 % with nudeles oters without; 232 if we skip over every 3 layers and use a 3d mask

        
        #masks1[nudels_pix_count < 0] = 0  # making empty mask for balancing training set
    
    #nudels2 =  np.where(masks1 == 1, scans1, -4000*np.ones(( masks1.shape[1], masks1.shape[2)))  ### was -2000
    #nudels1 =  np.where(masks1 == 1, scans1, masks1 - 4000)  ### was -2000

    #nudles1_rf = nudels1.flatten()
    #nudles1_rf = nudles1_rf[nudles1_rf > -4000]
    
    scans1 = normalize(scans1)
    
    ### after this scans1 becomes float64 ....
    
    useTestPlot = False
    if useTestPlot:
        
        plt.hist(scans1.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        #for i in range(scans.shape[0]):
        for i in range(20):
            print ('scan '+str(i))
            f, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(scans1[i,:,:],cmap=plt.cm.gray)
            ax[1].imshow(scans[i,:,:],cmap=plt.cm.gray)
            ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
            plt.show()
        

    #np.mean(scans)  # 0.028367 / 0.0204    
    #np.min(scans)   # 0
    #np.max(scans)    # 

    scans1 = zero_center(scans1)
    #masks = np.copy(masks1)
     
    scans1 = scans1.astype(np.float32)  # make it float 32 (not point carring 64, also because kears operates on float32, and originals were in int
       
    ## if needed do the resize here .... (img_rows and img_cols are global values defined externally)
    #img_rows = scans.shape[1]  ### redefine img_rows/ cols and add resize if needed
    #img_cols = scans.shape[2]
    
    
    # scans already are in the tensor mode  with 3 rgb elements ....
    
    #scans1 = scans   ## no change
    
    #scans1=np.zeros((scans.shape[0],3,img_rows,img_cols))
    #for i in range(scans.shape[0]):
    #    img=scans[i,:,:]
    #    ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
    #    scans1[i,0,:,:]=img
    
    if use_3d_mask:
        done = 1  # nothing to do
    else:      
        masks = np.copy(masks1)        
        masks1=np.zeros((masks.shape[0],1,img_rows,img_cols))
        for i in range(masks.shape[0]):
            img=masks[i,:,:]
            ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
            masks1[i,0,:,:]=img
        
    return scans1, masks1


def convert_scans_and_masks_3d(scans, masks, only_with_nudels):
    
    
    # reuse scan to reduce memory footprint
    work = []     # 3 layers
    #scan = scans[0]
    for scan in scans:
        tmp = []
        #i = 0
        #for i in range(1, scan.shape[0]-1, 3):  # SKIP EVERY 3
        for i in range(1, scan.shape[0]-1):  
            img1 = scan[i-1]
            img2 = scan[i]
            img3 = scan[i+1]
            rgb = np.stack((img1, img2, img3))
            tmp.append(rgb)
        work.append(np.array(tmp))      

    #flattened1 = [val for sublist in work for val in sublist ] # NO skipping as we have already cut the first and the last layer    
    #scans1 = np.stack(flattened1)
    scans1 =  np.stack([val for sublist in work for val in sublist ]) # NO skipping as we have already cut the first and the last layer     
    work = []
    
    use_3d_mask = False
    if use_3d_mask:
        work = []     # 3 layers
        #mask = masks[0]
        for mask in masks:
            tmp = []
            #i = 0
            for i in range(1, mask.shape[0]-1, 3):  # SKIP EVERY 3
                img1 = mask[i-1]
                img2 = mask[i]
                img3 = mask[i+1]
                rgb = np.stack((img1, img2, img3))
                tmp.append(rgb)
            work.append(np.array(tmp))      
    
        masks1 = np.stack([val for sublist in work for val in sublist ] )# NO skipping as we have already cut the first and the last layer    
    else:
        masks1 = np.stack([val for sublist in masks for val in sublist[1:-1]] ) # skip one element at the beginning and at the end
        #masks1 = np.stack(flattened1)  # 10187
        
    
    #only_with_nudels = True
    if only_with_nudels:  
        if use_3d_mask:
            nudels_pix_count = np.sum(masks1, axis = (1,2,3))
        else:
            nudels_pix_count = np.sum(masks1, axis = (1,2))  
        scans1 = scans1[nudels_pix_count>0]  
        masks1 = masks1[nudels_pix_count>0] # 493 -- circa 5 % with nudeles oters without; 232 if we skip over every 3 layers and use a 3d mask
    
    
    #nudels2 =  np.where(masks1 == 1, scans1, -4000*np.ones(( masks1.shape[1], masks1.shape[2)))  ### was -2000
    #nudels1 =  np.where(masks1 == 1, scans1, masks1 - 4000)  ### was -2000

    #nudles1_rf = nudels1.flatten()
    #nudles1_rf = nudles1_rf[nudles1_rf > -4000]
    
    scans1 = normalize(scans1)
    
    useTestPlot = False
    if useTestPlot:
        
        plt.hist(scans1.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        #for i in range(scans.shape[0]):
        for i in range(20):
            print ('scan '+str(i))
            f, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(scans1[i,:,:],cmap=plt.cm.gray)
            ax[1].imshow(scans[i,:,:],cmap=plt.cm.gray)
            ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
            plt.show()
        

    #np.mean(scans)  # 0.028367 / 0.0204    
    #np.min(scans)   # 0
    #np.max(scans)    # 

    scans1 = zero_center(scans1)
    #masks = np.copy(masks1)
       
       
    ## if needed do the resize here .... (img_rows and img_cols are global values defined externally)
    #img_rows = scans.shape[1]  ### redefine img_rows/ cols and add resize if needed
    #img_cols = scans.shape[2]
    
    
    # scans already are in the tensor mode  with 3 rgb elements ....
    
    #scans1 = scans   ## no change
    
    #scans1=np.zeros((scans.shape[0],3,img_rows,img_cols))
    #for i in range(scans.shape[0]):
    #    img=scans[i,:,:]
    #    ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
    #    scans1[i,0,:,:]=img
    
    if use_3d_mask:
        done = 1  # nothing to do
    else:      
        masks = np.copy(masks1)        
        masks1=np.zeros((masks.shape[0],1,img_rows,img_cols))
        for i in range(masks.shape[0]):
            img=masks[i,:,:]
            ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
            masks1[i,0,:,:]=img
        
    return scans1, masks1



def view_scans(scans):
    #%matplotlib inline
    for i in range(scans.shape[0]):
        print ('scan '+str(i))
        plt.imshow(scans[i,0,:,:], cmap=plt.cm.gray)
        plt.show()

def view_scans_widget(scans):
    #%matplotlib tk
    for i in range(scans.shape[0]):
        plt.figure(figsize=(7,7))
        plt.imshow(scans[i,0,:,:], cmap=plt.cm.gray)
        plt.show()

def get_masks(scans,masks_list):
    #%matplotlib inline
    scans1=scans.copy()
    maxv=255
    masks=np.zeros(shape=(scans.shape[0],1,img_rows,img_cols))
    for i_m in range(len(masks_list)):
        for i in range(-masks_list[i_m][3],masks_list[i_m][3]+1):
            for j in range(-masks_list[i_m][3],masks_list[i_m][3]+1):
                masks[masks_list[i_m][0],0,masks_list[i_m][2]+i,masks_list[i_m][1]+j]=1
        for i1 in range(-masks_list[i_m][3],masks_list[i_m][3]+1):
            scans1[masks_list[i_m][0],0,masks_list[i_m][2]+i1,masks_list[i_m][1]+masks_list[i_m][3]]=maxv=255
            scans1[masks_list[i_m][0],0,masks_list[i_m][2]+i1,masks_list[i_m][1]-masks_list[i_m][3]]=maxv=255
            scans1[masks_list[i_m][0],0,masks_list[i_m][2]+masks_list[i_m][3],masks_list[i_m][1]+i1]=maxv=255
            scans1[masks_list[i_m][0],0,masks_list[i_m][2]-masks_list[i_m][3],masks_list[i_m][1]+i1]=maxv=255
    for i in range(scans.shape[0]):
        print ('scan '+str(i))
        f, ax = plt.subplots(1, 2,figsize=(10,5))
        ax[0].imshow(scans1[i,0,:,:],cmap=plt.cm.gray)
        ax[1].imshow(masks[i,0,:,:],cmap=plt.cm.gray)
        plt.show()
    return(masks)

def augmentation(scans,masks,n):
    datagen = ImageDataGenerator(
        featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=25,   # was 25
        width_shift_range=0.3,  # ws 0.3; was 0.1# tried 0.01
        height_shift_range=0.3,   # was 0.3; was 0.1 # tried 0.01
        horizontal_flip=True,   
        vertical_flip=True,  
        zoom_range=False)
    i=0
    scans_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000): 
        scans_g=np.vstack([scans_g,batch])
        i += 1
        if i > n:
            break
    i=0
    masks_g=masks.copy()
    for batch in datagen.flow(masks, batch_size=1, seed=1000): 
        masks_g=np.vstack([masks_g,batch])
        i += 1
        if i > n:
            break
    return((scans_g,masks_g))



def hu_to_pix (hu):
    return (hu - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN

def pix_to_hu (pix):
    return (pix + PIXEL_MEAN)  * (MAX_BOUND - MIN_BOUND) + MIN_BOUND

from scipy import stats
def eliminate_incorrectly_segmented(scans, masks):
       

        skip = dim // 2  # To Change see below ...
        sxm =   scans *   masks
    
        near_air_thresh = (-900 - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN  # version 3  # -750 gives one more (for 0_3, d4, -600 give 15 more than -900
        near_air_thresh  #0.08628  for -840 # 0.067     # for -867; 0.1148 for -800
        cnt = 0        
        for i in range(sxm.shape[0]):
             #sx = sxm[i,skip]
             sx = sxm[i]
             mx = masks[i]
             if np.sum(mx) > 0:     # only check non-blanks ...(keep blanks)
                 sx_max = np.max(sx)
                 if (sx_max) <= near_air_thresh:
                     cnt += 1
                     print ("Entry, count # and max: ", i, cnt, sx_max)
                     print (stats.describe(sx, axis=None))
                     #plt.imshow(sx, cmap='gray')
                     plt.imshow(sx[0,skip], cmap='gray')    # selecting the mid entry
                     plt.show()
            
        s_eliminate = np.max(sxm, axis=(1,2,3,4)) <= near_air_thresh  # 3d
        s_preserve = np.max(sxm, axis=(1,2,3,4)) > near_air_thresh    #3d

        s_eliminate_sum = sum(s_eliminate) 
        s_preserve_sum = sum(s_preserve)
        print ("Eliminate, preserve =", s_eliminate_sum, s_preserve_sum)
        
        masks = masks[s_preserve]
        scans = scans[s_preserve]
        del(sxm)
        
        return scans, masks



# the following 3 functions to read LUNA files are from: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def seq(start, stop, step=1):
	n = int(round((stop - start)/float(step)))
	if n > 1:
		return([start + step*i for i in range(n+1)])
	else:
		return([])

'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''

def draw_circles(image,cands,origin,spacing):
	#make empty matrix, which will be filled with the mask
	image_mask = np.zeros(image.shape, dtype=np.int16)

	#run over all the nodules in the lungs
	for ca in cands.values:
		#get middel x-,y-, and z-worldcoordinate of the nodule
		#radius = np.ceil(ca[4])/2     ## original:  replaced the ceil with a very minor increase of 1% ....
		radius = (ca[4])/2 + 0.51 * spacing[0]  # increasing by circa half of distance in z direction .... (trying to capture wider region/border for learning ... and adress the rough net .
    
		coord_x = ca[1]
		coord_y = ca[2]
		coord_z = ca[3]
		image_coord = np.array((coord_z,coord_y,coord_x))

		#determine voxel coordinate given the worldcoordinate
		image_coord = world_2_voxel(image_coord,origin,spacing)

		#determine the range of the nodule
		#noduleRange = seq(-radius, radius, RESIZE_SPACING[0])  # original, uniform spacing 
		noduleRange_z = seq(-radius, radius, spacing[0])
		noduleRange_y = seq(-radius, radius, spacing[1])
		noduleRange_x = seq(-radius, radius, spacing[2])

          #x = y = z = -2
		#create the mask
		for x in noduleRange_x:
			for y in noduleRange_y:
				for z in noduleRange_z:
					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
					#if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:  ### original (contrained to a uniofrm RESIZE)
					if (np.linalg.norm((image_coord-coords) * spacing)) < radius:
						image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
	

	return image_mask

'''
This function takes the path to a '.mhd' file as input and 
is used to create the nodule masks and segmented lungs after 
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as 
input.
'''

def load_scans_masks_or_blanks(luna_subset, useAll, use_unsegmented=True):
    
    #luna_subset = "[0-6]"
    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))

    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()

    candidates =    pd.read_csv(LUNA_CANDIDATES)
    candidates_false = candidates[candidates["class"] == 0]  # only select the false candidates
    candidates_true = candidates[candidates["class"] == 1]  # only select the false candidates

    sids = []
    scans = []
    masks = []
    blankids = []  # class/id whether scan is with nodule or without, 0 - with, 1 - without 
    cnt = 0
    skipped = 0
    #file=files[7]
    for file in files:    
        imagePath = file
        seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
        seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid
        
        path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
        if use_unsegmented:
            path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)
        else:
            path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)

        cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
        ctrue = candidates_true[seriesuid == candidates_true.seriesuid] 
        cfalse = candidates_false[seriesuid == candidates_false.seriesuid] 
        
        blankid = 1 if (len(cands) == 0 and len(ctrue) == 0 and len(cfalse) > 0) else 0
        
        skip_nodules_entirely = False  # was False
        use_only_nodules = False
        if skip_nodules_entirely and blankid ==0:
            ## manual switch to generate extra data for the corrupted set
            print("Skipping nodules  (skip_nodules_entirely) ", seriesuid)
            skipped += 1
            
        elif use_only_nodules and (len(cands) == 0):
            ## manual switch to generate only nodules data due lack of time and repeat etc time pressures
            print("Skipping blanks  (use_only_nodules) ", seriesuid)
            skipped += 1
        else:  # NORMAL operations
             if (len(cands) > 0 or 
                    (blankid >0) or
                    useAll):
                sids.append(seriesuid)  
                blankids.append(blankid)  
                
                
                if use_unsegmented:
                    scan_z = np.load(''.join((path_segmented  + '_lung' + '.npz'))) 
                else:
                    scan_z = np.load(''.join((path_segmented  + '_lung_seg' + '.npz'))) 
                scan = scan_z['arr_0']
                mask_z = np.load(''.join((path_segmented  + '_nodule_mask_wblanks' + '.npz'))) 
                mask = mask_z['arr_0']   
                
                testPlot = False
                if testPlot:
                    maskcheck_z = np.load(''.join((path_segmented  + '_nodule_mask' + '.npz'))) 
                    maskcheck = maskcheck_z['arr_0']
                    
                    f, ax = plt.subplots(1, 2, figsize=(10,5))
                    ax[0].imshow(np.sum(np.abs(maskcheck), axis=0),cmap=plt.cm.gray)
                    ax[1].imshow(np.sum(np.abs(mask), axis=0),cmap=plt.cm.gray)
                    #ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                    plt.show()
                    
                scans.append(scan)
                masks.append(mask)       
                cnt += 1
             else:
                print("Skipping non-nodules and non-blank entry ", seriesuid)
                skipped += 1
            
            
    print ("Summary: cnt & skipped: ", cnt, skipped)
    
    return scans, masks, sids, blankids

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.028  ## for LUNA subset 0 and our preprocessing, only with nudels was 0.028, all was 0.020421744071562546 (in the tutorial they used 0.25)

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def convert_scans_and_masks_xd3(scans, masks, only_with_nudels, dim=3, crop=16, blanks_per_axis = 4, add_blank_spacing_size=0, add_blank_layers = 0):
    
    # reuse scan to reduce memory footprint
    dim_orig = dim    
    
    skip_low = dim // 2  # dim shoudl be uneven -- it is recalculated anyway to this end
    skip_high = dim -skip_low - 1
    do_not_allow_even_dim = False ## now we allow odd numbers ...
    if do_not_allow_even_dim:
        dim = 2 * skip_low + 1
        skip_low = dim // 2
        skip_high = dim -skip_low - 1
        
        if dim != dim_orig:
            print ("convert_scans_and_masks_x: Dim must be uneven, corrected from .. to:", dim_orig, dim)
    work = []     
    for scan in scans:
        tmp = []
        for i in range(skip_low, scan.shape[0]-skip_high):  
            #img1 = scan[i-1]
            #img2 = scan[i]
            #img3 = scan[i+1]
            #rgb = np.stack((img1, img2, img3))
            rgb = np.stack(scan[i-skip_low:i+skip_high+1])
            tmp.append(rgb)
        work.append(np.array(tmp))      

    scans1 =  np.stack([val for sublist in work for val in sublist ]) # NO skipping as we have already cut the first and the last layer     
    work = []
    
    dxrange = scans[0].shape[-1] - 2 * crop
    dyrange = scans[0].shape[-2] - 2 * crop
    if add_blank_spacing_size > 0:
        for mask in masks:
            if (np.min(mask) < 0):
                ## we have a blank 
                ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
                # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
                for i in range(skip_low+(add_blank_spacing_size//2), mask.shape[0]-skip_high, add_blank_spacing_size):                  
                    mask[i, np.random.randint(0,dyrange), np.random.randint(0,dxrange)] = -1  # negative pixel to be picked up below and corrected back to none
    
    if add_blank_layers > 0:
        for mask in masks:
            if (np.min(mask) < 0):
                dzrange = mask.shape[0]-dim
                ## we have a blank 
                ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
                # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..      
                for k in range(add_blank_layers):
                    i = np.random.randint(0, dzrange) + skip_low    
                    #print ("dz position, random, mask.shape ", i, mask.shape)
                    mask[i, np.random.randint(0,dyrange), np.random.randint(0,dxrange)] = -1  # negative pixel to be picked up below and corrected back to none
    
    add_random_blanks_in_blanks = False  ## NO need for the extra random blank pixels now, 20170327
    if add_random_blanks_in_blanks:
      for mask in masks:
        if (np.min(mask) < 0):
            ## we have a blank 
            ### ADD ariticial mask pixel every  add_blank_spacing layers for each blankids ...   
            # set the (0,0) pixel to -1   every add_blank_spacing_size for blanks ..
            #zlow = skip_low
            #zhigh = mask.shape[0]-skip_high
            pix_sum = np.sum(mask, axis=(1,2))
            idx_blanks =   np.min(mask, axis=(1,2)) < 0   ## don't use it - let's vary the position across the space
            for iz in range(mask.shape[0]):
                if (np.min(mask[iz])) < 0:  
                    for ix in range(blanks_per_axis):
                        #xpos = crop + (ix)*dx + dx //2  
                        for iy in range(blanks_per_axis):
                            #ypos = crop + (iy)*dy + dy //2
                            xpos = crop + np.random.randint(0,dxrange)
                            ypos = crop + np.random.randint(0,dyrange)
                            #print (iz, xpos, ypos)                    
                            #mask[idx_blanks, ypos, xpos] = -1  # negative pixel to be picked up below and corrected back to none
                            mask[iz, ypos, xpos] = -1  

    use_3d_mask = True  ## 
    if use_3d_mask:
        work = []     # 3 layers
        for mask in masks:
            tmp = []
            #i = 0
            for i in range(skip_low, mask.shape[0]-skip_high):  
                rgb = np.stack(mask[i-skip_low:i+skip_high+1])
                tmp.append(rgb)
            work.append(np.array(tmp))      
    
        masks1 = np.stack([val for sublist in work for val in sublist ] )# NO skipping as we have already cut the first and the last layer    
    else:
        masks1 = np.stack([val for sublist in masks for val in sublist[skip_low:-skip_high]] ) # skip one element at the beginning and at the end
                
    if only_with_nudels:  
        if use_3d_mask:
            nudels_pix_count = np.sum(np.abs(masks1), axis = (1,2,3))  ## USE ANY March 1; CHANGE IT WED - use ANY i.e. remove skip_low abd added for the potential blanks; modified that the centre mask be mask!
        else:
            nudels_pix_count = np.sum(np.abs(masks1), axis = (1,2)) 
        scans1 = scans1[nudels_pix_count != 0]  
        masks1 = masks1[nudels_pix_count != 0]  
        
    scans1 = normalize(scans1)
    
    useTestPlot = False
    if useTestPlot:
        
        plt.hist(scans1.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        for i in range(20):
            print ('scan '+str(i))
            f, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(scans1[i,:,:],cmap=plt.cm.gray)
            ax[1].imshow(scans[i,:,:],cmap=plt.cm.gray)
            ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
            plt.show()
        

    scans1 = zero_center(scans1)
     
    scans1 = scans1.astype(np.float32)  # make it float 32 (not point carring 64, also because kears operates on float32, and originals were in int
    
    if use_3d_mask:
        done = 1  # nothing to do
    else:      
        masks = np.copy(masks1)        
        masks1=np.zeros((masks.shape[0],1,img_rows,img_cols))
        for i in range(masks.shape[0]):
            img=masks[i,:,:]
            ###img =cv2.resize(img, (img_rows, img_cols))  ## add/test resizing if needed
            masks1[i,0,:,:]=img
        
    return scans1, masks1

def eliminate_incorrectly_segmented(scans, masks):
       
        skip = dim // 2  # To Change see below ...
        sxm =   scans *   masks
    
        near_air_thresh = (-900 - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN  # version 3  # -750 gives one more (for 0_3, d4, -600 give 15 more than -900
        #near_air_thresh  #0.08628  for -840 # 0.067     # for -867; 0.1148 for -800
        cnt = 0        
        for i in range(sxm.shape[0]):
             #sx = sxm[i,skip]
             sx = sxm[i]
             mx = masks[i]
             if np.sum(mx) > 0:     # only check non-blanks ...(keep blanks)
                 sx_max = np.max(sx)
                 if (sx_max) <= near_air_thresh:
                     cnt += 1
                     print ("Entry, count # and max: ", i, cnt, sx_max)
                     print (stats.describe(sx, axis=None))
                     #plt.imshow(sx, cmap='gray')
                     plt.imshow(sx[0,skip], cmap='gray')    # selecting the mid entry
                     plt.show()
                    
        s_eliminate = np.max(sxm, axis=(1,2,3,4)) <= near_air_thresh  # 3d
        s_preserve = np.max(sxm, axis=(1,2,3,4)) > near_air_thresh    #3d

        s_eliminate_sum = sum(s_eliminate)
        s_preserve_sum = sum(s_preserve) 
        print ("Eliminate, preserve =", s_eliminate_sum, s_preserve_sum)

        
        masks = masks[s_preserve]
        scans = scans[s_preserve]
        del(sxm)
        
        return scans, masks


def grid_data(source, grid=32, crop=16, expand=12):
    gridsize = grid + 2 * expand
    stacksize = source.shape[0]
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
    
    height = cells.shape[3]  # should be 224 for our data
    width = cells.shape[4]
    crop = (width - grid ) // 2 ## for simplicity we are assuming the same crop (and grid) vertically and horizontally
    
    dspacing = gridwidth * gridheight
    layers = cells.shape[0] // dspacing
    
    if crop > 0:  # do NOT crop with 0 as we get empty cells ...
        cells = cells[:,:,:,crop:-crop,crop:-crop]     
    
    if crop > 2*grid:
        print ("data_from_grid Warning, unusually large crop (> 2*grid); crop, & grid, gridwith, gridheight: ", (crop, grid, gridwidth, gridheight))
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
    dspacing = gridwidth * gridheight
    layers = cells.shape[0] // dspacing
        
    shape = cells.shape
    new_shape_1_dim = shape[0]// (gridwidth * gridheight)  # ws // 36 -- Improved on 20170306
    
    ### NOTE tha we invert the order of shapes below to get the required proximity type ordering
    new_shape = (new_shape_1_dim, gridwidth * gridheight,  ) +  tuple([x for x in shape][1:])   # was 36,  Improved on 20170306
    
    # swap ordering of axes 
    cells = np.reshape(cells, new_shape) 
    cells = cells.swapaxes(0, 1)
    cells = np.reshape(cells, shape) 
    
    cells = data_from_grid (cells, gridwidth, gridheight, grid)
    
    return cells


def find_voxels(dim, grid, images3, images3_seg, pmasks3, nodules_threshold=0.999, voxelscountmax = 1000, mid_mask_only = True, find_blanks_also = True, centralcutonly=True):

    zsel = dim // 2
    sstart = 0
    send = images3.shape[0]
    if mid_mask_only:
        pmav = pmasks3[:,0,dim // 2]  # using the mid mask
        
        pmav.shape
    else:
        pmav =  pmasks3[:,0]   ### NOTE this variant has NOT been tested fully YET

    run_UNNEEDED_code = False
    
    ims = images3[sstart:send,0,zsel]      # selecting the zsel cut for nodules calc ...
    ims_seg = images3_seg[sstart:send,0,zsel] 
    ims.shape
    #pms = pmasks3[sstart:send,0,0]
    pms = pmav[sstart:send]
    images3.shape
    
    thresh = nodules_threshold  # for testing , set it here and skip the loop
    segment = 2     # for compatibility of the naming convention
    # threshold the precited nasks ...
    #for thresh in [0.5, 0.9, 0.9999]:
    #for thresh in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
    for thresh in [nodules_threshold]:  # jusst this one - keeping loop for a while

        if find_blanks_also:
            idx = np.abs(pms) > thresh
        else:
            idx = pms > thresh
        idx.shape
        nodls = np.zeros(pms.shape).astype(np.int16)
        nodls[idx] = 1
        
        nx =  nodls[idx]

        nodules_pixels = ims[idx]   # flat
        nodules_hu = pix_to_hu(nodules_pixels)
        part_name = ''.join([str(segment), '_', str(thresh)])
        ### DO NOT do them here      
        use_corrected_nodules = True  # do it below from 20170311
        if not use_corrected_nodules:
            df = hu_describe(nodules_hu, uid=uid, part=part_name)
        
        add_projections = False
        axis = 1
        nodules_projections = []
        for axis in range(3):
             nodls_projection = np.max(nodls, axis=axis)
             naxis_name = ''.join(["naxis_", str(axis),"_", part_name])
             if add_projections:   
                 df[naxis_name] = np.sum(nodls_projection)
             nodules_projections.append(nodls_projection)
        
        idx.shape
        ## find the individual nodules ... as per the specified probabilities 
        labs, labs_num = measure.label(idx, return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
        voxels = []  
        vmasks = []
        if labs_num > 0 and labs.shape[0] >1:   # checking for height > 1 is needed as  measure.regionprops fails when it is not, for instance for shape (1, 20, 20) we get ValueError: Label and intensity image must have the same shape.

            print("Befpre measure.regionprops, labs & intensity shapes: ", labs.shape, ims.shape)
            regprop = measure.regionprops(labs, intensity_image=ims)  # probkem here on 20170327
            voxel_volume = np.product(RESIZE_SPACING) 
            areas = [rp.area for rp in regprop] # this is in cubic mm now (i.e. should really be called volume)                       
            volumes = [rp.area * voxel_volume for rp in regprop] 
            diameters = [2 * (3* volume / (4 * np.pi ))**0.3333 for volume in volumes]
 
 
            labs_ids =  [rp.label for rp in regprop]
            #ls = [rp.label for rp in regprop]
            max_val = np.max(areas)
            max_index = areas.index(max_val)
            max_label = regprop[max_index].label
            bboxes = [r.bbox for r in regprop]
        
            idl = labs ==  regprop[max_index].label   #  400
            nodules_pixels = ims[idl]
            nodules_hu = pix_to_hu(nodules_pixels)
            
            if run_UNNEEDED_code:
                nodules_hu_reg = []
                for rp in regprop:
                    idl = labs == rp.label
                    nodules_pixels = ims[idl]
                    nodules_hu = pix_to_hu(nodules_pixels)
                    nodules_hu_reg.append(nodules_hu)           # NOTE some are out of interest, i.e. are equal all (or near all) to  MAX_BOUND (400)
                
            dfn = pd.DataFrame(
                {
                 "area":    areas,
                 "diameter":   diameters,
                 "bbox":       bboxes
                 },
                 index=labs_ids)
            
            
            nodules_count = len(dfn)  # 524 for file 1 of part 8 ..
                     
            max_nodules_count = voxelscountmax                         
            n=0
            for n in range(max_nodules_count):
                 if n < len(dfn): # use the nodule data, otheriwse empty
                        
                    bb = dfn.iloc[n]["bbox"]
                    zmin = bb[0]
                    zmax = bb[3]    
                    zlen = bb[3] - bb[0]
                    ylen = bb[4] - bb[1]
                    xlen = bb[5] - bb[2]
                    
                    xmin = np.max([bb[2] - np.max([(grid - xlen ) //2, 0]), 0])      ## do not go beyond 0/left side of the image
                    xmax = np.min([xmin + grid, ims.shape[2]])   ## do not beyond the right side 
                    xmin = xmax - grid
                    if (xmax - xmin) != grid:
                        print ("ERROR in calculating the cut-offs ..., xmin, xmax =", xmin, xmax)
                    
                    ymin = np.max([bb[1] - np.max([(grid - ylen ) //2, 0]), 0])    ## do not go beyond 0/left side of the image
                    ymax = np.min([ymin + grid, ims.shape[1]])   ## do not beyond the right side 
                    ymin = ymax - grid
                    if (ymax - ymin) != grid:
                        print ("ERROR in calculating the cut-offs ..., ymin, ymax =", ymin, ymax)
                    
                    zmin_sel = zmin
                    zmax_sel = zmax
                    if centralcutonly:  #include only one voxel representation
                        zmin_sel = zmin + zlen // 2
                        zmax_sel = zmin_sel + 1
                    iz=zmin_sel # for testing                    
                
                    for iz in range(zmin_sel,zmax_sel):    
                        voxel = images3[iz,:,:, ymin:ymax, xmin:xmax]
                        vmask = pmasks3[iz,:,:, ymin:ymax, xmin:xmax]
                                                             
                        voxels.append(voxel)
                        vmasks.append(vmask)
                        
                        testPlot = False
                        if testPlot:    
                            print ('scan '+str(iz))
                            f, ax = plt.subplots(1, 8, figsize=(24,3))
                            ax[0].imshow(nodls[iz,ymin:ymax, xmin:xmax],cmap=plt.cm.gray)
                            ax[1].imshow(ims[iz,ymin:ymax, xmin:xmax],cmap=plt.cm.gray)
                            ax[2].imshow(images3_amp[iz,0, dim//2, ymin:ymax, xmin:xmax],cmap=plt.cm.gray)
                            ax[3].imshow(voxel[0,dim//2],cmap=plt.cm.gray)
                            ax[4].imshow(voxel[0,dim],cmap=plt.cm.gray)
                            ax[5].imshow(voxel[0,dim+1],cmap=plt.cm.gray)
                            ax[6].imshow(voxel[0,dim+2],cmap=plt.cm.gray)
                            ax[7].imshow(voxel[0,dim+3],cmap=plt.cm.gray)
    
    if len(voxels) > 0:
        voxel_stack = np.stack(voxels)   
        vmask_stack = np.stack(vmasks)
    else:
        print_warning = False
        if print_warning:
            print("WARNING, find_voxels, not single voxel found even though expected")        
        voxel_stack = []
        vmask_stack = []
   
    if testPlot:    
        print ('voxels count ', len(voxel_stack))
        for ii in range(0,len(voxel_stack),len(voxel_stack)//10):
            f, ax = plt.subplots(1, 2, figsize=(6,3))
            ax[0].imshow(voxel_stack[ii, 0, dim // 2],cmap=plt.cm.gray)
            ax[1].imshow(vmask_stack[ii, 0, dim // 2],cmap=plt.cm.gray)
                           
    return voxel_stack, vmask_stack



def measure_voxels(labs, ims):

    #print("Befpre measure.regionprops, labs & intensity shapes: ", labs.shape, ims.shape)
    regprop = measure.regionprops(labs, intensity_image=ims)  # probkem here on 20170327
    voxel_volume = np.product(RESIZE_SPACING) 
    areas = [rp.area for rp in regprop] # this is in cubic mm now (i.e. should really be called volume)                       
    volumes = [rp.area * voxel_volume for rp in regprop] 
    diameters = [2 * (3* volume / (4 * np.pi ))**0.3333 for volume in volumes]
 
 
    labs_ids =  [rp.label for rp in regprop]
    #ls = [rp.label for rp in regprop]
    max_val = np.max(areas)
    max_index = areas.index(max_val)
    max_label = regprop[max_index].label
    bboxes = [r.bbox for r in regprop]
    #max_ls = ls[max_index]

    idl = labs ==  regprop[max_index].label   #  400
    nodules_pixels = ims[idl]
    nodules_hu = pix_to_hu(nodules_pixels)
    
    run_UNNEEDED_code = False
    if run_UNNEEDED_code:
        nodules_hu_reg = []
        for rp in regprop:
            idl = labs == rp.label
            nodules_pixels = ims[idl]
            nodules_hu = pix_to_hu(nodules_pixels)
            nodules_hu_reg.append(nodules_hu)           # NOTE some are out of interest, i.e. are equal all (or near all) to  MAX_BOUND (400)
        
    dfn = pd.DataFrame(
        {
         #"zcenter": zcenters,
         #"ycenter": ycenters, 
         #"xcenter": xcenters, 
         "area":    areas,
         "diameter":   diameters,
         #"irreg_vol":  irreg_vol,
         #"irreg_shape": irreg_shape,
         #"nodules_hu": nodules_hu_reg,
         "bbox":       bboxes
         },
         index=labs_ids)
         
    return dfn



def find_voxels_and_blanks(dim, grid, images3, images3_seg, pmasks3, nodules_threshold=0.999, voxelscountmax = 1000, find_blanks_also = True, centralcutonly=True, diamin=2, diamax=10):

    
    if np.sum(pmasks3) > 0:
        centralcutonly = False      # override centralcut for True nodule masks
    
    zsel = dim // 2 if centralcutonly else range(0,dim)
        
    pmav = pmasks3[:,0,zsel]
    ims = images3[:,0,zsel]      # selecting the zsel cut for nodules calc ...
    ims_seg = images3_seg[:,0,zsel] 
    
    
    sstart = 0
    send = images3.shape[0]
    pms = pmav[sstart:send]
        
    run_UNNEEDED_code = False
    
    thresh = nodules_threshold  # for testing , set it here and skip the loop
    segment = 2     # for compatibility of the naming convention
    for thresh in [nodules_threshold]:  # jusst this one - keeping loop for a while
        if find_blanks_also:
            idx = np.abs(pms) > thresh
        else:
            idx = pms > thresh
        idx.shape
        nodls = np.zeros(pms.shape).astype(np.int16)
        nodls[idx] = 1
        
        nx =  nodls[idx]
        volume = np.sum(nodls)  # A check calculation ... :wcounted as a count within hu_describe 

        nodules_pixels = ims[idx]   # flat
        nodules_hu = pix_to_hu(nodules_pixels)
        part_name = ''.join([str(segment), '_', str(thresh)])
        ### DO NOT do them here      
        use_corrected_nodules = True  # do it below from 20170311
        if not use_corrected_nodules:
            df = hu_describe(nodules_hu, uid=uid, part=part_name)
        
        add_projections = False
        if add_projections:
            nodules_projections = []
            for axis in range(3):
                 #sxm_projection = np.max(sxm, axis = axis)
                 nodls_projection = np.max(nodls, axis=axis)
                 naxis_name = ''.join(["naxis_", str(axis),"_", part_name])
                 if add_projections:   
                     df[naxis_name] = np.sum(nodls_projection)
                 nodules_projections.append(nodls_projection)
        
        voxels = []  
        vmasks = []


        if not centralcutonly:  
            for k in range(idx.shape[0]):
                if np.sum(idx[k]) > 0:                 
                    ## find the nodules and take a cut             
                    labs, labs_num = measure.label(idx[k], return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
                    dfn = measure_voxels(labs, ims[k])
                 
                 
                    nodules_count_0 = len(dfn)
                    ## CUT out anything that is outside of the specified diam range
                    dfn = dfn[(dfn["diameter"] >= diamin) & ((dfn["diameter"] < diamax))]  # CUT OUT anything that is less than 3 mm (essentially less than 7 voxels for 2x2x2
                    nodules_count = len(dfn)  # 524 for file 1 of part 8 ..
          
                    max_nodules_count = voxelscountmax                         
                    n=0
                    for n in range(max_nodules_count):
                         if n < len(dfn): # use the nodule data, otheriwse empty
                                
                            bb = dfn.iloc[n]["bbox"]
                            zmin = bb[0]
                            zmax = bb[3]    
                            zlen = bb[3] - bb[0]
                            ylen = bb[4] - bb[1]
                            xlen = bb[5] - bb[2]
                            
                            xmin = np.max([bb[2] - np.max([(grid - xlen ) //2, 0]), 0])      ## do not go beyond 0/left side of the image
                            xmax = np.min([xmin + grid, ims.shape[-1]])   ## do not beyond the right side 
                            xmin = xmax - grid
                            if (xmax - xmin) != grid:
                                print ("ERROR in calculating the cut-offs ..., xmin, xmax =", xmin, xmax)
                            
                            ymin = np.max([bb[1] - np.max([(grid - ylen ) //2, 0]), 0])    ## do not go beyond 0/left side of the image
                            ymax = np.min([ymin + grid, ims.shape[-2]])   ## do not beyond the right side 
                            ymin = ymax - grid
                            if (ymax - ymin) != grid:
                                print ("ERROR in calculating the cut-offs ..., ymin, ymax =", ymin, ymax)
                                                        
                            # here simply takje the entire voxel we have 
                            
                            #images3.shape
                            voxel = images3[k,:,:, ymin:ymax, xmin:xmax]
                            vmask = pmasks3[k,:,:, ymin:ymax, xmin:xmax]
                                                                 
                            voxels.append(voxel)
                            vmasks.append(vmask)
 
                            #voxel.shape     
        
        else:# essentially taking the central cuts of the blanks 
            ## find the individual nodules ... as per the specified probabilities 
            labs, labs_num = measure.label(idx, return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
    
            if labs_num > 0 and labs.shape[0] >1:   # checking for height > 1 is needed as  measure.regionprops fails when it is not, for instance for shape (1, 20, 20) we get ValueError: Label and intensity image must have the same shape.
                #labs_num_to_store = 5        
    
                dfn = measure_voxels(labs, ims)
                 
                nodules_count = len(dfn)  # 524 for file 1 of part 8 ..
      
                max_nodules_count = voxelscountmax                         
                n=0
                for n in range(max_nodules_count):
                     if n < len(dfn): # use the nodule data, otheriwse empty
                            
                        bb = dfn.iloc[n]["bbox"]
                        zmin = bb[0]
                        zmax = bb[3]    
                        zlen = bb[3] - bb[0]
                        ylen = bb[4] - bb[1]
                        xlen = bb[5] - bb[2]
                        
                        xmin = np.max([bb[2] - np.max([(grid - xlen ) //2, 0]), 0])      ## do not go beyond 0/left side of the image
                        xmax = np.min([xmin + grid, ims.shape[-1]])   ## do not beyond the right side 
                        xmin = xmax - grid
                        if (xmax - xmin) != grid:
                            print ("ERROR in calculating the cut-offs ..., xmin, xmax =", xmin, xmax)
                        
                        ymin = np.max([bb[1] - np.max([(grid - ylen ) //2, 0]), 0])    ## do not go beyond 0/left side of the image
                        ymax = np.min([ymin + grid, ims.shape[-2]])   ## do not beyond the right side 
                        ymin = ymax - grid
                        if (ymax - ymin) != grid:
                            print ("ERROR in calculating the cut-offs ..., ymin, ymax =", ymin, ymax)
                        
                        zmin_sel = zmin
                        zmax_sel = zmax
                        if centralcutonly:  #include only one voxel representation
                            zmin_sel = zmin + zlen // 2
                            zmax_sel = zmin_sel + 1
                        iz=zmin_sel # for testing
                        
                    
                        for iz in range(zmin_sel,zmax_sel):    
                            voxel = images3[iz,:,:, ymin:ymax, xmin:xmax]
                            vmask = pmasks3[iz,:,:, ymin:ymax, xmin:xmax]
                                                                 
                            voxels.append(voxel)
                            vmasks.append(vmask)
                            
                            testPlot = False
                            if testPlot:    
                                print ('scan '+str(iz))
                                f, ax = plt.subplots(1, 8, figsize=(24,3))
                                ax[0].imshow(nodls[iz,ymin:ymax, xmin:xmax],cmap=plt.cm.gray)
                                ax[1].imshow(ims[iz,ymin:ymax, xmin:xmax],cmap=plt.cm.gray)
                                ax[2].imshow(images3_amp[iz,0, dim//2, ymin:ymax, xmin:xmax],cmap=plt.cm.gray)
                                ax[3].imshow(voxel[0,dim//2],cmap=plt.cm.gray)
                                ax[4].imshow(voxel[0,dim],cmap=plt.cm.gray)
                                ax[5].imshow(voxel[0,dim+1],cmap=plt.cm.gray)
                                ax[6].imshow(voxel[0,dim+2],cmap=plt.cm.gray)
                                ax[7].imshow(voxel[0,dim+3],cmap=plt.cm.gray)
        
    if len(voxels) > 0:
        voxel_stack = np.stack(voxels)   
        vmask_stack = np.stack(vmasks)
    else:
        print_warning = False
        if print_warning:        
            print("WARNING, find_voxels, not single voxel found even though expected")        
        voxel_stack = []
        vmask_stack = []
   
    #print("Nodules, voxels_aggregated: ", len(dfn), len(voxel_stack))
    #np.savez_compressed(path_voxels_variant, voxel_stack)
    testPlot = False
    if testPlot:    
        print ('voxels count ', len(voxel_stack))
        for ii in range(0,len(voxel_stack),len(voxel_stack)//10):
            #plt.imshow(voxel_stack[ii,0,dim // 2], cmap=plt.cm.gray)
            #plt.show()
            f, ax = plt.subplots(1, 2, figsize=(6,3))
            ax[0].imshow(voxel_stack[ii, 0, dim // 2],cmap=plt.cm.gray)
            ax[1].imshow(vmask_stack[ii, 0, dim // 2],cmap=plt.cm.gray)
       
                           
    return voxel_stack, vmask_stack

def shuffle_scans_masks(scans, masks, seed):
    np.random.seed(seed)
    index_shuf = np.arange(len(scans))
    np.random.shuffle(index_shuf)
    scans = scans[index_shuf]
    masks = masks[index_shuf]
    return scans, masks    

def create_unet_training_files (dim, recreate_grid8_March_data=True):  # version with backward compatibility
    
    grid8_March_data_str = "a" if recreate_grid8_March_data else ""  # used for the the original data/approach
    # the main procedure to create training files for the nodule identifier (consolidated version, with backward compatibility for grid 8)
    create_main_grid = True
    if create_main_grid:
        diamins_2_10 = not recreate_grid8_March_data  # backward compatible option
        if diamins_2_10:         
            grids = [10, 20]
            diamins = [2, 10]
            diamaxs = [10, 100]
            crops2 = [7, 2]   # not used in this option, added for flow 
        else:
            grids = [20, 40]
            diamins = [2, 2]
            diamaxs = [100, 100]  
            crops2 = [2, 12]  # added to recreate_grid8_March_data
            
    else:
        ## created separately -- as an addition - for extra augmentation
        grids = [10]
        diamins = [2]
        diamaxs = [5]
        crops2 = [7]
        
    create_missing_grid_file = False
    if create_missing_grid_file:
        grids = [20]
        diamins = [19]
        diamaxs = [99] 
        crops2 = [2]
    resolution_str = RESOLUTION_STR
    
    grid=20
    
    centralcutonly = True
    
    grid_multiple = 1  # do not aggregate any of the grids/data crreated -- save verbatim
    
    grid_dest = grid * grid_multiple

    eliminate_blanks_for_mid_extra_cut = False  # typically False, only true for the extra data
    if eliminate_blanks_for_mid_extra_cut:
        crop=12 # leading to 200x200 image cut 10 x 10 times 
        model_grid_name = "8g10"
    else:
        crop=22 #provigind with grid 9x20 9x20
        model_grid_name = "8g9" #"16g3"  # was 16g9
    
    dim = dim    
    
    include_ba_partial_height = dim//2
    grid_passes = 1 # was 10  # gp10 standard must be 1
    if grid_passes > 1:
        model_grid_name = "8g10x%s" % grid_passes
    elif grid_passes < 1:
        grid_passes = 1
    print ("grid_passes, include_ba_partial_height, model_grid_name: ", grid_passes, include_ba_partial_height, model_grid_name)
               
    data_generation=True
    testPrint = False
    if data_generation:  # DO ONE BY ONE as convert_scans_and_masks_xd requires a lot of memory and can swap ...
      exclude_blanks = False if create_main_grid else True  # replaces the manual change done in the interactive mode
      include_below_above_nodule = False 
      if not include_below_above_nodule:
          ba0 = dim //2 - include_ba_partial_height
          ba1 = np.min([dim //2 + include_ba_partial_height + 1, dim])

      split_into_nodules_and_blanks = True
      
      for pt in range(0,3):  # splitting into 2 parts due to memory needs

        np.random.seed(1000+pt)
        scans_all_grid = []
        masks_all_grid = []
        scans_all_grid2 = []
        masks_all_grid2 = []
        scans_all_grid3 = []
        masks_all_grid3 = []
        
        
        if pt == 0:
            istart = 4*pt
            iend = 4*(pt+1)
        elif pt == 1:
            istart = 4*pt
            iend = 4*(pt+1)
            iend += 1  # increase by 1 to cover 9
        else:
            istart = 9
            iend = 10

        for i in range(istart, iend):
            scans_all = []
            masks_all = []
            sids_all = []
            scans_all2 = []
            masks_all2 = []
            sids_all2 = []
            scans_all3 = []
            masks_all3 = []
            sids_all3 = []

            print ("\n\n################################# EXECUTING subset ", i)
            scans, masks, sids, blankids = load_scans_masks_or_blanks(i, useAll = False, use_unsegmented=DO_NOT_USE_SEGMENTED)
            
            if include_below_above_nodule:
                 only_with_nudels = True # This may be False or True must be False so we do not loose the info
            else:
                 only_with_nudels = True # could be True ...
            
            for j in range(len(scans)):
                
                extra_test=False
                if extra_test:
                    mtemp = masks[j]
                    np.sum(mtemp)
                    np.min(mtemp)
                    idx = np.sum(masks[j], axis=(1,2)) != 0    # at this stage, with this more memory friendly version there should be only items with nodules
                    idx_nodules =  np.sum(masks[j], axis=(1,2)) > 0  
                    idx_blanks =   np.sum(masks[j], axis=(1,2)) < 0    
                    print ("Masks, with nodules and blanks: ", np.sum(idx_nodules), np.sum(idx_blanks))

                blanks_per_axis = 0  # we now rnadomly position this 
                scans1 = [scans[j]]
                masks1 = [masks[j]]
                
                use_standard_convert = True if recreate_grid8_March_data else False  # added for backward compatbility
                
                if use_standard_convert:         
                    scans1, masks1 = convert_scans_and_masks_xd3 (scans1, masks1, only_with_nudels = only_with_nudels, dim=dim,  crop=crop, blanks_per_axis = blanks_per_axis,
                                                                 add_blank_spacing_size=1, add_blank_layers = 0)  # as per March data generation
    
                    if not include_below_above_nodule:
                        ### take the centrale values                       
                        idx = np.sum(np.abs(masks1[:,ba0:ba1]), axis=(-1,-2, -3)) != 0    #dim // 2
                        idx_nodules =  np.sum(masks1[:,ba0:ba1], axis=(-1,-2, -3)) > 0  
                        idx_blanks =   np.sum(masks1[:,ba0:ba1], axis=(-1,-2, -3)) < 0  
                    else:
                        idx = np.sum(np.abs(masks1), axis=(-1,-2,-3)) != 0  
                        idx_nodules =  np.sum(masks1, axis=(-1,-2,-3)) > 0  
                        idx_blanks =   np.sum(masks1, axis=(-1,-2,-3)) < 0 
                                             
                    count_nodules = np.sum(idx_nodules)
                    count_blanks = np.sum(idx_blanks)
                    count_all =  np.sum(idx, axis=0)
    
                    print ("sidj, Total masks, and with nodules and blanks: ", sids[j], len(idx), count_nodules, count_blanks)
                    if (count_nodules == 0):
                        # cut down the blanks only to the centrally located, whatever the include_below_above_nodule
                        idx_blanks =   np.sum(masks1[:,dim // 2], axis=(-1,-2)) < 0  
                        count_blanks = np.sum(idx_blanks)
                        print("Selecting only the central blanks, count of: ", count_blanks)
                        
                        masks1 = masks1[idx_blanks]
                        scans1 = scans1[idx_blanks]
                    elif not include_below_above_nodule:
                        #print("Not including the below and above nodules' entries, beyond partial_height of , remaining count: ", include_ba_partial_height, count_all)
                        print("Using ba partial_height; remaining count: ", count_all)                      
                        
                        masks1 = masks1[idx]
                        scans1 = scans1[idx]
                    else:
                        print("Keeping all entries of: ", count_all )
    
                else:
                    ## just convert into 3d rep and find the vosel in the entire space 
                    scans1, masks1 = convert_scans_and_masks_xd3 (scans1, masks1, only_with_nudels = False, dim=dim,  crop=crop, blanks_per_axis = blanks_per_axis,
                                                                 add_blank_spacing_size=0, add_blank_layers = 0)
                                                                 
            
                scans1 = scans1[:, np.newaxis]  # do NOT change these as we iterate by different grids now 20170327
                masks1 = masks1[:, np.newaxis]  # do NOT change these as we iterate by different grids now 20170327
                
                for ig in range(len(grids)):
                    grid_masks = []
                    grid_scans = []
                    grid = grids[ig]
                    crop12 = crops2[ig]  
                            
                    if exclude_blanks and np.sum(masks1) <0:
                        print("Completely excluding blanks & gridding of them ...")
                        scans1_c = []
                        masks1_c = []
                    else:
                        for gpass in range(grid_passes):
                            
                            if grid_passes != 1:
                                shift = grid // grid_passes
                                shifting_gridwith = img_cols // grid - 1 # minus 1 to accomodate the shift
                                crop_top_left = (img_cols - (shifting_gridwith+1)*grid) // 2 + gpass*shift
                                crop_bottom_right = crop_top_left + shifting_gridwith*grid
                                
                                masks1_c = masks1[:,:,:,crop_top_left:crop_bottom_right,crop_top_left:crop_bottom_right]
                                scans1_c = scans1[:,:,:,crop_top_left:crop_bottom_right,crop_top_left:crop_bottom_right]                         
                                
                                
                                if recreate_grid8_March_data:
                                    grid_masks1, gridwidth, gridheight = grid_data(masks1_c, grid=grid, crop=0, expand=0 )
                                    grid_scans1, gridwidth, gridheight = grid_data(scans1_c, grid=grid, crop=0, expand=0)
                                else:
                                    #### NOTE the following has NOT been tested
                                    print("WARNING: grid_passes option has NOT been tested working with the find_voxels procedure")
                                    grid_scans1, grid_masks1 = find_voxels_and_blanks(dim, grid, scans1_c, scans1_c, masks1_c, nodules_threshold=0.999, voxelscountmax = 1000, 
                                                                      find_blanks_also = True, centralcutonly = centralcutonly, diamin=diamins[ig], diamax=diamaxs[ig])

                            
                            else:  # just a single standard pass - no shifting grid      
                                if recreate_grid8_March_data:
                                    grid_masks1, gridwidth, gridheight = grid_data(masks1, grid=grid, crop=crop12, expand=0 )
                                    grid_scans1, gridwidth, gridheight = grid_data(scans1, grid=grid, crop=crop12, expand=0)
                                else:
                                    grid_scans1, grid_masks1 = find_voxels_and_blanks(dim, grid, scans1, scans1, masks1, nodules_threshold=0.999, voxelscountmax = 1000, 
                                                                      find_blanks_also = True, centralcutonly = centralcutonly, diamin=diamins[ig], diamax=diamaxs[ig])                                            
                                    
                                testPlot = False
                                if testPlot:
                                    for ii in range(0, len(grid_scans1)):  # was 2, 20
                                        print ('gridscans1 scan/cut '+str(ii))
                                        f, ax = plt.subplots(1, 2, figsize=(8,4))
                                        ax[0].imshow(grid_scans1[ii,0,dim // 2],cmap=plt.cm.gray)
                                        #ax[1].imshow(masks_pred[ii,0,0],cmap=plt.cm.gray)
                                        ax[1].imshow(grid_masks1[ii,0,dim // 2] ,cmap=plt.cm.gray)
                                        #ax[2].imshow(np.abs(masks_pred[ii,0,0] - masks_pred_prev[ii,0,0]) ,cmap=plt.cm.gray)
                                        #ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                                        plt.show()

                                                                                                                             
                            if len(grid_masks1)   > 0:                   
                                idx_blanks =   np.sum(grid_masks1[:,:,dim // 2], axis=(-1,-2, -3)) < 0  
                                idx = np.sum(np.abs(grid_masks1), axis=(1,2,3,4)) != 0  
                                
                                if not include_below_above_nodule:
                                    idx_nodules =  np.sum(grid_masks1[:,:,ba0:ba1], axis=(1,2,3,4)) > 0  
                                else:
    
                                    idx_nodules =  np.sum(grid_masks1, axis=(1,2,3,4)) > 0  # this may be inaccurate whene blanks was somewhere there
                                 # cut down the blanks only to the centrally located
        
                                if testPrint: 
                                    print ("Total masks (after grid), and with nodules and blanks: ", len(idx), np.sum(idx_nodules), np.sum(idx_blanks))
                 
                                idx_nodules_central_blanks = idx_nodules | idx_blanks
                                if exclude_blanks:
                                    if testPrint: 
                                         print("Not including blanks ....")
                                    grid_masks1 = grid_masks1[idx_nodules]  
                                    grid_scans1 = grid_scans1[idx_nodules]
                                else:
                                    grid_masks1 = grid_masks1[idx_nodules_central_blanks]  # ONLY keep the masks and scans with nodules(central)
                                    grid_scans1 = grid_scans1[idx_nodules_central_blanks]
                                if testPrint: 
                                    print ("Total masks (after another central blanks cut): ", len(grid_masks1))

                                grid_masks.append(grid_masks1)
                                grid_scans.append(grid_scans1)
                                    
                        if len(grid_masks):
                            masks1_c = np.concatenate(grid_masks)
                            scans1_c = np.concatenate(grid_scans)
                        else:
                            masks1_c = []
                            scans1_c = []
                        print ("=== Grid, Sub-total masks1 : ", (grid, len(masks1_c)))

                    if (len(masks1_c) > 0): 
                        if ig == 0:
                            scans_all.append(scans1_c)
                            masks_all.append(masks1_c)
                            sids_all.append(sids[j])  # ????
                        elif ig == 1:
                            scans_all2.append(scans1_c)
                            masks_all2.append(masks1_c)
                            sids_all2.append(sids[j])  # ????
                        elif ig == 2:
                            scans_all3.append(scans1_c)
                            masks_all3.append(masks1_c)
                            sids_all3.append(sids[j])  # ???
                        else:
                            print("Warning: 4 separate grids are not implemented for automatic data generation")
                ## end of the grid_and_limit_data LOOP --------------------------------------------------------
                
        
            scans =  np.concatenate(scans_all)      #e.g. [0:4])
            masks =  np.concatenate(masks_all)      #[0:4])
            
            if len(grids) > 1:
                scans2 = np.concatenate(scans_all2)
                masks2 =  np.concatenate(masks_all2) 
            if len(grids) > 2:
                scans3 = np.concatenate(scans_all3)
                masks3 =  np.concatenate(masks_all3) 
 
            ################### end o the scans loop  #### ######################################################
            ig =0
            for ig in range(len(grids)):
                if ig == 0:
                    scansx = scans
                    masksx = masks
                elif ig == 1:
                    scansx = scans2
                    masksx = masks2
                elif ig == 2:
                    scansx = scans3
                    masksx = masks3
               
                # select only non-zero grids .. (essentially decimating the data; for subset 1: from 17496 dow to 1681)
                idx = np.sum(np.abs(masksx), axis=(1,2,3,4)) != 0    # at this stage, with this more memory friendly version there should be only items with nodules
                idx_nodules =  np.sum(masksx, axis=(1,2,3,4)) > 0  
                idx_blanks =   np.sum(masksx, axis=(1,2,3,4)) < 0    
                
                count_nodules = np.sum(idx_nodules)
                count_blanks = np.sum(idx_blanks)
                count_all =  np.sum(idx, axis=0)
    
                print ("All entries, grid,  total, nodules and blanks: ", grids[ig], len(idx), count_all, count_nodules, count_blanks)
                
                testPlot = False
                if testPlot:
                    jump=len(idx) // 20
                    jump =1
                    for ii in range(0, len(idx)//20, jump):  # was 2, 20
                        print ('scan/cut '+str(ii))
                        f, ax = plt.subplots(1, 2, figsize=(8,4))
                        ax[0].imshow(masksx[ii,0,dim // 2],cmap=plt.cm.gray)
                        ax[1].imshow(scansx[ii,0,dim // 2] ,cmap=plt.cm.gray)
                        plt.show()
 
                len(masksx)
                if not include_below_above_nodule:  
                    masksx = masksx[idx]
                    scansx = scansx[idx]
                len(masksx)
    
    
                if eliminate_blanks_for_mid_extra_cut:
                    masksx = masksx[~ idx_blanks]
                    scansx = scansx[~ idx_blanks]
                    idx = np.sum(np.abs(masksx), axis=(1,2,3,4)) != 0    # at this stage, with this more memory friendly version there should be only items with nodules
                    idx_nodules =  np.sum(masksx, axis=(1,2,3,4)) > 0  
                    idx_blanks =   np.sum(masksx, axis=(1,2,3,4)) < 0    
                    
                    count_nodules = np.sum(idx_nodules)
                    count_blanks = np.sum(idx_blanks)
                    count_all =  np.sum(idx, axis=0)
        
                    print ("Final for the Extra cut: All entries, grid, total, nodules and blanks: ", grids[ig], len(idx), count_all, count_nodules, count_blanks)
          
                if (len(scansx)>0):
                    if ig == 0:
                        scans_all_grid.append(scansx)
                        masks_all_grid.append(masksx)
                    elif ig == 1:
                        scans_all_grid2.append(scansx)
                        masks_all_grid2.append(masksx)
                    elif ig == 2:
                        scans_all_grid3.append(scansx)
                        masks_all_grid3.append(masksx)
        
        scans =  np.concatenate(scans_all_grid)      #[0:4])
        masks =  np.concatenate(masks_all_grid)      #
        if len(grids) > 1:
            scans2 =  np.concatenate(scans_all_grid2)      
            masks2 =  np.concatenate(masks_all_grid2)
        if len(grids) > 2:
            scans3 =  np.concatenate(scans_all_grid3)      
            masks3 =  np.concatenate(masks_all_grid3)  
            
        idx = np.sum(np.abs(masks), axis=(1,2,3,4)) > 0
        sum(idx)
        print ("FINAL result part %s  ==========================================" % pt)
        print ("grid1, scans.shape ", grids[0], scans.shape[0])
        print ("grid1, masks.shape ", grids[0], masks.shape[0])
        if len(grids) > 1:
            print ("grid2, scans.shape ", grids[1], scans2.shape[0])
            print ("grid2, masks.shape ", grids[1], masks2.shape[0])
        if len(grids) > 2:
            print ("grid3, scans.shape ", grids[2], scans3.shape[0])
            print ("grid3, masks.shape ", grids[2], masks3.shape[0])
                
                 
        del(scans_all)
        del(masks_all)
        del(scans_all_grid)
        del(masks_all_grid)
        
        
        del(scans_all2)
        del(masks_all2)
        del(scans_all_grid2)
        del(masks_all_grid2) 
        if len(grids) > 2:
            del(scans_all3)
            del(masks_all3)
            del(scans_all_grid3)
            del(masks_all_grid3)
            ### eliminate incorrectly segmented scans ...
        no_need_to_eliminate_and_blanks_exist = True
        if not no_need_to_eliminate_and_blanks_exist:
            scans, masks = eliminate_incorrectly_segmented (scans, masks)
            scans2, masks2 = eliminate_incorrectly_segmented (scans2, masks2)
            if len(grids) > 2:
                scans3, masks3 = eliminate_incorrectly_segmented (scans3, masks3)
                
        
        do_not_correct_masks_leave_for_potential_processing = True
        #### SPLITTING into nodules(+ba) and blanks files 
        if split_into_nodules_and_blanks:
            for ig in range(len(grids)):
                if ig == 0:
                    masksx = masks
                    scansx = scans
                elif ig == 1:
                    masksx = masks2
                    scansx = scans2
                elif ig == 2:
                    masksx = masks3
                    scansx = scans3
                    
                idx = np.sum(np.abs(masksx), axis=(1,2,3,4)) != 0    # at this stage, with this more memory friendly version there should be only items with nodules
                idx_nodules =  np.sum(masksx, axis=(1,2,3,4)) > 0  
                idx_blanks =   np.sum(masksx, axis=(1,2,3,4)) < 0    
                           
                
                count_nodules = np.sum(idx_nodules)
                count_blanks = np.sum(idx_blanks)
                count_all =  np.sum(idx, axis=0)
                
                masks_blanksx = masksx[idx_blanks]
                scans_blanksx = scansx[idx_blanks]
      
                masksx = masksx[~ idx_blanks]
                scansx = scansx[~ idx_blanks]
                
                # ensure no blanks are marked anywhere
                if not do_not_correct_masks_leave_for_potential_processing:
                    masks_blanksx[masks_blanksx] = 0  # ### NOW set the blank masks to 0
                    masksx[masksx < 0] = 0            # just in case, should not contain any blanks with the split as above
      
                print ("Grid, Splitting into mask and masks_blanks, shapes: ", grids[ig], masksx.shape, masks_blanksx.shape)
                print ("Grid, Splitting into scans and scans_blanks, shapes: ", grids[ig], scansx.shape, scans_blanksx.shape)
                if ig == 0:
                    masks = masksx
                    scans = scansx
                    masks_blanks = masks_blanksx
                    scans_blanks = scans_blanksx
                elif ig == 1:
                    masks2 = masksx
                    scans2 = scansx
                    masks_blanks2 = masks_blanksx
                    scans_blanks2 = scans_blanksx
                elif ig == 2:
                    masks3 = masksx
                    scans3 = scansx
                    masks_blanks3 = masks_blanksx
                    scans_blanks3 = scans_blanksx
        else:
           
            
            if not do_not_correct_masks_leave_for_potential_processing:
                 masks[masks < 0] = 0   ### NOW just set the blank masks to 0
                 masks[masks < 0] = 0   ### NOW just set the blank masks to 0

        ###  combingin blanks into a multi-grid data ...
        combine_blanks_into_mgrid = False  ### NOTE we do NOT extend it into the grids approach !
        if combine_blanks_into_mgrid:
             ##### combine 3x3 blanks for 120x120 net 
            grid_dest = 120
            gridwith_dest = grid_dest // grid  
            gridheight_dest = grid_dest // grid
            # calculate the length divisible by the size of multi-grid 
            len_dest = len(scans) // (gridwith_dest*gridheight_dest) * ((gridwith_dest*gridheight_dest))
            scans = scans[:len_dest] 
            masks = masks[:len_dest]
            
            scans = data_from_grid_by_proximity(scans, gridwith_dest, gridheight_dest, grid=grid)
            masks =  data_from_grid_by_proximity(masks, gridwith_dest, gridheight_dest, grid=grid)  # 13942 
            print ("FINAL MULTI-GRID result part %s  ==========================================" % pt)
            print (scans.shape)
            print(masks.shape)

        masks_out_base_dir_format = "../luna/models/masks_d%sg%sx%sba%s%s_%s"
        masks_out_base_name = masks_out_base_dir_format % (str(dim), str(grid_multiple),str(grids[ig]) , str(include_ba_partial_height), grid8_March_data_str, resolution_str)
        scans_out_base_name = masks_out_base_name.replace("masks", "scans", 1)
  
        pname_nodules = "_nodules_%s_%s_" % (str(istart), str(iend-1))
        pname_blanks = "_blanks_%s_%s_" % (str(istart), str(iend-1))
            
        for ig in range(len(grids)):             
            masks_out_base_name = masks_out_base_dir_format % (str(dim), str(grid_multiple),str(grids[ig]) , str(include_ba_partial_height), grid8_March_data_str, resolution_str)
            scans_out_base_name = masks_out_base_name.replace("masks", "scans", 1)
            if ig == 0:
                 masks_count = len(masks)
                 scans_count = len(scans)
            elif ig == 1:
               masks_count = len(masks2)
               scans_count = len(scans2)
            elif ig ==2:
               masks_count = len(masks3)
               scans_count = len(scans3)
            masks_name = ''.join((masks_out_base_name, pname_nodules, "%s" % str(masks_count)) )
            scans_name = ''.join((scans_out_base_name, pname_nodules, "%s" % str(scans_count)) )              
            print ("Saving: ", masks_name)
            print ("Saving: ", scans_name)
            if ig == 0:
                np.savez_compressed (masks_name, masks)  
                np.savez_compressed (scans_name, scans)  
            elif ig == 1:
                np.savez_compressed (masks_name, masks2)   
                np.savez_compressed (scans_name, scans2)   
            elif ig == 2:
                np.savez_compressed (masks_name, masks3)   
                np.savez_compressed (scans_name, scans3)   
           
        if split_into_nodules_and_blanks:
           
            for ig in range(len(grids)):
                masks_out_base_name = masks_out_base_dir_format % (str(dim), str(grid_multiple),str(grids[ig]) , str(include_ba_partial_height), grid8_March_data_str, resolution_str)
                scans_out_base_name = masks_out_base_name.replace("masks", "scans", 1)
                if ig == 0:
                     masks_count = len(masks_blanks)
                     scans_count = len(scans_blanks)
                elif ig == 1:
                   masks_count = len(masks_blanks2)
                   scans_count = len(scans_blanks2)
                elif ig ==2:
                   masks_count = len(masks_blanks3)
                   scans_count = len(scans_blanks3)
                masks_name = ''.join((masks_out_base_name, pname_blanks, "%s" % str(masks_count)) )
                scans_name = ''.join((scans_out_base_name, pname_blanks, "%s" % str(scans_count)) )              
                print ("Saving: ", masks_name)
                print ("Saving: ", scans_name)                  
                if ig == 0:
                    np.savez_compressed (masks_name, masks_blanks)  # 
                    np.savez_compressed (scans_name, scans_blanks)  # 
                elif ig == 1:
                    np.savez_compressed (masks_name, masks_blanks2)  # 
                    np.savez_compressed (scans_name, scans_blanks2)  #     
                elif ig == 2:
                    np.savez_compressed (masks_name, masks_blanks3)  # 
                    np.savez_compressed (scans_name, scans_blanks3)  #  

        del(scans) 
        del(masks)
        if len(grids) > 1:   
            del(scans2)
            del(masks2)   
        if len(grids) > 2:   
            del(scans3)
            del(masks3) 
                        
        del(scans_blanks) # remove them as they may take a lot of memory that we may need for the next iteration to create the files above
        del(masks_blanks)
        if len(grids) > 1:
            del(scans_blanks2) # remove them as they may take a lot of memory
            del(masks_blanks2)   
        if len(grids) > 2:
            del(scans_blanks3) # remove them as they may take a lot of memory
            del(masks_blanks3)   



simple_luna_data_availability = True
if simple_luna_data_availability:
    print("simple_luna_data_validation")
    ### check for the availability of data, and set-up correctnes
    ### IF you get an errot "IndexError: list index out of range" around files[12] ---> it means your /input/ directory or similar is wrong or data has not been extracted etc.
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))
    file = files[12]  # 1 - 2 elements, 12 - 3 nodules; it would fail if soemthing is wrong with the path or directory structure
    imagePath = file
    seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
    seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid

    luna_subset = 0       # initial 
    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()
    cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
    print (cands)
    
    print("mean diameter for nodules in annotations: ", np.mean(annotations["diameter_mm"]))
    


if __name__ == '__main__':
    
    
    # Part 1, Option 3 step    
    dim = 8
    recreate_grid8_March_data = True
    
    # Part 1, Option 2: To recreate data for that option uncomment the next line ...
    #recreate_grid8_March_data = False # switch to obtain intermediary data for Part 1, Option 2 
    
    print ("Starting:  create_unet_training_files, recreating grid8 data: ", recreate_grid8_March_data)
    start_time = time.time()
    
    create_unet_training_files(dim, recreate_grid8_March_data)
    print ("Completed, create_unet_training_files, time: ", time.time()-start_time)


    # PART 2 - prepare competitions data, for stage 1 and stage 2  
    part = 0
    processors = 1          # you may run several of these jobs; define processors to say 4, and start 4 separate jobs with part = 0, 1, 2, 3 respectively
    showSummaryPlot = True
    for stage in ["stage1", "stage2"]:  
        
        start_time = time.time()
        print ("Starting segmentation, stage, part of a set/processors: ", stage, part, processors)
        part, processors, count = segment_all(stage, part, processors, showSummaryPlot)
        print ("Completed, part, processors,total count tried, total time: ", stage, part, processors, count, time.time()-start_time)
                         
