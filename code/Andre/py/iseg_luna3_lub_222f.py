"""
Created on Thu Jan 26 17:04:11 2017

Preprocess Luna datasets and create nodule masks (and/or blank subsets)

NOTE that:
    1. we do NOT segment the lungs at all -- we will use the raw images for training (DO_NOT_SEGMENT = True)
    2. No corrections are made to the nodule radius in relation to the thickness of the layers (radius = (ca[4])/2, simply)
    
@author: Andre Stochniol, andre@stochniol.com
Some functions have reused from the respective examples/kernels openly published at the https://www.kaggle.com/arnavkj95/data-science-bowl-2017/ , as referenced within the file
"""

#%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.ndimage as ndimage

import scipy.ndimage # added for scaling\

import cv2
import time
import glob


from skimage import measure, morphology, segmentation
import SimpleITK as sitk

DO_NOT_SEGMENT = True       #### major difference with the initial/Feb version
RESIZE_SPACING = [2,2,2] 
###                 z, y, x  (x & y MUST be the same)


luna_subset = 0       # initial 
LUNA_BASE_DIR = "../luna/data/original_lungs/subset%s/"  # added on AWS; data as well 
LUNA_DIR = LUNA_BASE_DIR % luna_subset
CSVFILES = "../luna/data/original_lungs/CSVFILES/%s"
LUNA_ANNOTATIONS = CSVFILES % "annotations.csv"
LUNA_CANDIDATES =  CSVFILES % "candidates.csv"

  
MARKER_INTERNAL_THRESH = -400  # was -400; maybe use -320 ??
MARKER_FRAME_WIDTH = 9      # 9 seems OK for the half special case ...
def generate_markers(image):
    #Creation of the internal Marker
    
    useTestPlot = False
    if useTestPlot:
        timg = image
        plt.imshow(timg, cmap='gray')
        plt.show()

    add_frame_vertical = True  # NOT a good idea; no added value
    if add_frame_vertical:   # add frame for potentially closing the lungs that touch the edge, but only vertically
                        
        fw = MARKER_FRAME_WIDTH  # frame width (it looks that 2 is the minimum width for the algorithms implemented here, namely the first 2 operations for the marker_internal)
   
        xdim = image.shape[1]
        #ydim = image.shape[0]
        img2 = np.copy(image)
    
        
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



BINARY_CLOSING_SIZE = 7     ## added for tests; 5 for disk seems sufficient - fo safety let's go with 6 or even 7
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

#image = image_slices[70]
def seperate_lungs_cv2(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
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
    

    blackhat_struct = ndimage.iterate_structure(blackhat_struct, rescale_n(8,reduce_factor))  # dyanmically adjust the number of iterattions; original was 8
    
    blackhat_struct_cv2 = blackhat_struct.astype(np.uint8)
    #Perform the Black-Hat
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
    structure3 = morphology.disk(rescale_n(BINARY_CLOSING_SIZE,reduce_factor)) # dynanically adjust; better , 5 seems sufficient, we use 7 for safety/just in case
    
    
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
    
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)   # nor mode= "wrap"/xxx, nor cval=-1024 can ensure that the min and max values are unchanged .... # cval added
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')  ### early orig modified 
    #image = scipy.ndimage.zoom(image, real_resize_factor, order=1)    # order=1 bilinear , preserves the min and max of the image -- pronbably better for us (also faster than spkine/order=2)
    
    #image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest', order=1)    # order=1 bilinear , preserves the min and max of the image -- pronbably better for us (also faster than spkine/order=2)
    
    return image, new_spacing

def segment_one(image_slices):
                       
    useTestPlot = False
    if useTestPlot:
        print("Shape before segmenting\t", image_slices.shape)
        plt.hist(image_slices.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
  
    shape = image_slices.shape
    l_segmented = np.zeros(shape).astype(np.int16)
    l_lungfilter = np.zeros(shape).astype(np.bool)
    l_outline = np.zeros(shape).astype(np.bool)
    l_watershed = np.zeros(shape).astype(np.int16)
    l_sobel_gradient = np.zeros(shape).astype(np.float32)
    l_marker_internal = np.zeros(shape).astype(np.bool)
    l_marker_external = np.zeros(shape).astype(np.bool)
    l_marker_watershed = np.zeros(shape).astype(np.int16)  
    
    i=0
    for i in range(shape[0]):
        l_segmented[i], l_lungfilter[i], l_outline[i], l_watershed[i], l_sobel_gradient[i], l_marker_internal[i], l_marker_external[i], l_marker_watershed[i] = seperate_lungs_cv2(image_slices[i])
        
    
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

    mask = l_lungfilter.astype(np.int8)
    
    regions = measure.regionprops(mask)  # this measures the largest region and may lead to incorrect results when the mask is not the largest region !!!

 
    bb = regions[0].bbox
    #print(bb)
    zlen = bb[3] - bb[0]
    ylen = bb[4] - bb[1]
    xlen = bb[5] - bb[2]
    
    dx = 0  
    ## have to reduce dx to 0 as for instance at least one image of the lungs stretch right to the border even without cropping 
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
    
    #maskOK = False
    if bxy_min >0 and bxy_max < 512 and mask_volume_check and zlen/mask_shape[0] > crop_max_ratio_z and ylen/mask_shape[1] > crop_max_ratio_y and xlen/mask_shape[2]  > crop_max_ratio_x:
        # mask OK< crop the image and mask
        ### full crop
        #image = image[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        #mask =   mask[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        
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
        print("Mask too small, NOT auto-cropping x-y: shape, cropped, bbox, ratios, violume:", mask_shape, image.shape, bb, zlen/mask_shape[0], ylen/mask_shape[1], xlen/mask_shape[2], mask_volume)


    else:
        image = l_segmented[0:mask_shape[0], dx: image_shape[1] - dx, dx: image_shape[2] - dx]
        #mask =   mask[0:mask_shape[0], dx: mask_shape[1] - dx, dx: mask_shape[2] - dx]
        print("Mask wrong, NOT auto-cropping: shape, cropped, bbox, ratios, volume:", mask_shape, image.shape, bb, zlen/mask_shape[0], ylen/mask_shape[1], xlen/mask_shape[2], mask_volume)
        
    
    useSummaryPlot = True
    if useSummaryPlot:
        
        img_sel_i = shape[0] // 2
        # Show some slice in the middle
        plt.imshow(l_segmented[img_sel_i], cmap='gray')
        plt.show()
        
    return l_segmented, image


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
		radius = (ca[4])/2   # VERSION iseg_luna3 - DO NOT CORRECT the radiius in ANY way ...!!
             
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

		#create the mask
		for x in noduleRange_x:
			for y in noduleRange_y:
				for z in noduleRange_z:
					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
					#if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:  ### original (constrained to a uniofrm RESIZE)
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

luna_subset = 0       # initial 
LUNA_DIR = LUNA_BASE_DIR % luna_subset

files = glob.glob(''.join([LUNA_DIR,'*.mhd']))
file = files[12]  # rough empty set test - if file is empty this would fail; 12th - 3 nodules
imagePath = file
seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid

print ("Luna annotations (head)")
annotations =    pd.read_csv(LUNA_ANNOTATIONS)
annotations.head()
cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
print (cands)

def create_nodule_mask(imagePath, cands):
    #if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)

    #calculate resize factor
    resize_factor = spacing / RESIZE_SPACING         # was [1, 1, 1]
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize 
	
    start = time.time()
    #resize image     
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize,  mode='nearest')  # Andre mode added
    if DO_NOT_SEGMENT:
        lung_seg = lung_img
        lung_seg_crop = lung_img
        print("Rescale time, and path: ", ((time.time() - start)), imagePath )

    else:
        lung_seg, lung_seg_crop = segment_one(lung_img)
        print("Rescale & Seg time, and path: ", ((time.time() - start)), imagePath )

    useTestPlot = False
    if useTestPlot:
        plt.hist(img.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        
        
        plt.hist(lung_img.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        plt.hist(lung_seg.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
  
    
        img_sel_i = img.shape[0] // 2
        # Show some slice in the middle
        plt.imshow(img[img_sel_i], cmap=plt.cm.gray)
        plt.show()

        img_sel_i = lung_seg.shape[0] // 2
        # Show some slice in the middle
        plt.imshow(lung_seg[img_sel_i], cmap='gray')
        plt.show()
        
        # Show some slice in the middle
        plt.imshow(lung_seg_crop[img_sel_i], cmap='gray')
        plt.show()

	#create nodule mask
    nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)
    
    if useTestPlot:    
        lung_img.shape
        lung_seg.shape
        lung_seg_crop.shape
        nodule_mask.shape
    
        for i in range(nodule_mask.shape[0]):
            print ("Slice: ", i)        
            plt.imshow(nodule_mask[i], cmap='gray')
            plt.show()
    
    
        img_sel_i = 146  # 36
        plt.imshow(lung_seg[img_sel_i], cmap=plt.cm.gray)
        plt.show()
        
        plt.imshow(nodule_mask[img_sel_i], cmap='gray')
        plt.show()
    
    
        for i in range (141, 153):
            print ("Slice: ", i)        
            plt.imshow(lung_seg[i], cmap='gray')  
            plt.show()
            #plt.imshow(nodule_mask[i], cmap='gray')
            #plt.show()

    w448 = int(448 // RESIZE_SPACING[1])  # we use 448 as this would be not enough just for 3 out of 1595 patients giving the pixels resolution ...:
    #lung_img_448, lung_seg_448, nodule_mask_448 = np.zeros((lung_img.shape[0], w448, w448)), np.zeros((lung_seg.shape[0], w448, w448)), np.zeros((nodule_mask.shape[0], w448, w448))
    lung_img_448 = np.full ((lung_img.shape[0], w448, w448), -2000,  dtype=np.int16)
    lung_seg_448 = np.full ((lung_seg.shape[0], w448, w448), -2000,  dtype=np.int16)
    nodule_mask_448 = np.zeros((nodule_mask.shape[0], w448, w448), dtype=np.int16)


    original_shape = lung_img.shape	
    if (original_shape[1] > w448):
        ## need to crop the image to w448 size ...
    
        print("Warning: additional crop from ... to width of: ", original_shape, w448)
        offset = (w448 - original_shape[1])
        
        y_min = abs(offset // 2 ) ## we use the same diff order as for offset below to ensure correct cala of new_origin (if we ever neeed i)
        y_max = y_min + w448
        lung_img = lung_img[:,y_min:y_max,:]
        lung_seg = lung_seg[:,y_min:y_max,:]
        nodule_mask = nodule_mask[:,y_min:y_max,:]
        
        upper_offset = offset// 2
        lower_offset = offset - upper_offset
        
        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)
        origin = new_origin
        original_shape = lung_img.shape
        
    if (original_shape[2] > w448):
        x_min = (original_shape[2] - w448) // 2
        x_max = x_min + w448
        lung_img = lung_img[:,:,x_min:x_max]
        lung_seg = lung_seg[:,:,x_min:x_max]
        nodule_mask = nodule_mask[:,:,x_min:x_max]
        original_shape = lung_img.shape
    
    offset = (w448 - original_shape[1])
    upper_offset = offset// 2
    lower_offset = offset - upper_offset   
    new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

    if offset > 0:      #    
        for z in range(lung_img.shape[0]):
            
            ### if new_origin is used check the impact of the above crop for instance for:
            ### path = "'../luna/original_lungs/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'
            
            lung_img_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
            lung_seg_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_seg[z,:,:]
            nodule_mask_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]
    else:
        lung_img_448 = lung_img   # equal dimensiona, just copy all (no nee to add the originals withion a frame)
        lung_seg_448 = lung_seg
        nodule_mask_448 = nodule_mask
        

    nodule_mask_448_sum = np.sum(nodule_mask_448, axis=0) 
    if useTestPlot:    
        lung_img_448.shape
        lung_seg_448.shape
        #lung_seg_crop.shape
        nodule_mask_448.shape
            
        img_sel_i = 146  # 36
        
        plt.imshow(lung_img_448[img_sel_i], cmap=plt.cm.gray)
        plt.show()
        
        plt.imshow(lung_seg_448[img_sel_i], cmap=plt.cm.gray)
        plt.show()
        
        plt.imshow(nodule_mask_448[img_sel_i], cmap='gray')
        plt.show()
    
    
        for i in range (141, 153):
            print ("Slice: ", i)        
            plt.imshow(lung_seg_448[i], cmap='gray')  
            plt.show()
            #plt.imshow(nodule_mask[i], cmap='gray')
            #plt.show()
    
    useSummaryPlot = True
    if useSummaryPlot:
        mask_sum_mean_x100 = 100 * np.mean(nodule_mask_448_sum) 
        
        axis = 1
        lung_projections = []
        mask_projections = []
        for axis in range(3):
             #sxm_projection = np.max(sxm, axis = axis)
             lung_projections.append(np.mean(lung_seg_448, axis=axis))
             mask_projections.append(np.max(nodule_mask_448, axis=axis))


              
        f, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(lung_projections[0],cmap=plt.cm.gray)
        ax[1].imshow(lung_projections[1],cmap=plt.cm.gray)
        ax[2].imshow(lung_projections[2],cmap=plt.cm.gray)
        plt.show()
        f, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(mask_projections[0],cmap=plt.cm.gray)
        ax[1].imshow(mask_projections[1],cmap=plt.cm.gray)
        ax[2].imshow(mask_projections[2],cmap=plt.cm.gray)
        plt.show()
        
        print ("Mask_sum_mean_x100: ", mask_sum_mean_x100)



    # save images.    
    path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
    
    if DO_NOT_SEGMENT:
        path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)  # data removed from the second part on AWS
    else:
        path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)
  
    if DO_NOT_SEGMENT:
        np.savez_compressed(path_segmented + '_lung', lung_seg_448)   
    else:
        np.savez_compressed(path_segmented + '_lung_seg', lung_seg_448)
        
    np.savez_compressed(path_segmented + '_nodule_mask', nodule_mask_448)

    return

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


def update_nodule_mask_or_blank (imagePath, cands, true_mask=True):
    #if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    # load the old one and copy across
    path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
    
    if DO_NOT_SEGMENT:
        path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)
       
    else:
        path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)
    
    if true_mask:
        # nothing to update reload and copy over
        mask_img_z = np.load(''.join((path_segmented  + '_nodule_mask' + '.npz'))) 
        nodule_mask_448 = mask_img_z['arr_0']
        print("Loading and saving _nodule_mask as _nodule_mask_wblanks for: ", path_segmented)        
        
    else:
        
        img, origin, spacing = load_itk(imagePath)
        
        #calculate resize factor
        resize_factor = spacing / RESIZE_SPACING
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img.shape
        new_spacing = spacing / real_resize 
        	
        # loading of the  image/images to sync with the update  -- DOES NOT WORK
        attempt_through_reloading = False  ## this has failed
        if attempt_through_reloading:
            if DO_NOT_SEGMENT:
                lung_img_z = np.load(''.join((path_segmented  + '_lung' + '.npz'))) 
            else:
                lung_img_z = np.load(''.join((path_segmented  + '_lung_seg' + '.npz'))) 
            
            lung_img = lung_img_z['arr_0']
            
        else:
            ## have to redo the calculations
            start = time.time()
            #resize image     
            lung_img = scipy.ndimage.interpolation.zoom(img, real_resize,  mode='nearest')  # Andre mode added
            if DO_NOT_SEGMENT:
                lung_seg = lung_img
                lung_seg_crop = lung_img
                print("Rescale time, and path: ", ((time.time() - start)), imagePath )
        
            else:
                lung_seg, lung_seg_crop = segment_one(lung_img)
                print("Rescale & Seg time, and path: ", ((time.time() - start)), imagePath )
        
    
        nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)
        
        if not true_mask:
            nodule_mask = -1 * nodule_mask  # mark it as invalid to be zeroed later on (needed to get the blanks)
  
        useTestPlot = False
        if useTestPlot:    
            lung_img.shape
            lung_seg.shape
            lung_seg_crop.shape
            nodule_mask.shape
            
            #mask0 = np.load(''.join((path_segmented  + '_module_mask' + '.npz')))

        
            for i in range(nodule_mask.shape[0]):
                print ("Slice: ", i)        
                plt.imshow(nodule_mask[i], cmap='gray')
                plt.show()
        
    
        w448 = int(448 // RESIZE_SPACING[1])  # we use 448 as this would be not enough just for 3 out of 1595 patients giving the pixels resolution ...:
        nodule_mask_448 = np.zeros((nodule_mask.shape[0], w448, w448), dtype=np.int16)
    
    
        original_shape = lung_img.shape	
        if (original_shape[1] > w448):
            ## need to crop the image to w448 size ...
        
            print("Warning: additional crop from ... to width of: ", original_shape, w448)
            offset = (w448 - original_shape[1])
            
            y_min = abs(offset // 2 ) ## we use the same diff order as for offset below to ensure correct calculations of new_origin (if we ever neeed i)
            y_max = y_min + w448
            nodule_mask = nodule_mask[:,y_min:y_max,:]
            
            upper_offset = offset// 2
            lower_offset = offset - upper_offset
            
            new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)
            origin = new_origin
            original_shape = lung_img.shape
            
        if (original_shape[2] > w448):
            x_min = (original_shape[2] - w448) // 2
            x_max = x_min + w448
            nodule_mask = nodule_mask[:,:,x_min:x_max]
            original_shape = lung_img.shape
        
        offset = (w448 - original_shape[1])
        upper_offset = offset// 2
        lower_offset = offset - upper_offset   
        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)
    
        if offset > 0:      #    
            for z in range(lung_img.shape[0]):
                
                nodule_mask_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]
        else:
            nodule_mask_448 = nodule_mask
            
    
        nodule_mask_448_sum = np.sum(nodule_mask_448, axis=0)
     
        if useTestPlot:    
            nodule_mask_448.shape
            img_sel_i = 146  # 36
                    
            plt.imshow(nodule_mask_448[img_sel_i], cmap='gray')
            plt.show()
        
        
        
        useSummaryPlot = False
        if useSummaryPlot:
            mask_sum_mean_x100 = 100 * np.mean(nodule_mask_448_sum) 
            count_blanks = np.sum(nodule_mask_448 < 0)
            
            axis = 1
            lung_projections = []
            mask_projections = []
            for axis in range(3):
                 #sxm_projection = np.max(sxm, axis = axis)
                 #lung_projections.append(np.mean(lung_seg_448, axis=axis))
                 mask_projections.append(np.max(nodule_mask_448, axis=axis))
                  
            #f, ax = plt.subplots(1, 3, figsize=(15,5))
            #ax[0].imshow(lung_projections[0],cmap=plt.cm.gray)
            #ax[1].imshow(lung_projections[1],cmap=plt.cm.gray)
            #ax[2].imshow(lung_projections[2],cmap=plt.cm.gray)
            #plt.show()
            f, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(mask_projections[0],cmap=plt.cm.gray)
            ax[1].imshow(mask_projections[1],cmap=plt.cm.gray)
            ax[2].imshow(mask_projections[2],cmap=plt.cm.gray)
            plt.show()
            
            print ("Mask_sum_mean_x100, blanks built-in: ", mask_sum_mean_x100, count_blanks)
        
    np.savez_compressed(path_segmented + '_nodule_mask_wblanks', nodule_mask_448)

    return    

def create_nodule_mask_or_blank (imagePath, cands, true_mask=True):
    #if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)

    #calculate resize factor
    resize_factor = spacing / RESIZE_SPACING         # was [1, 1, 1]
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize 
	
    start = time.time()
    #resize image     
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize,  mode='nearest')  # Andre mode added
    if DO_NOT_SEGMENT:
        lung_seg = lung_img
        lung_seg_crop = lung_img
        print("Rescale time, and path: ", ((time.time() - start)), imagePath )

    else:
        lung_seg, lung_seg_crop = segment_one(lung_img)
        print("Rescale & Seg time, and path: ", ((time.time() - start)), imagePath )
    
    useTestPlot = False
    if useTestPlot:
        plt.hist(img.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        
        plt.hist(lung_img.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()

        plt.hist(lung_seg.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
  
    
        img_sel_i = img.shape[0] // 2
        # Show some slice in the middle
        plt.imshow(img[img_sel_i], cmap=plt.cm.gray)
        plt.show()


        img_sel_i = lung_img.shape[0] // 2
        # Show some slice in the middle
        plt.imshow(lung_img[img_sel_i], cmap='gray')
        plt.show()
        
        
        # Show some slice in the middle
        plt.imshow(lung_img[:, 4* lung_img.shape[1] // 6], cmap='gray')
        plt.show()
        
        HU_LUNGS_MIN = -900  # the algo is sensitive to this value -- keep it 900 unless retested
        HU_LUNGS_MAX = -400
        jsteps = 10
        for j in range(jsteps):
              # Show some slice in the middle
              
              img_sel_i = j * lung_img.shape[1] // jsteps
              img_cut = lung_img[:, img_sel_i]
              lix = (img_cut > HU_LUNGS_MIN) & (img_cut < HU_LUNGS_MAX)
              lix_y = np.sum(lix, axis=1)
              print ("Cut & ratio, lix_y (min, mean, max): ", j, j/jsteps, np.min(lix_y),np.mean(lix_y), np.max(lix_y) )
              noise = 3 * np.min(lix_y)
              noise = 0.05 * np.max(lix_y)
              noise = max([3 * np.min(lix_y), 0.05 * np.max(lix_y)])
              print ("Lungs range: ", find_lungs_range(lix_y, noise))
              
              plt.imshow(img_cut, cmap='gray')
              plt.show()
              
              plt.imshow(lix, cmap='gray')
              plt.show()
              
              plt.plot  (lix_y)
              plt.show()

        ymin = int(0.4 * lung_img.shape[1])
        ymax = int(0.6 * lung_img.shape[1])
        zmin_new = lung_img.shape[0] // 2
        zmax_new = lung_img.shape[0] // 2
        j = ymin
        for j in range(ymin, ymax+1):
             img_cut = lung_img[:, j]
             img_cut_lungs = (img_cut > HU_LUNGS_MIN) & (img_cut < HU_LUNGS_MAX)
             lungs_across = np.sum(img_cut_lungs, axis = 1)
             #noise_bottom_some = np.mean(lungs_across[0:5])
             noise = np.max([3*np.min(lungs_across), 0.05 * np.max(lungs_across)])  # experimanetal -- could fail is scan has only central part of lungs and no borders at all -- CHECK
             zmin, zmax = find_lungs_range(lungs_across, noise)
             if zmin < zmin_new:
                 zmin_new = zmin
             if zmax > zmax_new:
                 print ("j, zmax: ", j, zmax)
                 zmax_new = zmax
             
    
             plt.imshow(img_cut, cmap='gray')
             plt.show()
              
             plt.imshow(img_cut_lungs, cmap='gray')
             plt.show()
              
             plt.plot  (lungs_across)
             plt.show()

              
        HU_LUNGS_MIN = -950
        HU_LUNGS_MAX = -400
        ling = img #lung_img # lung_img  # for our testing here
        step = 400
        for HU_LUNGS_MIN in range(-1000, 1000, step):
            HU_LUNGS_MAX = HU_LUNGS_MIN + step
            print ("HU_LUNGS_MIN, HU_LUNGS_MAX: ", HU_LUNGS_MIN, HU_LUNGS_MAX)
        
 
            lix = (ling > HU_LUNGS_MIN) & (ling < HU_LUNGS_MAX)
            lix_z = np.max(lix, axis=0).astype(np.int16)
            
            plt.imshow(lix_z, cmap='gray')
            plt.show()

        HU_LUNGS_MIN = -900
        HU_LUNGS_MAX = -500
        ling = img #lung_img # lung_img  # for our testing here
        print ("HU_LUNGS_MIN, HU_LUNGS_MAX: ", HU_LUNGS_MIN, HU_LUNGS_MAX)
    
        lix = (ling > HU_LUNGS_MIN) & (ling < HU_LUNGS_MAX)
        
        lix_z = np.max(lix, axis=0).astype(np.int16)
        lix_z_x = np.sum(lix_z, axis=0)
        lix_z_y = np.sum(lix_z, axis=1)
        
        plt.imshow(lix_z, cmap='gray')
        plt.show()
        plt.plot  (lix_z_x)
        plt.show()

        plt.plot  (lix_z_y)
        plt.show()
        
        for i in range(0,lung_img.shape[0], 10):
            print("section: ", i)
            plt.imshow(lung_img[i], cmap='gray')
            plt.show()
            

        img_sel_i = lung_seg.shape[0] // 2
        # Show some slice in the middle
        plt.imshow(lung_seg[img_sel_i], cmap='gray')
        plt.show()
        
        # Show some slice in the middle
        plt.imshow(lung_seg_crop[img_sel_i], cmap='gray')
        plt.show()

	#create nodule mask
    #cands.diameter_mm = 3.2
    
    nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)
    
    if not true_mask:
        nodule_mask = -1 * nodule_mask  # mark it as invalid to be zeroed later on (needed to get the blanks)
    #np.sum(nodule_mask)
    if useTestPlot:    
        lung_img.shape
        lung_seg.shape
        lung_seg_crop.shape
        nodule_mask.shape
    
        for i in range(nodule_mask.shape[0]):
            print ("Slice: ", i)        
            plt.imshow(nodule_mask[i], cmap='gray')
            plt.show()
    
    
        img_sel_i = 146  # 36
        plt.imshow(lung_seg[img_sel_i], cmap=plt.cm.gray)
        plt.show()
        
        plt.imshow(nodule_mask[img_sel_i], cmap='gray')
        plt.show()
    
    
        for i in range (141, 153):
            print ("Slice: ", i)        
            plt.imshow(lung_seg[i], cmap='gray')  
            plt.show()
            #plt.imshow(nodule_mask[i], cmap='gray')
            #plt.show()



    w448 = int(448 // RESIZE_SPACING[1])  # we use 448 as this would be not enough just for 3 out of 1595 patients giving the pixels resolution ...:
    #lung_img_448, lung_seg_448, nodule_mask_448 = np.zeros((lung_img.shape[0], w448, w448)), np.zeros((lung_seg.shape[0], w448, w448)), np.zeros((nodule_mask.shape[0], w448, w448))
    lung_img_448 = np.full ((lung_img.shape[0], w448, w448), -2000,  dtype=np.int16)
    lung_seg_448 = np.full ((lung_seg.shape[0], w448, w448), -2000,  dtype=np.int16)
    nodule_mask_448 = np.zeros((nodule_mask.shape[0], w448, w448), dtype=np.int16)


    original_shape = lung_img.shape	
    if (original_shape[1] > w448):
        ## need to crop the image to w448 size ...
    
        print("Warning: additional crop from ... to width of: ", original_shape, w448)
        offset = (w448 - original_shape[1])
        
        y_min = abs(offset // 2 ) ## we use the same diff order as for offset below to ensure correct cala of new_origin (if we ever neeed i)
        y_max = y_min + w448
        lung_img = lung_img[:,y_min:y_max,:]
        lung_seg = lung_seg[:,y_min:y_max,:]
        nodule_mask = nodule_mask[:,y_min:y_max,:]
        
        upper_offset = offset// 2
        lower_offset = offset - upper_offset
        
        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)
        origin = new_origin
        original_shape = lung_img.shape
        
    if (original_shape[2] > w448):
        x_min = (original_shape[2] - w448) // 2
        x_max = x_min + w448
        lung_img = lung_img[:,:,x_min:x_max]
        lung_seg = lung_seg[:,:,x_min:x_max]
        nodule_mask = nodule_mask[:,:,x_min:x_max]
        original_shape = lung_img.shape
    
    offset = (w448 - original_shape[1])
    upper_offset = offset// 2
    lower_offset = offset - upper_offset   
    new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

    if offset > 0:      #    
        for z in range(lung_img.shape[0]):
            
            ### if new_origin is used check the impact of the above crop for instance for:
            ### path = "'../luna/original_lungs/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'
            
            lung_img_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
            lung_seg_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_seg[z,:,:]
            nodule_mask_448[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]
    else:
        lung_img_448 = lung_img   # equal dimensiona, just copy all (no nee to add the originals withion a frame)
        lung_seg_448 = lung_seg
        nodule_mask_448 = nodule_mask
        

    nodule_mask_448_sum = np.sum(nodule_mask_448, axis=0)
    #lung_seg_448_mean = np.mean(lung_seg_448, axis=0)
 
    if useTestPlot:    
        lung_img_448.shape
        lung_seg_448.shape
        #lung_seg_crop.shape
        nodule_mask_448.shape
            
        
        img_sel_i = 146  # 36
        
        plt.imshow(lung_img_448[img_sel_i], cmap=plt.cm.gray)
        plt.show()
        
        plt.imshow(lung_seg_448[img_sel_i], cmap=plt.cm.gray)
        plt.show()
        
        plt.imshow(nodule_mask_448[img_sel_i], cmap='gray')
        plt.show()
    
    
        for i in range (141, 153):
            print ("Slice: ", i)        
            plt.imshow(lung_seg_448[i], cmap='gray')  
            plt.show()
            #plt.imshow(nodule_mask[i], cmap='gray')
            #plt.show()
    
    useSummaryPlot = True
    if useSummaryPlot:
        mask_sum_mean_x100 = 100 * np.mean(nodule_mask_448_sum) 
        
        axis = 1
        lung_projections = []
        mask_projections = []
        for axis in range(3):
             #sxm_projection = np.max(sxm, axis = axis)
             lung_projections.append(np.mean(lung_seg_448, axis=axis))
             mask_projections.append(np.max(nodule_mask_448, axis=axis))
              
        f, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(lung_projections[0],cmap=plt.cm.gray)
        ax[1].imshow(lung_projections[1],cmap=plt.cm.gray)
        ax[2].imshow(lung_projections[2],cmap=plt.cm.gray)
        plt.show()
        f, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(mask_projections[0],cmap=plt.cm.gray)
        ax[1].imshow(mask_projections[1],cmap=plt.cm.gray)
        ax[2].imshow(mask_projections[2],cmap=plt.cm.gray)
        plt.show()
        
        print ("Mask_sum_mean_x100: ", mask_sum_mean_x100)

    # save images.    

    path = imagePath[:len(imagePath)-len(".mhd")]  # cut out the suffix to get the uid
    
    if DO_NOT_SEGMENT:
        path_segmented = path.replace("original_lungs", "lungs_2x2x2", 1)
    else:
        path_segmented = path.replace("original_lungs", "segmented_2x2x2", 1)
  
    #np.save(imageName + '_lung_img.npz', lung_img_448)
    if DO_NOT_SEGMENT:
        np.savez_compressed(path_segmented + '_lung', lung_seg_448)   
    else:
        np.savez_compressed(path_segmented + '_lung_seg', lung_seg_448)
        
    np.savez_compressed(path_segmented + '_nodule_mask', nodule_mask_448)

    return
   

def create_nodule_mask_subset(luna_subset):

    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))
    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()
   
    file = "../luna/original_lungs/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.564534197011295112247542153557.mhd"
    for file in files:
        imagePath = file
        seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
        seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid
        
        cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
        #print (cands)
        create_nodule_mask (imagePath, cands)

def create_nodule_mask_or_blank_subset(luna_subset, create_wblanks_mask_only=False):

    LUNA_DIR = LUNA_BASE_DIR % luna_subset
    files = glob.glob(''.join([LUNA_DIR,'*.mhd']))
    annotations =    pd.read_csv(LUNA_ANNOTATIONS)
    annotations.head()
    
    candidates =    pd.read_csv(LUNA_CANDIDATES)
    print ("Luna subset(s) and candidates count: ",  (luna_subset, len(candidates)))
    candidates_false = pd.DataFrame(candidates[candidates["class"] == 0])  # only select the false candidates
    candidates_true = candidates[candidates["class"] == 1]  # only select the false candidates
    print ("False & true candidates: ",  len(candidates_false), len(candidates_true))
    #candidates.head()
    
    use_all_blanks = True
    if use_all_blanks:
        aggregatez = 1  # from version unseg_blanks
    else:  # orginal aggregation
        aggregatez = int(4 * RESIZE_SPACING[0])  # originally it was 4 -- for tjhe blanks version do NOT aggregate by Z
    candidates_false["coordZ_8"] =   candidates_false["coordZ"].round(0) // aggregatez * aggregatez

   
    file = "../luna/original_lungs/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.564534197011295112247542153557.mhd"
    for file in files:
        imagePath = file
        seriesuid =  file[file.rindex('/')+1:]  # everything after the last slash
        seriesuid = seriesuid[:len(seriesuid)-len(".mhd")]  # cut out the suffix to get the uid
        
        cands = annotations[seriesuid == annotations.seriesuid]  # select the annotations for the current series
        ctrue = candidates_true[seriesuid == candidates_true.seriesuid] 
        cfalse = candidates_false[seriesuid == candidates_false.seriesuid] 
         
        if len(cands) == 0 and len(ctrue) == 0 and len(cfalse) > 0:
            if use_all_blanks:
                print ("annonations and false candidates count (using all): ", len(cands), len(cfalse))
                cfalse_sel_all = pd.DataFrame(cfalse)
                cfalse_sel_all["diameter_mm"] = 1.6 *RESIZE_SPACING[0] 
                cfalse_sel_all = cfalse_sel_all.drop(["class", "coordZ_8"], 1)
                if create_wblanks_mask_only:
                    update_nodule_mask_or_blank(imagePath, cfalse_sel_all, False)
                else:
                    create_nodule_mask_or_blank (imagePath, cfalse_sel_all, False)
                
            else:   # original version 
            
                vals, counts = np.unique(cfalse["coordZ_8"], return_counts=True)
                ## take the coordZ_8 that has most entries
                val_sel = vals[np.argmax(counts)]
                # as hook calculate the average x and y coordinates
                id_sel = cfalse["coordZ_8"] == val_sel
                xc = np.mean(cfalse[id_sel]["coordX"])
                yc = np.mean(cfalse[id_sel]["coordY"])
                zc = np.mean(cfalse[id_sel]["coordZ"])
                
                cfalse_sel = pd.DataFrame(cfalse[id_sel][:1])
                cfalse_sel["coordX"] = xc
                cfalse_sel["coordY"] = yc
                cfalse_sel["coordZ"] = zc
                cfalse_sel["diameter_mm"] = 1.6 *RESIZE_SPACING[0]  # add the diameter and drop all other columns
                cfalse_sel = cfalse_sel.drop(["class", "coordZ_8"], 1)
                
                print ("annonations and false candidates count, using average at the most frequent z: ", len(cands), len(cfalse), cfalse_sel)
                if create_wblanks_mask_only:
                    update_nodule_mask_or_blank(imagePath, cfalse_sel, False)
                else:
                    create_nodule_mask_or_blank (imagePath, cfalse_sel, False)
        
        elif len(cands) > 0:
             if create_wblanks_mask_only:
                 update_nodule_mask_or_blank(imagePath, cands, True)
             else:
                create_nodule_mask_or_blank (imagePath, cands, True)
    return

if __name__ == '__main__':
    
    
    #### STEP 1
    start = time.time()
    print ("Starting creating nodule masks ...")
    
    create_nodule_mask_or_blank_subset("[0-7]", create_wblanks_mask_only = False)
    create_nodule_mask_or_blank_subset("[8-9]", create_wblanks_mask_only = False)

    print("Total time for create_nodule_mask_or_blank_subset: ", ((time.time() - start)))  #6551 sec
   
   
    start = time.time()
    print ("Creaing wblanks masks ...")
    create_nodule_mask_or_blank_subset("[0-7]", create_wblanks_mask_only = True)  # only updates the  masks; False creates all
    print("Total time for create wblanks masks: ", ((time.time() - start)))
       
    start = time.time()
    print ("Creaing wblanks masks ...")
    create_nodule_mask_or_blank_subset("[8-9]", create_wblanks_mask_only = True)  # only updates the  masks; False creates all
    print("Total time for create wblanks masks: ", ((time.time() - start)))
 