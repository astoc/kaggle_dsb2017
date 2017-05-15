Simple instructions to create the DSB17 submission. 
Note that the problem and solution are very data intensive, and it may take several days to build the entire solution.

This part describes how to create 3 feature sets to be used by the code under the Shize directory to produce the team's best submission.

The csv file names for these three feature sets are (stage1 and stage2): 
1) 1st set: feats_keras8_0313_stage1_1595_0411.csv, feats_keras8_0313_stage2_506_0411.csv; 
2) 2nd set: feats_keras_0311_stage1_1595_0411.csv,  feats_keras_0311_stage2_506_0411.csv; 
3) 3rd set: feats_8_stage1_1595_0411.csv,           feats_8_stage2_506_0411.csv


0. Create a sub-directory structure for the grid 2x2x2
======================================================

NOTE: recreating the Luna data structure is not necessary if option 1 or 2 is chosen in Part 1.

mkdir luna luna/data luna/data/lungs_2x2x2 luna/original_lung_masks 
for i in 0 1 2 3 4 5 6 7 8 9; do mkdir luna/data/lungs_2x2x2/subset$i; done
cd luna/data/original_lungs

# after downloading the Luna dataset from https://luna16.grand-challenge.org/home/ (10 subsets, and the CSVFILES), do:

7z x  ../../../download/luna/CSVFILES.zip
for i in 0 1 2 3 4 5 6 7 8 9;  do 7z x  ../../../download/luna/subset$i.zip; done
cd ../../..
mkdir input
cd input
mkdir stage2_2x2x2 stage2_segmented_2x2x2 stage2_segmented_2x2x2_crop 
mkdir stage2_2x2x2_mpred3_8

# after downloading the competition data from Kaggle do:
7z x ../download/stage1_labels.csv.zip
7z x ../download/stage1_sample_submission.csv.zip
7z x ../download/stage2_sample_submission.csv.zip

### NOTE after the following lines a decryption password for the relevant stage data must be entered
7z x ../download/stage1.7z 
7z x ../download/stage2.7z 

### NOTE: the spyder program is used to run some parts of the model to provide visual validation of intermediate files created. Thus it can also be run from within the AWS by starting the X server during the initial connection (i.e. add "-X" while connecting to the AWS using the ssh program for instance).


1. PART 1
=========
PART 1: Reuse or recreate a nodule identifier. 
        Use any of the following options (arranged from a simple & fast to complex & slow)

  Option 1: Copy the nodule identifier weights created in this competition  to ../luna/models  (time: couple of minutes)
    mkdir luna luna/models
    cp -p ./nodule_identifiers/d8_2x2x2_best_weights.h5 luna/models

  Option 2: Retrain the nodule identifier from the training and validation data created from the Luna database, starting from the model used for the final submissions (time: hours)
    mkdir luna luna/models
    cp -p ./nodule_identifiers/d8_2x2x2_best_weights.h5 luna/models  
    If the intermediary model_data has been provided (circa 1.1 GB), do:
        cp -p ./model_data/masks*.npz ../model_data/scans*.npz luna/models  
    If the intermediary data has not been provided (as is the case for the GitHub repository, given the large size of the data), allocate an additional day or so of computing processing; follow partially the Option 3 below, till you are about to run the unet_d8g_222f.py program at the end of the Sub-step 2. Then, execute the unet_d8g_222f.py as follows (after setting the recreate_grid8_March_data to False at the beginning of the main routine). 
        spyder --new-instance -w $PWD unet_d8g_222f.py &       # run Part 1, Option 2 step (optional first part of the main procedure); it creates relevant scans* and masks* files within ../luna/models

    Subsequently, retrain/refit the NN nodule identifier by running the following scripts: 
        refit_unet_d8g_222_swrap_01.py, fit*_02.py, fit*_03.py, ...., fit*_10.py, fit*_11.py, in that sequence (i.e. run python refit_unet_d8g_222_swrap_01.py etc.)

    As the new (refitted) nodule generator use the last model or weights created, namely: 
    ../luna/models/d8g_bre_model_50.h5  and ../luna/models/d8g_bre_weights_50.h5

    
  Option 3: Using publically available LUNA dataset create the nodule identifier from scratch, starting from obtaining and preprocessing the data, creating training and validation data, and training the nodule identifier (time: couple of days)
    Sub-step 1: Obtain & install the lungs LUNA dataset (may take couple of hours)

      mkdir luna luna/data luna/data/lungs_2x2x2 luna/data/original_lungs luna/models
      for i in 0 1 2 3 4 5 6 7 8 9; do mkdir luna/data/lungs_2x2x2/subset$i; done

      # download Luna dataset from https://luna16.grand-challenge.org/home/ (10 subsets, and the CSVFILES)
      # Extract  extract the Luna 10 subsets and CSVFILES to luna/original_lung_masks (modify the download path below as needed)
      cd luna/data/original_lungs
      7z x  ../../../download/luna/CSVFILES.zip
      for i in 0 1 2 3 4 5 6 7 8 9;  do 7z x  ../../../download/luna/subset$i.zip; done
      # You may wish to validate the output of the 7z program and/or the size and count of files extracted (the space should be circa 112GB, count of directories 11, and count of files 1813 as shown below)
      du -h .       # 112 GB
      ls  | wc -l   # 11
      ls -R | wc -l # 1813

      cd ../../..
    
    Sub-step 2: Create data files for training nodule identifier on a slicing architecture used shown in the solution description
      ## run the following iseg_luna3_lub_222f.py to create data files for training a nodule identifier on a slicing architecture (in the luna/data/lungs_2x2x2)
      ## (the program creates *_lung.npz and *_nodule_maks.npz files in the luna/data/lungs_2x2x2/subset? directories)
      ## as the program displays projections of the 3D lung and masks data, run the program within spyder or similar
      cd py
      spyder --new-instance -w $PWD iseg_luna3_lub_222f.py & # press F5 to run within spyder (time circa 2 hours)
      spyder --new-instance -w $PWD unet_d8g_222f.py &       # run Part 1, Option 3 step (first part of the main procedure); it creates scans* and masks* files within ../luna/models

    Sub-step 3: Train/fit NN nodule identifier on the prepared data
    For the detailed training see the script: 
    #run the following scripts: fit_unet_d8g_222_swrap_01.py, fit*_02.py, fit*_03.py, ...., fit*_10.py, fit*_11.py, in that sequence:
    python fit_unet_d8g_222_swrap_01.py
    ...
    python fit_unet_d8g_222_swrap_06.py
    ...
    python fit_unet_d8g_222_swrap_11.py

    As the new nodule generator use later the last model or weights created (*_74.h5), namely: 
    ../luna/models/d8g4a_model_74.h5 and ../luna/models/d8g4a_weights_74.h5

    (for some validation or comparisons you may wish to use also model d8g4a_model_69, created after running the script 06)

PART 2: - prepare the competition data
=================================
  Step 1: Create 2x2x2 resolution grid, and segment lungs data (stage1 and stage2)
   cd ../input
   ## create an empty stage2_labels.csv to resemble stage1 setup, as follows:
   head -1 stage1_labels.csv > stage2_labels.csv
   cd py
   spyder --new-instance -w $PWD unet_d8g_222f.py &    # run Part 2 of the main procedure (segment all)
   # if you don't want or need progress images per patient you may define showSummary as False

PART 3 - identify the nodules and create feature sets for the final/stage 2 calculations
========================================================================================
Identify the nodules masks using the Luna trained nodule identifier and save the graphical masks, and calculate the feature files
   spyder --new-instance -w $PWD lungs_var3_d8g_222f.py &
  
#Note: identifying the nodules is time consuming (a day or so). Thus, once identified you may wish to set the make_predictions variable as False (in the main function). Any further calculation of features would use the precalculated nodule masks. 



Dependency notes
================
1. Key data dependencies
------------------------
Data dependency: the Luna dataset available at: https://luna16.grand-challenge.org/home/ (10 subsets, and the CSVFILES), and the Kaggle competition data for stage 1 and stage2.

2. Key software dependencies (all open software)
------------------------------------------------
Keras 1.2.2, Theano, Python3, conda (recommended), spyder, cv2/opencv, pydicom, scipy, scikit-image, simpleitk, numpy, and pandas. Linux environment preferred/assumed.

