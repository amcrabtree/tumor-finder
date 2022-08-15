#!/bin/bash

# This script automates the download of 20 PCam (patch camelyon 17) WSIs from an FTP server
# FTP download site: (from http://gigadb.org/dataset/view/id/100439/File_page/1)
# Each WSI zip file takes ~1hr to download. Zipped file size is 4-15 Gb. 
# There are multiple WSIs in each zipped file, presumably multiple slides from same biopsy. 

#   Author: Angela Crabtree

# make directories
mkdir data data/wsi data/tiles output models
cd data/wsi

# download tumor annotations:
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/lesion_annotations.zip
mv lesion_annotations.zip lesion_annotations_training.zip 
unzip lesion_annotations_training.zip -d ./lesion_annotations_training

# download WSIs (these are all under Pcam's "training" category, meaning their annotations are in the lesion_annotations 
# file with "training" in its link address)
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_004.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_009.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_010.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_012.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_015.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_016.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_0/patient_017.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_020.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_021.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_022.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_024.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_034.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_036.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_038.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_1/patient_039.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_040.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_041.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_042.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_044.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_045.zip

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_046.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_048.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_051.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_2/patient_052.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_060.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_061.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_062.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_064.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_066.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_067.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_068.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_072.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_073.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_3/patient_075.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_4/patient_080.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_4/patient_081.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_4/patient_086.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_4/patient_087.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_4/patient_088.zip
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/center_4/patient_089.zip

# unzip WSI folders and delete WSIs which are not annotated
unzip patient_004.zip
rm patient_004_node_0.tif patient_004_node_1.tif patient_004_node_2.tif patient_004_node_3.tif 
unzip patient_009.zip
rm patient_009_node_0.tif patient_009_node_2.tif patient_009_node_3.tif patient_009_node_4.tif 
unzip patient_010.zip
rm patient_010_node_0.tif patient_010_node_1.tif patient_010_node_2.tif patient_010_node_3.tif 
unzip patient_012.zip
rm patient_012_node_1.tif patient_012_node_2.tif patient_012_node_3.tif patient_012_node_4.tif 
unzip patient_015.zip
rm patient_015_node_0.tif patient_015_node_2.tif patient_015_node_3.tif patient_015_node_4.tif 
unzip patient_016.zip
rm patient_016_node_0.tif patient_016_node_2.tif patient_016_node_3.tif patient_016_node_4.tif 
unzip patient_017.zip
rm patient_017_node_0.tif patient_017_node_2.tif patient_017_node_3.tif patient_017_node_4.tif 
unzip patient_020.zip
rm patient_020_node_0.tif patient_020_node_1.tif patient_020_node_3.tif patient_020_node_4.tif 
unzip patient_021.zip
rm patient_021_node_0.tif patient_021_node_1.tif patient_021_node_2.tif patient_021_node_4.tif 
unzip patient_022.zip
rm patient_022_node_0.tif patient_022_node_1.tif patient_022_node_2.tif patient_022_node_3.tif 
unzip patient_024.zip
rm patient_024_node_0.tif patient_024_node_2.tif patient_024_node_3.tif patient_024_node_4.tif 
unzip patient_034.zip
rm patient_034_node_0.tif patient_034_node_1.tif patient_034_node_2.tif patient_034_node_4.tif 
unzip patient_036.zip
rm patient_036_node_0.tif patient_036_node_1.tif patient_036_node_2.tif patient_036_node_4.tif 
unzip patient_038.zip
rm patient_038_node_0.tif patient_038_node_1.tif patient_038_node_3.tif patient_038_node_4.tif 
unzip patient_039.zip
rm patient_039_node_0.tif patient_039_node_2.tif patient_039_node_3.tif patient_039_node_4.tif 
unzip patient_040.zip
rm patient_040_node_0.tif patient_040_node_1.tif patient_040_node_3.tif patient_040_node_4.tif 
unzip patient_041.zip
rm patient_041_node_1.tif patient_041_node_2.tif patient_041_node_3.tif patient_041_node_4.tif 
unzip patient_042.zip
rm patient_042_node_0.tif patient_042_node_1.tif patient_042_node_2.tif patient_042_node_4.tif 
unzip patient_044.zip
rm patient_044_node_0.tif patient_044_node_1.tif patient_044_node_2.tif patient_044_node_3.tif 
unzip patient_045.zip
rm patient_045_node_0.tif patient_045_node_2.tif patient_045_node_3.tif patient_045_node_4.tif 

unzip patient_046.zip
rm patient_046_node_0.tif patient_046_node_1.tif patient_046_node_2.tif patient_046_node_4.tif 
unzip patient_048.zip
rm patient_048_node_0.tif patient_048_node_2.tif patient_048_node_3.tif patient_048_node_4.tif 
unzip patient_051.zip
rm patient_051_node_0.tif patient_051_node_1.tif patient_051_node_3.tif patient_051_node_4.tif 
unzip patient_052.zip
rm patient_052_node_0.tif patient_052_node_2.tif patient_052_node_3.tif patient_052_node_4.tif 
unzip patient_060.zip
rm patient_060_node_0.tif patient_060_node_1.tif patient_060_node_2.tif patient_060_node_4.tif 
unzip patient_061.zip
rm patient_061_node_0.tif patient_061_node_1.tif patient_061_node_2.tif patient_061_node_3.tif 
unzip patient_062.zip
rm patient_062_node_0.tif patient_062_node_1.tif patient_062_node_3.tif patient_062_node_4.tif 
unzip patient_064.zip
rm patient_064_node_1.tif patient_064_node_2.tif patient_064_node_3.tif patient_064_node_4.tif 
unzip patient_066.zip
rm patient_066_node_0.tif patient_066_node_1.tif patient_066_node_3.tif patient_066_node_4.tif 
unzip patient_067.zip
rm patient_067_node_0.tif patient_067_node_1.tif patient_067_node_2.tif patient_067_node_3.tif 
unzip patient_068.zip
rm patient_068_node_0.tif patient_068_node_2.tif patient_068_node_3.tif patient_068_node_4.tif 
unzip patient_072.zip
rm patient_072_node_1.tif patient_072_node_2.tif patient_072_node_3.tif patient_072_node_4.tif 
unzip patient_073.zip
rm patient_073_node_0.tif patient_073_node_2.tif patient_073_node_3.tif patient_073_node_4.tif 
unzip patient_075.zip
rm patient_075_node_0.tif patient_075_node_1.tif patient_075_node_2.tif patient_075_node_3.tif 
unzip patient_080.zip
rm patient_080_node_0.tif patient_080_node_2.tif patient_080_node_3.tif patient_080_node_4.tif 
unzip patient_081.zip
rm patient_081_node_0.tif patient_081_node_1.tif patient_081_node_2.tif patient_081_node_3.tif 
unzip patient_086.zip
rm patient_086_node_1.tif patient_086_node_2.tif patient_086_node_3.tif patient_086_node_4.tif 
unzip patient_087.zip
rm patient_087_node_1.tif patient_087_node_2.tif patient_087_node_3.tif patient_087_node_4.tif 
unzip patient_088.zip
rm patient_088_node_0.tif patient_088_node_2.tif patient_088_node_3.tif patient_088_node_4.tif 
unzip patient_089.zip
rm patient_089_node_0.tif patient_089_node_1.tif patient_089_node_2.tif patient_089_node_4.tif 

# move corresponding xml files into folder with WSIs
for file in *node*.tif; do cp lesion_annotations_training/$(basename $file .tif).xml .; done

# navigate back to main folder
cd ../..
