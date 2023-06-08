import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[])


    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        list_of_calcification = []
        list_of_internal_structure = []
        list_of_sphericity = []
        list_of_margin = []
        list_of_lobulation = []
        list_of_spiculation = []
        list_of_texture = []
        list_of_subtlety = []

        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)
            list_of_calcification.append(annotation.calcification)
            list_of_internal_structure.append(annotation.internalStructure)
            list_of_sphericity.append(annotation.sphericity)
            list_of_margin.append(annotation.margin)
            list_of_lobulation.append(annotation.lobulation)
            list_of_spiculation.append(annotation.spiculation)
            list_of_texture.append(annotation.texture)
            list_of_subtlety.append(annotation.subtlety)

        malignancy = median_high(list_of_malignancy)
        calcification = median_high(list_of_calcification)
        internal_structure = median_high(list_of_internal_structure)
        sphericity = median_high(list_of_sphericity)
        margin = median_high(list_of_margin)
        lobulation = median_high(list_of_lobulation)
        spiculation = median_high(list_of_spiculation)
        texture = median_high(list_of_texture)
        subtlety = median_high(list_of_subtlety)

        return malignancy, calcification, internal_structure, sphericity, margin, lobulation, spiculation, texture, subtlety

    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['image_dir', 'malignancy', 'calcification', 'internal_structure', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'subtlety'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    max_width = 0
    max_height = 0
    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        # count = 10
        for patient in tqdm(self.IDRI_list):
            # count-=1
            # if count == 0:
            #     break
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

            if scan is None:
                print(f"No scan found for patient ID: {pid}")
                continue

            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    lung_np_array = vol[cbbox]

                    # We calculate the malignancy information
                    malignancy, calcification, internal_structure, sphericity, margin, lobulation, spiculation, texture, subtlety = self.calculate_malignancy(nodule)

                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        # Segment Lung part only
                        # lung_segmented_np_array = segment_lung(lung_np_array[:,:,nodule_slice])
                        lung_np_array_2d = lung_np_array[:,:,nodule_slice]
                        # I am not sure why but some values are stored as -0. <- this may result in datatype error in pytorch training # Not sure
                        # lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        
                        img_dir = "./"+str(patient_image_dir) + "/"+ nodule_name + ".png"
                        meta_list = [img_dir, malignancy, calcification, internal_structure, sphericity, margin, lobulation, spiculation, texture, subtlety]

                        self.save_meta(meta_list)
                        # np.save(patient_image_dir / nodule_name,lung_segmented_np_array)

                        #apply min max normalization
                        scaler = MinMaxScaler(feature_range=(0, 255))
                        lung_np_array_2d = scaler.fit_transform(lung_np_array_2d)
                        img = Image.fromarray(lung_np_array_2d.astype(np.uint8))
                        
                        if img.width > self.max_width:
                            self.max_width = img.width
                            print(self.max_width)
                            print(self.max_height)
                        
                        if img.height > self.max_height:
                            self.max_height = img.height
                            print(self.max_width)
                            print(self.max_height)
                        
                        
                        img.save(patient_image_dir / f"{nodule_name}.png")
                        

                        # np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
            else:
                print("Clean entry ignored")



        print("Saved Meta data")
        print(self.meta.head())
        print(self.meta_path)
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)
        print("Max Width: ",self.max_width)
        print("Max Height: ",self.max_height)


if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()


    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()
