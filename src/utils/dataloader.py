import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from .misc import list_img

class Dataset(BaseDataset):
# return 4 inputs, one segmentation mask along with segmentration masks  
    def __init__(
            self, 
            t1_list, 
            t1ce_list,
            t2_list, 
            flair_list,
            seg_list,
            augmentation=None, 
            preprocessing=True,
    ):
        self.t1_list = t1_list
        self.t1ce_list= t1ce_list
        self.t2_list = t2_list
        self.flair_list = flair_list
        self.seg_list = seg_list
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def standardize(self, x, mean, std):
        """Standardize the given image.
        Args:
            x (np.ndarray): Image to standardize.
        Returns:
            np.ndarray: Standardized image.
        """ 
        x = x.reshape(1,240,240)
        x = x/255
        x -= mean
        x /= std
        return x.astype(np.float32)
    
    def __getitem__(self, i):
        
        # read data
        t1image = np.array(cv2.imread(self.t1_list[i],cv2.IMREAD_GRAYSCALE))
        t1ceimage = np.array(cv2.imread(self.t1ce_list[i],cv2.IMREAD_GRAYSCALE))
        t2image = np.array(cv2.imread(self.t2_list[i],cv2.IMREAD_GRAYSCALE))
        flairimage = np.array(cv2.imread(self.flair_list[i],cv2.IMREAD_GRAYSCALE))
        seg = np.load(self.seg_list[i])
        seg = seg.reshape(240,240,1).astype(np.float32)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=t1image,image1=t1ceimage,image2=t2image,image3=flairimage,mask=seg)
            t1image, seg, t1ceimage, t2image, flairimage= sample['image'], sample['mask'], sample['image1'],sample['image2'],sample['image3']
            seg = seg.reshape(240,240)
        # seg = seg.reshape(4,240,240)
#         timage = timage.reshape(480,640,1)
        if self.preprocessing:
            t1image = self.standardize(t1image, 0.0999, 0.23646)
            t1ceimage = self.standardize(t1ceimage, 0.05345, 0.13268)
            t2image = self.standardize(t2image, 0.0567, 0.14415)
            flairimage = self.standardize(flairimage, 0.0725, 0.1804)
            seg = seg.astype(np.int64)
            
        return t1image,t1ceimage,t2image,flairimage,seg 
        
    def __len__(self):
        return len(self.t1_list)#,len(self.t1ce_list),len(self.t2_list),len(self.flair_list)
        

class Dataset_Infer(BaseDataset):
# return 4 inputs, one segmentation mask along with segmentration masks  
    def __init__(
            self, 
            t1_list, 
            t1ce_list,
            t2_list, 
            flair_list,
            preprocessing=True,
    ):
        self.t1_list = t1_list
        self.t1ce_list= t1ce_list
        self.t2_list = t2_list
        self.flair_list = flair_list

        self.preprocessing = preprocessing

    def standardize(self, x, mean, std):
        """Standardize the given image.
        Args:
            x (np.ndarray): Image to standardize.
        Returns:
            np.ndarray: Standardized image.
        """ 
        x = x.reshape(1,240,240)
        x = x/255
        x -= mean
        x /= std
        return x.astype(np.float32)
    
    def __getitem__(self, i):
        
        # read data
        t1image = np.array(cv2.imread(self.t1_list[i],cv2.IMREAD_GRAYSCALE))
        t2image = np.array(cv2.imread(self.t2_list[i],cv2.IMREAD_GRAYSCALE))
        flairimage = np.array(cv2.imread(self.flair_list[i],cv2.IMREAD_GRAYSCALE))
        t1ceimage = np.array(cv2.imread(self.t1ce_list[i],cv2.IMREAD_GRAYSCALE))
        if self.preprocessing:
            t1image = self.standardize(t1image, 0.0999, 0.23646)
            t1ceimage = self.standardize(t1ceimage, 0.05345, 0.13268)
            t2image = self.standardize(t2image, 0.0567, 0.14415)
            flairimage = self.standardize(flairimage, 0.0725, 0.1804)
            
        return t1image,t1ceimage,t2image,flairimage,np.array([i])
        
    def __len__(self):
        return len(self.t1_list)#,len(self.t1ce_list),len(self.t2_list),len(self.flair_list)