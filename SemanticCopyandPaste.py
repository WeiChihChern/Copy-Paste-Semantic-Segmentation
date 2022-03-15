import albumentations as A
import random
import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
import glob
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


class SemanticCopyandPaste(A.DualTransform):
    
    def __init__(self, 
                 nClass, 
                 path2rgb, 
                 path2mask, 
                 shift_x_limit = [0,0], 
                 shift_y_limit = [0,0], 
                 rotate_limit  = [0,0],
                 scale         = [0,0],
                 class_weights = [],
                 always_apply  = False,
                 show_stats    = False,
                 auto_weights  = False,
                 p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.nClass     = nClass
        self.rgb_base   = path2rgb
        self.mask_base  = path2mask
        #################
        self.rgbs       = glob.glob(os.path.join(path2rgb, "*.*"))
        self.rgbs       = [x for x in self.rgbs if x.split('.')[-1].lower() in IMG_FORMATS]
        self.masks      = glob.glob(os.path.join(path2mask, "*.*"))
        self.masks      = [x for x in self.masks if x.split('.')[-1].lower() in IMG_FORMATS]
        #################
        self.nImages    = len(self.rgbs)
        self.threshold  = 0
        self.targetClass= 0
        self.c_image    = None # candidate image
        self.c_mask     = None # candidate mask
        self.found      = False
        self.imgRow     = None  # for image translation
        self.imgCol     = None  # for image translation

        ## Parameters for augmentation
        self.shift_x_limit = shift_x_limit
        self.shift_y_limit = shift_y_limit
        self.rotate_limit  = rotate_limit
        self.scale         = scale

        ## private variables
        self._transformation_matrix = None
        self._translated_mask       = None

        
        # Augment probability to each class
        self.class_weights   = [abs(ele) for ele in class_weights]
        self.class_pool      = []
        self.img_pool        = np.zeros((self.nClass, len(self.masks))) - 1 # Use -1 as the flag of empty
        self.class_pixels_statistics = np.zeros((self.nClass,1), dtype=np.float64)
        self.auto_weights    = auto_weights
        
        



        # Params checking
        assert len(self.rgbs) == len(self.masks), "rgb path's file count != mask path's file count"
        assert self.nClass     > 0,               "Incorrect class number"
        
        if shift_x_limit is not None:
            assert type(shift_x_limit) == list and type(shift_y_limit) == list and type(rotate_limit) == list and type(scale) == list      
            assert abs(shift_x_limit[0]) <= 1 and abs(shift_y_limit[0]) <= 1 and abs(rotate_limit[0]) <= 1 and abs(rotate_limit[1]) <= 1 and scale[0] >= 0 and scale[1] >= scale[0] and scale[1] >= 1, 'The range for shift_x/y_limit and rotate is [-1 to 1], and [0 to 1] for scale'



        # Image pool initialization for fast image lookup
        # Go through all masks, and find out what class(es) each mask has
        # [
        #  [mask#1, mask#2, etc] <-- for class idx 0
        #  [mask#4, mask#99, etc] <-- for class idx 1
        # ]
        class_count_tmp = np.zeros((self.nClass, 1), dtype=np.int)
        for i, mName in enumerate(self.masks):
            c_mask = cv2.imread(mName,0)
            for j in range(self.nClass):
                if self.target_class_in_image(c_mask, j): # If class pixel > sefl.threshold
                    self.img_pool[j, class_count_tmp[j, 0]] = i 
                    class_count_tmp[j, 0] += 1


        
        # For class augmentation probability control
        if self.auto_weights:
            print('- Copy and Paste: Auto weights calculation used -')
            tmp = np.copy(self.class_pixels_statistics)
            tmp = 1 / tmp
            tmp[0,0] = 0
            self.class_weights = np.round(tmp / np.sum(tmp) * 100) # Normalized
            for i in range(nClass):
                for j in range(np.int(self.class_weights[i])):
                    self.class_pool.append(i)
        else:
            if not class_weights:
                print('- Copy and Paste: Using equal weights for all classes (background not included) -')
                for i in range(1,self.nClass): self.class_pool.append(i)
            else:
                print('- Copy and Paste: Using user defined class weights -')
                self.class_weights = np.round(self.class_weights / np.sum(self.class_weights) * 100) # Normalized
                assert len(class_weights) == nClass, "class_weights' length != nClass, nClass should also include the background class."

                for i in range(nClass):
                    for j in range(np.int(self.class_weights[i])):
                        self.class_pool.append(i)
            
        if show_stats: print('Pixel Count for Each Class: \n', self.class_pixels_statistics)
    
    
    
    
    def apply(self, image, **params):
        '''
            Args:
                image: 3-channel RGB images

           This function will first randomly generate a class that being copied (Exclude 0, which is the background class). Then randomly picks a mask via provided path, and search whether it contains the previously picked target class. Keep randomly picks a new mask until a match is found. Finally start doing copy and paste process.

           Since semantic segmentation's annotation may not be labeled in the same way as instance segmentation therefore currently we copy and paste entire mask without further processing.
        '''

        self.targetClass = random.choice(self.class_pool)
        
        # Finding candidates with the target class
        ret = -1
        while ret == -1:
            ret = int(random.choice(self.img_pool[self.targetClass, :]))
        c_image   = cv2.imread(self.rgbs[ret])
        c_mask    = cv2.imread(self.masks[ret], 0)
        c_image   = cv2.cvtColor(c_image, cv2.COLOR_BGR2RGB)
        self.found       = True
        self.c_mask      = c_mask
        self.c_image     = c_image

                
        return self.copy_and_paste_image(self.c_image, self.c_mask, image, self.targetClass)
    

    
    def apply_to_mask(self, mask, **params):
        assert self.found == True
        return self.copy_and_paste_mask(self.c_mask, mask, self.targetClass)
    

    
    
    # Augmentation will be added to rgb2 (extract content from rgb1)
    # Mask1 is need to know where to extract pixels for color image copy and paste
    def copy_and_paste_image(self, rgb1, mask1, rgb2, targetClassForAug):
        assert rgb1  is not None
        assert rgb2  is not None
        assert mask1 is not None
        # assert mask1.shape[2] == 3 # We imread it without further process, so its a 3 channel
        
        if rgb2.shape != rgb1.shape:
            r, c, _ = rgb2.shape
            rgb1  = cv2.resize(rgb1, (c,r), interpolation = cv2.INTER_NEAREST)
            mask1 = cv2.resize(mask1, (c,r), interpolation = cv2.INTER_NEAREST)
          
        # tmp   = mask1[...,1] # All 3 channels have same content, we take 1 to process
        masks = [(mask1 == v) for v in range(self.nClass)] 
        masks = np.stack(masks, axis=-1).astype('float') # mask.shape = (x,y,ClassNums)
        self.c_mask = masks

        
        masks[..., targetClassForAug] = \
            self.imgTransform(masks[..., targetClassForAug], self.shift_x_limit, self.shift_y_limit)

        self._translated_mask = masks[..., targetClassForAug]

        rgb1 = cv2.warpAffine(rgb1, self._transformation_matrix, (self.imgCol, self.imgRow))
        

        # Pasting
        mask_3channel = np.stack((self._translated_mask,self._translated_mask,self._translated_mask),axis=2)
        idxs = mask_3channel > 0
        rgb2[idxs] = rgb1[idxs]
        
        
        return rgb2.astype('uint8')

    
    
    
    
    
    
    
    def copy_and_paste_mask(self, mask1, mask2, targetClassForAug):
        '''
            Args:
                mask1 = randomly picked qualified mask from apply(), has shape = (x, y, nClasses)
                mask2 = dataloader loaded mask, aug is added to mask2
        '''
        # assert mask2.shape[2] == self.nClass # Processed by dataloader, so its a nClass channel
        assert self._translated_mask is not None
        
        mask2_1channel = np.argmax(mask2, axis=2)
        
        # Pasting augmentation
        mask2_1channel[self._translated_mask > 0] = targetClassForAug
        
        masks   = [(mask2_1channel == v) for v in range(self.nClass)] # mask.shape = (x,y,ClassNums)
        masks   = np.stack(masks, axis=-1).astype('float')
        
        # Reset
        self.c_mask = None 
        self.found == False
        self._transformation_matrix = None
        self._translated_mask = None
        return masks
    
    
    
    
    
    
    # We imread the mask, so it's a 3-channel mask (not one-hot encoded)
    def target_class_in_image(self, mask, targetClassIdx):
        
        #hard coded pixel threshold
        s = np.sum(mask == targetClassIdx)
        self.class_pixels_statistics[targetClassIdx, 0] += s # record total pixel for each class
        if s > self.threshold: 
            return True
        
        return False
    
    
    
    def imgTransform(self, image, offset_x_limit, offset_y_limit ):
        '''
            Args:
                image: it can be mask or rgb image
                offset_x_limt: x-axis shift limit [-1,1]
                offset_y_limt: y-axis shift limit [-1,1]
        '''
        self.imgRow, self.imgCol = image.shape

        col_shift  = random.uniform(offset_x_limit[0], offset_x_limit[1])*self.imgCol
        row_shift  = random.uniform(offset_y_limit[0], offset_y_limit[1])*self.imgRow
        rotate_deg = random.uniform(self.rotate_limit[0], self.rotate_limit[1])*180
        scale_coef = random.uniform(self.scale[0]       , self.scale[1])
        
        self._transformation_matrix = cv2.getRotationMatrix2D((self.imgRow//2, self.imgCol//2), rotate_deg, scale_coef)
        self._transformation_matrix[0,2] += col_shift
        self._transformation_matrix[1,2] += row_shift
        
        return cv2.warpAffine(image, self._transformation_matrix, (self.imgCol, self.imgRow))


    
    
    
    
    
    
    

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("image", "mask")
    

        
    
