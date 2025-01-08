## https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/03_01_dataloader.html
from  matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import os.path as osp
import sys
import torch
import torchvision
import torch.utils.data as data
import cv2

from .Format import YOLO as cvtYOLO
from .Format import VOC as cvtVOC
from .smoke_augmentation import SmokeAugmentation

from .transform import DEFAULT_TRANSFORMS


ImageFile.LOAD_TRUNCATED_IMAGES = True
SMOKE_ID = 1.0


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, imga

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, root = './', train=True, transform=None, resize=448, class_path='./utils/fire.name.txt'):

        self.root = root
        DATA_ROOT = osp.join(root, 'data/fire/')
        self.transform = transform
        ##self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.class_path = class_path


        with open(class_path) as f:
            self.classes = f.read().splitlines()

        self.image_dir = osp.join(DATA_ROOT, 'images/')
        print(self.image_dir)
        self.annopath_dir = osp.join(DATA_ROOT, 'annotations/')
        self.image_ids = [ _ for _ in os.listdir(self.image_dir) if _.endswith('.jpg')]
        self.annotation_ids = os.listdir(self.annopath_dir)
        self.batch_count = 0
        
        self.dict_flag , self.dict_data, self.error_list =  self.cvtDictData()
        
        if self.dict_flag is True :
            self.data = self.cvtDictData2Yolo(self.dict_data)
        else :
            print("problem in parsing")
        self.yolo = cvtYOLO(os.path.abspath(self.class_path))
        #self.smokeAugmentation = SmokeAugmentation(root = DATA_ROOT)
        #self.smoke_dir = self.smokeAugmentation.smoke_dir
        #self.smoke_ids = os.listdir(self.smoke_dir)
        
    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = osp.join(self.image_dir,img_id)
        
        #### 한글파일 깨짐현상 해결 위한 코드
        img_array = np.fromfile(img_path , np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        ##img = cv2.imread(img_path)
        ##img = Image.open(img_path).convert('RGB')
        
        
        #if img is None:
        #    print("nonetype",img_path)
        ### label 가져오기     
        
        if self.dict_flag is not True :
            return -1;
        
        try:
            key = img_id[0:-4]
            target = self.data[key]
            semi = np.array([1])
            #target , img , factor = self.smoke_aug_function(target , img , 0)
            target = np.array(target)

        except:
            semi = np.array([0])
            target = np.zeros((1,5))
  
        #print(img_path)
        h , w , c = img.shape
        current_size = (h , w)
#         img = cv2.resize(img , (self.resize_factor, self.resize_factor))
#         img = torchvision.transforms.ToTensor()(img)
        # -----------
        #  Transform
        # -----------
        if self.transform is not None:
            img, target = self.transform((img, target))
          
        #print('a',img)

        return  img_path, img, target, semi, current_size 

    def __len__(self):
        return (len(self.image_ids))
    
    def cvtDictData(self):
        voc = cvtVOC()
        dict_flag, dict_data , error_list =voc.parse(os.path.join(self.annopath_dir))
        return dict_flag , dict_data ,error_list
    
    def cvtDictData2Yolo(self, dict_data):

        result = {}
      
        yolo = cvtYOLO(os.path.abspath(self.class_path))
      
        
        try:
            if self.dict_flag:
                flag, data =yolo.generate(dict_data)

            keys = list(data.keys())
#           keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

            for key in keys:
                contents = list(filter(None, data[key].split("\n")))
                target = []
                for i in range(len(contents)):
                    tmp = contents[i]
                    tmp = tmp.split(" ")
                    for j in range(len(tmp)):
                        tmp[j] = float(tmp[j])
                    target.append(tmp)
                result[key] = target

            return result

        except Exception as e:
            raise RuntimeError("Error : {}".format(e))
            
    def cvtData_By_Data (self, key):
        data_target = self.dict_data[key]
        new_target = {}
        new_target[key] = data_target
        flag, data =self.yolo.generate(new_target)
        if flag:
            contents = list(filter(None, data[key].split("\n")))
            result = []
            target = []
            for i in range(len(contents)):
                tmp = contents[i]
                tmp = tmp.split(" ")
                for j in range(len(tmp)):
                    tmp[j] = float(tmp[j])
                target.append(tmp)
            
        return target
    
    def search_smoke(self,target):
        for i in target:
            if i[0] == 1.0:
                return True
        return False
    
    def smoke_aug_function(self, target , dest_image , threshold):
        if self.search_smoke(target) is not True:
            x = random.randint(1,100)
            if x > threshold:
                
                smoke_index = random.randint(0,len(self.smokeAugmentation.smoke_ids)-1)
                smoke_image = cv2.imread(osp.join(self.smokeAugmentation.smoke_dir, self.smokeAugmentation.smoke_ids[smoke_index]))
                new_img , ((min_x , max_x),(min_y , max_y)) , factor = self.smokeAugmentation.patch_ex(ima_dest = dest_image , ima_src = smoke_image)
                ##print((min_x , max_x),(min_y , max_y))
                new_label = self.smokeaug_To_Yolo((new_img.shape[0],new_img.shape[1]) , min_x , max_x, min_y , max_y , factor)
                target.append(new_label)
                return target , new_img , factor

#                 try: 
#                     smoke_index = random.randint(0,len(smokeAugmentation.smoke_ids)-1)
#                     smoke_image = cv2.imread(osp.join(smokeAugmentation.smoke_dir, smokeAugmentation.smoke_ids[smoke_index]))
#                     new_img , ((min_x , max_x),(min_y , max_y)) , factor = smokeAugmentation.patch_ex(ima_dest = dest_image , ima_src = smoke_image)
#                     new = smokeaug_To_Yolo((img.shape[0],img.shape[1]) , min_x , max_x, min_y , max_y , factor)
#                     target.append(new)
#                     return target , new_img , factor
#                 except:
#                     return target , dest_image , -1
        return target , dest_image , -1
    def smokeaug_To_Yolo(self, size ,  min_x , max_x, min_y , max_y , factor):
        b = (float(min_x), float(max_x), float(min_y), float(max_y))
        bb = self.yolo.coordinateCvt2YOLO(size, (min_x , max_x, min_y , max_y))
        new_label = []
        new_label.append(1.0)
        new_label.append(bb[0])
        new_label.append(bb[1])
        new_label.append(bb[2])
        new_label.append(bb[3])
        return new_label
    def collate_fn(self,batch):
        #batch_count += 1

    # Drop invalid images
       # batch = [data for data in batch if data is not None]
        
        #print('batch_list',len(batch))
            
        paths, imgs, bb_targets, semis, current_size = list(zip(*batch))
        

    #     Selects new image size every tenth batch
    #     if self.multiscale and self.batch_count % 10 == 0:
    #         self.img_size = random.choice(
    #             range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.resize_factor) for img in imgs])
        
        #print('imgsize',imgs.size())

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        #return paths, imgs, bb_targets, semis
        return imgs, bb_targets, semis




if __name__ == '__main__':

    HOME = os.path.expanduser("~")

    dataset = ListDataset(root = HOME, transform = DEFAULT_TRANSFORMS) 


    print(dataset.__getitem__(0))


