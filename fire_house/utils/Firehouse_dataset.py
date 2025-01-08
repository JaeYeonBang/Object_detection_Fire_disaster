"""VOC Dataset Classes
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot
"""
import os
#from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    
import torchvision.transforms as transforms
from .smoke_augmentation import SmokeAugmentation

VOC_CLASSES = (  # always index 0
    'fire',
'smoke',
'person',
'firefighter'
)

HOME = os.path.expanduser("~")

# note: if you used our download scripts, this should be right
#DATA_ROOT = osp.join(HOME, "data/fire/")

####### 
# VOC_ROOT = osp.join(HOME, "data/fire/")
# DATA_ROOT = osp.join(HOME, "data/fire/")

# VOC_ROOT = osp.join("./", "data/")
# DATA_ROOT = osp.join("./", "data")

class VOCAnnotationTransform_con(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            #print(name)
            
            if name not in VOC_CLASSES:
                print(name)
            
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
            
        #if res == []:
        #    print('errrrrrrrr')

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Firehouse_dataset(data.Dataset):

    def __init__(self,root ,dataset_type = "VOC", transform = None , smokeAugmentation = False):
        
        #self.root = root
        
        ## HOME ROOT
        self.root = HOME

        self.dataset_type = dataset_type
        DATA_ROOT = osp.join(self.root, 'data/fire/')
        

        ##self.image_set = image_sets
        self.voc_transform = transform
        self.voc_target_transform = VOCAnnotationTransform_con()        
        self.yolo_target_transform = self.VOC_2_YOLO_AnnotationTransform_con
        
        self._annopath = osp.join(DATA_ROOT, 'annotations', '%s.xml')
    #     self.image_dir = osp.join(DATA_ROOT, 'images/')
        self.annopath_dir = osp.join(DATA_ROOT, 'annotations/')
        self.labeled_dir = osp.join(DATA_ROOT, 'labeled_img/')
        self.unlabeled_dir = osp.join(DATA_ROOT, 'unlabeled_img/')


        self.annotation_ids = os.listdir(self.annopath_dir)
        self.labeled_ids = os.listdir(self.labeled_dir)
        self.unlabeled_ids = os.listdir(self.unlabeled_dir)
        
#         self.smokeaug = SmokeAugmentation(root = self.root)      ## 
        self.smokeaug_flag = smokeAugmentation
    def __getitem__(self,index):
    
    ## img = np
        
        _,img, gt, h, w, semi = self.pull_voc_item(index , transform = self.voc_transform ,smokeaug_flag = self.smokeaug_flag )
        if(self.dataset_type == "VOC"):
    
            return torch.from_numpy(img).permute(2, 0, 1), gt, semi
        ## return -> img_path , img , target , semi 
        
        elif(self.dataset_type  == "YOLO"):

            yolo_gt = self.VOC_2_YOLO_AnnotationTransform_con(gt)
            
            img, yolo_gt = self.ToTensor((img, yolo_gt))
            
            return img,yolo_gt,semi


    def  __len__(self):
        return len(self.labeled_ids) + len(self.unlabeled_ids) 
  
    def pull_voc_item(self, index, transform , smokeaug_flag = False):
    ## labeled된지 안된지 확인
        if(index < len(self.labeled_ids)):
            labeled_flag = True
        else :
            labeled_flag = False


        if(labeled_flag is True):
            img_id = self.labeled_ids[index]
            img_path = osp.join(self.labeled_dir,img_id)
            semi = np.array([1])
            target = ET.parse(self._annopath % img_id[:-4]).getroot()
        else:
            img_id = self.unlabeled_ids[index - len(self.labeled_ids)]
            img_path = osp.join(self.unlabeled_dir,img_id)    
            semi = np.array([0])
            target = np.zeros([1,5])
        
        img_array = np.fromfile(img_path , np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        height, width, channels = img.shape
        if(labeled_flag is True):
            target = self.voc_target_transform(target, width, height)
            
        ## SmokeAug    
#         if (smokeaug_flag is True and labeled_flag is True ):
#             img , target = self.smokeaug((img,target))
            
        if transform is not None:
            target = np.array(target)
            if target.shape[0] ==0: 
                print(target.shape)
                print(img_id)
            img, boxes, labels = transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img_path , img, target, height, width, semi
    
    # return torch.from_numpy(img), target, height, width
    def VOC_2_YOLO_AnnotationTransform_con(self, targets):
        boxes = np.zeros((len(targets), 5))
        for box_idx, target in enumerate(targets):
            x1 = target[0]
            y1 = target[1]
            x2 = target[2]
            y2 = target[3]
            label = target[4]
            boxes[box_idx, 0] = label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return boxes

    def ToTensor(self,data):

        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets
