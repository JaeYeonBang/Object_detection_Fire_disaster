from .VOCfirehouse import VOC_firehouse_dataset, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
##from .VOCfirehouse_consistency_init import  VOCDetection_con_init, VOCAnnotationTransform_con_init, VOC_CLASSES, VOC_ROOT
from .VOCfirehouse_consistency import  VOC_firehouse_dataset_con, VOCAnnotationTransform_con, VOC_CLASSES, VOC_ROOT

from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    ### changed when semi-supervised
    targets = []
    imgs = []
    semis = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        if(len(sample)==3):
            semis.append(torch.FloatTensor(sample[2]))
    if(len(sample)==2):
        return torch.stack(imgs, 0), targets
    else:
        return torch.stack(imgs, 0), targets, semis
    # return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
