import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval_ffa_minus
import os
from net.models import *

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

gps=3
blocks=19
dataset = 'its'
ffa_model_dir= f'net/trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet=torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)
    
    ####FFA
    ckp=torch.load(ffa_model_dir,map_location='cuda')
    ffa_model = FFA(gps=gps,blocks=blocks)
    ffa_model= torch.nn.DataParallel(ffa_model).cuda()#,device_ids = [0,1])
    ffa_model.load_state_dict(ckp['model'])
    ffa_model.eval()

    
    retinanet.training = False
    retinanet.eval()
    ## if final
    retinanet.module.module.freeze_bn()

    print(csv_eval_ffa_minus.evaluate(dataset_val, retinanet,ffa_model,iou_threshold=float(parser.iou_threshold)))



if __name__ == '__main__':
    main()
