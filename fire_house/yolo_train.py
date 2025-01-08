import math

import torch
import torch.nn as nn

from yolov3 import load_model
import argparse
import os
import torch
from data import *
from utils.augmentations import SSDAugmentation
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils.loss import compute_loss
import torch.utils.data
from tqdm import tqdm
from layers.modules import MultiBoxLoss, CSDLoss, ISDLoss
#from utils.transform import *
#from utils.dataset import *
from utils.augmentations import SSDAugmentation
from utils.Firehouse_dataset import Firehouse_dataset


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
device = '0,1'
#torch.cuda.set_device('cuda:0,1')

parser = argparse.ArgumentParser(description="Trains the YOLO model.")
parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")

parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--pretrained_weights", type=str,default='./weights/yolov3.weights', help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--beta_dis', default=100.0, type=float,
                    help='beta distribution')
parser.add_argument('--type1coef', default=0.1, type=float,
                    help='type1coef')

# parser.add_argument('--gamma', default=0.1, type=float,
#                     help='Gamma update for SGD')

args = parser.parse_args()
# args = parser.parse_args([])

HOME = os.path.expanduser("~")

def collate_fn(batch):
    imgs, bb_targets, semis = list(zip(*batch))


#     # Resize images to input shape
    #imgs = torch.stack([resize(img, self.resize_factor) for img in imgs])
    imgs = torch.stack([img for img in imgs])
    semis = torch.stack([torch.from_numpy(semi) for semi in semis])
    

    for i, boxes in enumerate(bb_targets):
         boxes[:, 0] = i
    
    bb_targets = torch.cat(bb_targets, 0)
    
    

    #print('bb_____________',bb_targets)
    #print(type(bb_targets))

    #for i,val in enumerate(bb_targets):
    #    val = torch.from_numpy(val)
    #    idx = torch.ones(len(val),1)*i
    #    bb_targets[i] = torch.cat([idx,val],dim=1)

    return imgs, bb_targets, semis


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]


def train(args):
    yolo_net = load_model(args.model, args.pretrained_weights)
    
    cfg = coco512 #TODO voc300 cfg check 
    # dataset = VOC_firehouse_dataset_con(root=args.dataset_root,transform=SSDAugmentation(cfg['min_dim'] ,MEANS))
    #dataset = ListDataset(root = HOME , transform = DEFAULT_TRANSFORMS)
    dataset = Firehouse_dataset(root = HOME, dataset_type = 'YOLO', transform = SSDAugmentation(416,(104,117,123)))
    net = yolo_net
    #net = torch.nn.DataParallel(yolo_net, device_ids=[0,1])
    # cudnn.benchmark = True
    net = net.cuda()

        
    optimizer = optim.Adam(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # data_loader =torch.utils.data.DataLoader(dataset, args.batch_size,  # TODO? batch size 
    #                               num_workers=2,
    #                               shuffle=True, collate_fn=detection_collate,
    #                               pin_memory=True)
    data_loader =torch.utils.data.DataLoader(dataset, args.batch_size,  # TODO? batch size 
                                  num_workers=2,
                                  shuffle=True, collate_fn = collate_fn, #collate_fn=dataset.collate_fn,
                                  pin_memory=True,
                                  drop_last = True
                                  )    
    
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
    csd_criterion = CSDLoss(True) #TODO 
    isd_criterion = ISDLoss(True)
    
    for epoch in range(0,args.epochs):
        for data in tqdm(data_loader):

            images, targets, semi = data
            #images  = torch.tensor(images)
            
            #print(images)
            #print(targets)
            #print(semi)

            images = images.cuda()
            targets = targets.cuda()
            # batch index 

            images_flip = images.clone()
            images_flip = flip(images_flip, 3)
            
            
            #print(images.size())
                
            images_shuffle = images_flip.clone()
            images_shuffle[:int(args.batch_size / 2), :, :, :] = images_flip[int(args.batch_size / 2):, :, :, :]
            images_shuffle[int(args.batch_size / 2):, :, :, :] = images_flip[:int(args.batch_size / 2), :, :, :]

            lam = np.random.beta(args.beta_dis, args.beta_dis)

            images_mix = lam * images.clone() + (1 - lam) * images_shuffle.clone()

            out = net(images)
            out_flip = net(images_flip)
            out_mix = net(images_mix)

            # compute yolo loss
            loss, loss_components = compute_loss(out, targets,net)

            #csd, isd

            #origin
            conf_ori = list()
            loc_ori = list()

            for oo in out:
                append_conf = oo[:,:,:,:,:5].view(2,-1,5) #TODO 2 == batch size
                append_loc = oo[:,:,:,:,5:].view(2,-1,4)
                conf_ori.append(append_conf)
                loc_ori.append(append_loc)

            conf_ori = torch.cat([o for o in conf_ori],dim=1)
            loc_ori = torch.cat([o for o in loc_ori],dim=1)

            #flip and shuffle 
            conf_flip = list()
            loc_flip = list()
            conf_shuffle = list()
            loc_shuffle = list()

            for of in out_flip:
                conf_shuffle.append(of[:,:,:,:,:5].view(2,-1,5))  #TODO 2 == batch size
                loc_shuffle.append(of[:,:,:,:,5:].view(2,-1,4))

                append_conf = of[:,:,:,:,:5]
                append_loc = of[:,:,:,:,5:]
                append_conf = flip(append_conf,3).view(2,-1,5)
                append_loc = flip(append_loc,3).view(2,-1,4)
                conf_flip.append(append_conf)
                loc_flip.append(append_loc)


            conf_shuffle = torch.cat([o for o in conf_shuffle],dim=1)
            loc_shuffle = torch.cat([o for o in loc_shuffle],dim=1)

            conf_flip = torch.cat([o for o in conf_flip],dim=1)
            loc_flip = torch.cat([o for o in loc_flip],dim=1)        


            conf_mix = list()
            loc_mix  = list()


            #mixed
            for om in out_mix:
                append_conf = om[:,:,:,:,:5].view(2,-1,5) #TODO 2 == batch size
                append_loc = om[:,:,:,:,5:].view(2,-1,4)
                conf_mix.append(append_conf)
                loc_mix.append(append_loc)

            conf_mix = torch.cat([o for o in conf_mix],dim=1)
            loc_mix = torch.cat([o for o in loc_mix],dim=1)

            consistency_loss = csd_criterion(args, conf_ori, conf_flip, loc_ori, loc_flip, conf_consistency_criterion,yolo=True,num_classes=4)
            consistency_loss = consistency_loss.mean()
            
            interpolation_consistency_conf_loss, fixmatch_loss = isd_criterion(args, lam, conf_ori, conf_flip, loc_ori, loc_flip, conf_shuffle, conf_mix, loc_shuffle, loc_mix, conf_consistency_criterion,
                                                                      True,num_classes=4)
            interpolation_loss = torch.mul(interpolation_consistency_conf_loss.mean(), args.type1coef) + fixmatch_loss.mean()
    
            
            loss = loss + consistency_loss + interpolation_loss

            optimizer.zero_grad()
            loss.backward()
            

        #TODO eval 
            
        torch.save(net.state_dict(),'./weights/yolotest.pth')

if __name__ == '__main__':
    train(args)

