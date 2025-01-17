from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss, CSDLoss, ISDLoss
from ssd import build_ssd
# from ssd_consistency import build_ssd_con
from isd import build_ssd_con
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import copy
from data.VOCfirehouse_consistency import VOC_firehouse_dataset_con
from utils.Firehouse_dataset import Firehouse_dataset 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "6,7"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
#parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512'],
#                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset', default='fire', choices=['VOC300', 'VOC512','fire'],
                    type=str, help='VOC300 or VOC512')

parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,  # None  'weights/ssd300_COCO_80000.pth'
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--beta_dis', default=100.0, type=float,
                    help='beta distribution')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--type1coef', default=0.1, type=float,
                    help='type1coef')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

# torch.cuda.set_device(1)


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC300':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc300
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'VOC512':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc512
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'fire':
        #if args.dataset_root == DATA_ROOT:
        cfg = voc300
        #dataset = VOC_firehouse_dataset_con(root=args.dataset_root,transform=SSDAugmentation(cfg['min_dim'] ,MEANS))
        dataset = Firehouse_dataset(root=args.dataset_root,transform=SSDAugmentation(cfg['min_dim'] ,MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    finish_flag = True

    while(finish_flag):
        
        cfg['num_classes'] = 5
        ssd_net = build_ssd_con('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net

        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            ssd_net.load_weights(args.resume)
        else:
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)
            # ssd_net.vgg_t.load_state_dict(vgg_weights)

        if args.cuda:
            net = net.cuda()

        if not args.resume:
            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, args.cuda)
        csd_criterion = CSDLoss(args.cuda)
        isd_criterion = ISDLoss(args.cuda)
        conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()



        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0
        epoch = 0
        supervised_flag = 1
        print('Loading the dataset...')

        step_index = 0


        if args.visdom:
            vis_title = 'SSD.PyTorch on ' + dataset.name
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
            epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)


        total_un_iter_num = 0


        supervised_batch =  args.batch_size
        #unsupervised_batch = args.batch_size - supervised_batch
        #data_shuffle = 0

        #if(args.start_iter==0):
        #    dataset = VOCDetection_con_init(root=args.dataset_root,
        #                                             transform=SSDAugmentation(cfg['min_dim'],
        #                                                                          MEANS))
        #else:
        #    supervised_flag = 0
        #    dataset = VOCDetection_con(root=args.dataset_root,
        #                                             transform=SSDAugmentation(cfg['min_dim'],
        #                                                                          MEANS))#,shuffle_flag=data_shuffle)
            #data_shuffle = 1

        data_loader = data.DataLoader(dataset, args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, collate_fn=detection_collate,
                                                   generator = torch.Generator(device='cuda'),
                                                   pin_memory=True, drop_last=True)


        batch_iterator = iter(data_loader)

        for iteration in range(args.start_iter, cfg['max_iter']):
            if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
                update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                                'append', epoch_size)
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            try:
                images, targets, semis = next(batch_iterator)
            except StopIteration:
                #supervised_flag = 0
                #dataset = VOCDetection_con(root=args.dataset_root,
                #                                         transform=SSDAugmentation(cfg['min_dim'],
                #                                                                      MEANS))#, shuffle_flag=data_shuffle)
                #data_loader = data.DataLoader(dataset, args.batch_size,
                #                                           num_workers=args.num_workers,
                #                                           shuffle=True, collate_fn=detection_collate,
                #                                           pin_memory=True, drop_last=True)
                #batch_iterator = iter(data_loader)
                #images, targets, semis = next(batch_iterator)
                pass

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()

            images_flip = images.clone()
            images_flip = flip(images_flip, 3)

            images_shuffle = images_flip.clone()
            images_shuffle[:int(args.batch_size / 2), :, :, :] = images_flip[int(args.batch_size / 2):, :, :, :]
            images_shuffle[int(args.batch_size / 2):, :, :, :] = images_flip[:int(args.batch_size / 2), :, :, :]

            lam = np.random.beta(args.beta_dis, args.beta_dis)


            images_mix = lam * images.clone() + (1 - lam) * images_shuffle.clone()

            out, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation = net(images, images_flip, images_mix)


            sup_image_binary_index = np.zeros([len(semis),1])

            for super_image in range(len(semis)):
                if(int(semis[super_image])==1):
                    sup_image_binary_index[super_image] = 1
                else:
                    sup_image_binary_index[super_image] = 0

                if(int(semis[len(semis)-1-super_image])==0):
                    del targets[len(semis)-1-super_image]


            sup_image_index = np.where(sup_image_binary_index == 1)[0]
            unsup_image_index = np.where(sup_image_binary_index == 0)[0]

            loc_data, conf_data, priors = out

            if (len(sup_image_index) != 0):
                loc_data = loc_data[sup_image_index,:,:]
                conf_data = conf_data[sup_image_index,:,:]
                output = (
                    loc_data,
                    conf_data,
                    priors
                )

            # backprop
            # loss = Variable(torch.cuda.FloatTensor([0]))
            loss_l = Variable(torch.cuda.FloatTensor([0]))
            loss_c = Variable(torch.cuda.FloatTensor([0]))



            if(len(sup_image_index)!=0):
                try:
                    loss_l, loss_c = criterion(output, targets)
                except:
                    break
                    print('--------------')


            consistency_loss = csd_criterion(args, conf, conf_flip, loc, loc_flip, conf_consistency_criterion)
            interpolation_consistency_conf_loss, fixmatch_loss = isd_criterion(args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion)
            consistency_loss = consistency_loss.mean()
            interpolation_loss = torch.mul(interpolation_consistency_conf_loss.mean(), args.type1coef) + fixmatch_loss.mean()


            ramp_weight = rampweight(iteration)
            consistency_loss = torch.mul(consistency_loss, ramp_weight)
            interpolation_loss = torch.mul(interpolation_loss,ramp_weight)

            if(supervised_flag ==1):
                loss = loss_l + loss_c + consistency_loss + interpolation_loss
            else:
                if(len(sup_image_index)==0):
                    loss = consistency_loss + interpolation_loss
                else:
                    loss = loss_l + loss_c + consistency_loss + interpolation_loss


            if(loss.data>0):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t1 = time.time()
            if(len(sup_image_index)==0):
                loss_l.data = Variable(torch.cuda.FloatTensor([0]))
                loss_c.data = Variable(torch.cuda.FloatTensor([0]))
            else:
                loc_loss += loss_l.data  # [0]
                conf_loss += loss_c.data  # [0]


            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f || consistency_loss : %.4f ||' % (loss.data, consistency_loss.data), end=' ')
                print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, loss_interpolation: %.4f, lr : %.4f, super_len : %d\n' % (loss.data, loss_c.data, loss_l.data, consistency_loss.data, interpolation_loss.data, float(optimizer.param_groups[0]['lr']),len(sup_image_index)))


            if(float(loss)>100):
                break

            if args.visdom:
                update_vis_plot(iteration, loss_l.data, loss_c.data,
                                iter_plot, epoch_plot, 'append')

            #if iteration != 0 and (iteration+1) % 120000 == 0:
            if iteration != 0 and (iteration+1) % 40 == 0:
                print('Saving state, iter:', iteration)
                #torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                #           repr(iteration+1) + '.pth')
        #torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')
        torch.save(ssd_net.state_dict(),'weights/ssd300_fire.pth') 
        print('-------------------------------\n')
        print(loss.data)
        print('-------------------------------')

        if((iteration +1) ==cfg['max_iter']):
            finish_flag = False


def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000
    coef = 1

    if(iteration<ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end),2))
    elif(iteration>ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2))
#        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2))
    else:
        ramp_weight = 1  


    if(iteration==0):
        ramp_weight = 0

    return ramp_weight * coef




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]


if __name__ == '__main__':
    train()

