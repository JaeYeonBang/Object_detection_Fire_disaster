# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from data import coco as cfg
# from ..box_utils import match, log_sum_exp

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    eps = 1e-5
    
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh+eps) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


class ISDLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss, self).__init__()
        self.use_gpu = use_gpu

#     def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion):
    def forward(self, batch_size, lam, conf, loc, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion):


        ### interpolation regularization
        # out, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation
        conf_temp = conf_shuffle.clone()
        loc_temp = loc_shuffle.clone()
        
        conf_temp[:int(batch_size / 2), :, :] = conf_shuffle[int(batch_size / 2):, :, :]
        conf_temp[int(batch_size / 2):, :, :] = conf_shuffle[:int(batch_size / 2), :, :]
        loc_temp[:int(batch_size / 2), :, :] = loc_shuffle[int(batch_size / 2):, :, :]
        loc_temp[int(batch_size / 2):, :, :] = loc_shuffle[:int(batch_size / 2), :, :]
        
        
        
        ## original background elimination
        left_conf_class = conf[:, :, :].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val>0.5
        left_mask_val = left_mask_val.data

        ## flip background elimination
        right_conf_class = conf_temp[:, :, :].clone()
        right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
        right_mask_val = right_each_val>0.5
        right_mask_val = right_mask_val.data
        
#         conf = torch.nn.functional.log_softmax(conf,dim=2)
#         conf_temp = torch.nn.functional.log_softmax(conf_temp,dim=2)
#         conf_interpolation = torch.nn.functional.log_softmax(conf_interpolation,dim=2)
        
        
        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_right_mask_val = right_mask_val.float() * (1 - left_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        only_right_mask_val = only_right_mask_val.bool()

        intersection_mask_val = left_mask_val * right_mask_val
        
        ##################    Type-I_######################
        intersection_mask_conf_index = intersection_mask_val.unsqueeze(2).expand_as(conf)

        intersection_left_conf_mask_sample = conf.clone()
#         intersection_left_conf_sampled = intersection_left_conf_mask_sample[intersection_mask_conf_index].view(-1,
#                                                                                                                21)
        intersection_left_conf_sampled = intersection_left_conf_mask_sample[intersection_mask_conf_index].view(-1,
                                                                                                               4)

        
        intersection_right_conf_mask_sample = conf_temp.clone()
#         intersection_right_conf_sampled = intersection_right_conf_mask_sample[intersection_mask_conf_index].view(-1,
#                                                                                                                  21)
        intersection_right_conf_sampled = intersection_right_conf_mask_sample[intersection_mask_conf_index].view(-1,
                                                                                                                 4)


        intersection_intersection_conf_mask_sample = conf_interpolation.clone()
#         intersection_intersection_sampled = intersection_intersection_conf_mask_sample[
#             intersection_mask_conf_index].view(-1, 21)
        intersection_intersection_sampled = intersection_intersection_conf_mask_sample[
        intersection_mask_conf_index].view(-1, 4)

    
        if (intersection_mask_val.sum() > 0):

            mixed_val = lam * intersection_left_conf_sampled + (1 - lam) * intersection_right_conf_sampled

            mixed_val = mixed_val + 1e-7
            intersection_intersection_sampled = intersection_intersection_sampled + 1e-7

            interpolation_consistency_conf_loss_a = conf_consistency_criterion(mixed_val.log(),
                                                                               intersection_intersection_sampled.detach()).sum(
                -1).mean()
            interpolation_consistency_conf_loss_b = conf_consistency_criterion(
                intersection_intersection_sampled.log(),
                mixed_val.detach()).sum(-1).mean()
            interpolation_consistency_conf_loss = interpolation_consistency_conf_loss_a + interpolation_consistency_conf_loss_b
            interpolation_consistency_conf_loss = torch.div(interpolation_consistency_conf_loss, 2)
        else:
            interpolation_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            interpolation_consistency_conf_loss = interpolation_consistency_conf_loss.data[0]

        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)
        only_left_mask_loc_index = only_left_mask_val.unsqueeze(2).expand_as(loc)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_loc_mask_sample = loc.clone()
#         ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 4)
        ori_fixmatch_loc_sampled = ori_fixmatch_loc_mask_sample[only_left_mask_loc_index].view(-1, 4)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_loc_mask_sample_interpolation = loc_interpolation.clone()
#         ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
#             only_left_mask_conf_index].view(-1, 21)
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 4)

        
        ori_fixmatch_loc_sampled_interpolation = ori_fixmatch_loc_mask_sample_interpolation[
            only_left_mask_loc_index].view(-1, 4)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

            ## LOC LOSS
            only_left_consistency_loc_loss_x = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 0] - ori_fixmatch_loc_sampled[:, 0].detach(),
                exponent=2))
            only_left_consistency_loc_loss_y = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 1] - ori_fixmatch_loc_sampled[:, 1].detach(),
                exponent=2))
            only_left_consistency_loc_loss_w = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 2] - ori_fixmatch_loc_sampled[:, 2].detach(),
                exponent=2))
            only_left_consistency_loc_loss_h = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 3] - ori_fixmatch_loc_sampled[:, 3].detach(),
                exponent=2))

            only_left_consistency_loc_loss = torch.div(
                only_left_consistency_loc_loss_x + only_left_consistency_loc_loss_y + only_left_consistency_loc_loss_w + only_left_consistency_loc_loss_h,
                4)

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]
            only_left_consistency_loc_loss = only_left_consistency_loc_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss + only_left_consistency_loc_loss




        ##################    Type-II_B ######################

        only_right_mask_conf_index = only_right_mask_val.unsqueeze(2).expand_as(conf)
        only_right_mask_loc_index = only_right_mask_val.unsqueeze(2).expand_as(loc)

        flip_fixmatch_conf_mask_sample = conf_temp.clone()
        flip_fixmatch_loc_mask_sample = loc_temp.clone()
#         flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 21)
        flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 4)

        flip_fixmatch_loc_sampled = flip_fixmatch_loc_mask_sample[only_right_mask_loc_index].view(-1, 4)

        flip_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        flip_fixmatch_loc_mask_sample_interpolation = loc_interpolation.clone()
#         flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
#             only_right_mask_conf_index].view(-1, 21)
        flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
            only_right_mask_conf_index].view(-1, 4)

        flip_fixmatch_loc_sampled_interpolation = flip_fixmatch_loc_mask_sample_interpolation[
            only_right_mask_loc_index].view(-1, 4)

        if (only_right_mask_val.sum() > 0):
            ## KLD !!!!!1
            flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_sampled_interpolation + 1e-7
            flip_fixmatch_conf_sampled = flip_fixmatch_conf_sampled + 1e-7
            only_right_consistency_conf_loss_a = conf_consistency_criterion(
                flip_fixmatch_conf_sampled_interpolation.log(),
                flip_fixmatch_conf_sampled.detach()).sum(-1).mean()
            # consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
            #                                                      conf_sampled.detach()).sum(-1).mean()
            # consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b
            only_right_consistency_conf_loss = only_right_consistency_conf_loss_a

            ## LOC LOSS
            only_right_consistency_loc_loss_x = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 0] - flip_fixmatch_loc_sampled[:, 0].detach(),
                    exponent=2))
            only_right_consistency_loc_loss_y = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 1] - flip_fixmatch_loc_sampled[:, 1].detach(),
                    exponent=2))
            only_right_consistency_loc_loss_w = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 2] - flip_fixmatch_loc_sampled[:, 2].detach(),
                    exponent=2))
            only_right_consistency_loc_loss_h = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 3] - flip_fixmatch_loc_sampled[:, 3].detach(),
                    exponent=2))

            only_right_consistency_loc_loss = torch.div(
                only_right_consistency_loc_loss_x + only_right_consistency_loc_loss_y + only_right_consistency_loc_loss_w + only_right_consistency_loc_loss_h,
                4)

        else:
            only_right_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_conf_loss = only_right_consistency_conf_loss.data[0]
            only_right_consistency_loc_loss = only_right_consistency_loc_loss.data[0]

        # consistency_loss = consistency_conf_loss  # consistency_loc_loss
        only_right_consistency_loss = only_right_consistency_conf_loss + only_right_consistency_loc_loss
        #            only_right_consistency_loss = only_right_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss + only_right_consistency_loss
        return interpolation_consistency_conf_loss, fixmatch_loss

