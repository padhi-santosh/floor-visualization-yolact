import sys
sys.path.append("../django_yolact/floor_mask/")



from yolact_cpu.data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from yolact_cpu.utils.augmentations import BaseTransform, FastBaseTransform, Resize
from yolact_cpu.utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from yolact_cpu.utils import timer
from yolact_cpu.utils.functions import SavePath
from yolact_cpu.layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import cv2
from PIL import Image



### ARGUMETS ##################################################################################

trained_model='/home/sky/Desktop/yolact/django_yolact/floor_mask/yolact_cpu/weights/yolact_plus_resnet50_floor_1249_30000.pth'
#image="/home/sky/Desktop/yolact/yolact-cpu/test_images/floor14.jpg"
config = 'yolact_resnet50_pascal_config'
top_k=15
cuda = False
fast_nms = True
display_masks = True
display_bboxes = True
display_text = True 
display_scores = True
display_lincomb = False
mask_proto_debug = False
score_threshold = 0
dataset = None
crop=True
        

if config is not None:
    set_cfg(config)

if trained_model == 'interrupt':
    trained_model = SavePath.get_interrupt('weights/')
elif trained_model == 'latest':
    trained_model = SavePath.get_latest('weights/', cfg.name)


######### Resize image to 600*600 #########

#path="/home/sky/Desktop/yolact/yolact-cpu/test_images/floor14.jpg"
#im = Image.open(path)

#image = im.resize((600,600), Image.ANTIALIAS)

#image = cv2.imread(path)
#image = cv2.resize(image,(600,600))

##################################################################################



def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        # img_gpu = torch.Tensor(img_numpy).cuda()
        img_gpu = torch.Tensor(img_numpy)
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb = display_lincomb,
                                        crop_masks        = crop,
                                        score_threshold   = score_threshold)
        # torch.cuda.synchronize()
        # torch.synchronize()

    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][:top_k]
        classes, scores, boxes = [x[:top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break
    
    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
                # color = np.asarray(color)
                # print("HI")
                # pdb.set_trace()
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        # pdb.set_trace()
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        # colors = torch.cat([get_color(j, on_gpu=img.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        colors = torch.cat([torch.FloatTensor(get_color(j, on_gpu=img.device.index)).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        # for j in range num_dets_to_consider
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    if display_text or display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy

##################################################################################

def img_out(image):

    def evalimage(net:Yolact, path:str, save_path:str=None):
        # frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        frame = torch.from_numpy(cv2.imread(path)).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)

        img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        
        if save_path is None:
            img_numpy = img_numpy[:, :, (2, 1, 0)]
            return img_numpy

        #if save_path is None:
            #plt.imshow(img_numpy)
            #plt.title(path)
            #plt.show()

        #else:
            #cv2.imwrite(save_path, img_numpy)

      
    ##################################################################################


    def evaluate(net:Yolact, dataset, train_mode=False):
        net.detect.use_fast_nms = fast_nms
        cfg.mask_proto_debug = mask_proto_debug

        if image is not None:
            if ':' in image:
                inp, out = image.split(':')
                img = evalimage(net, inp, out)
            else:
                img = evalimage(net, image)
            return img
        
        

    #################################################################################


    with torch.no_grad():

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(trained_model)
        # pdb.set_trace()
        net.eval()
        print(' Done.')

        if cuda:
            # net = net.cuda()
            net = net

        img = evaluate(net, dataset)
        return img


