from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
import shutil
import logging

def set_logger(opt):
    # Set logger
    msg = []
    logger = logging.getLogger('%s' % opt.name)
    logger.setLevel(logging.INFO)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.isdir(save_dir):
        msg.append('%s not exist, make it' % save_dir)
        os.mkdir(save_dir)
    log_file_path = os.path.join(save_dir, 'log.log')
    os.system('cp models/%s_model.py %s'%(opt.model, save_dir))
    os.system('cp models/base_model.py %s'%save_dir)
    if os.path.isfile(log_file_path):
        target_path = log_file_path + '.%s' % time.strftime("%Y%m%d%H%M%S")
        msg.append('Log file exists, backup to %s' % target_path)
        shutil.move(log_file_path, target_path)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def sdmkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8,scale=None):
    
    if len(input_image.shape)<3: return None
   # if scale>0 and input_image.size()[1]==3:
   #     return tensor2im_logc(input_image, imtype=np.uint8,scale=scale)

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy[image_numpy<0] = 0
    image_numpy[image_numpy>255] = 255
    return image_numpy.astype(imtype)

def tensor2im_logc(image_tensor, imtype=np.uint8,scale=255):
    image_numpy = image_tensor.data[0].cpu().double().numpy()
    image_numpy = np.transpose(image_numpy,(1,2,0))
    image_numpy = (image_numpy+1) /2.0  
    image_numpy = image_numpy * (np.log(scale+1)) 
    image_numpy = np.exp(image_numpy) -1
    image_numpy = image_numpy.astype(np.uint8)

    return image_numpy.astype(np.uint8)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
