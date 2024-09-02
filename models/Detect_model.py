import os
import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from .distangle_model import DistangleModel
from util.cal_accuracy_and_ber import BER
from torchvision import transforms as T

class DetectModel(DistangleModel):
    def name(self):
        return 'Detection Net'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(pool_size=0, no_lsgan=True)
        parser.set_defaults(dataset_mode='expo_param')
        parser.add_argument('--wdataroot', default='None',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--use_our_mask', action='store_true')
        parser.add_argument('--mask_train', type=str, default=None)
        parser.add_argument('--mask_test', type=str, default=None)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_param', 'alpha', 'rescontruction']
        self.visual_names = ['input_img', 'litgt', 'alpha_pred', 'out', 'final', 'outgt']
        self.model_names = ['D']
        opt.output_nc = 3

        self.ks = ks = opt.ks
        self.n = n = opt.n
        self.shadow_loss = opt.shadow_loss
        self.netD = networks.define_G(3, 1, opt.ngf, 'unet_256', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD.to(self.device)
        print(self.netD)

        if self.isTrain:
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []

            if opt.optimizer == 'adam':
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            elif opt.optimizer == 'sgd':
                self.optimizer_D = torch.optim.SGD(self.netD.parameters(), momentum=0.9, lr=opt.lr, weight_decay=1e-5)
            else:
                assert False

            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.imname = input['imname']
        self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float)  # mask >> [0,1]
        self.nim = self.input_img.shape[1]

    def forward(self):
        inputD = self.input_img
        self.detection = self.netD(inputD)  # [-1,1]

    # add by chen
    def calloss(self):

        lambda_ = self.opt.lambda_L1
        loss_F_my = self.bce(self.detection, self.shadow_mask) * lambda_
        # loss_F_my_1 = self.criterionL1(self.detection, self.shadow_mask*2-1) * lambda_*5
        # loss_F_my_2 = self.focalloss(self.detection, self.shadow_mask) * lambda_
        
        self.loss_my = loss_F_my

    def backward(self):
        lambda_ = self.opt.lambda_L1
        self.loss_F = self.bce(self.detection, self.shadow_mask) * lambda_
        # self.loss_F_1 = self.criterionL1(self.detection, self.shadow_mask*2-1) * lambda_*5
        # self.loss_F_2 = self.focalloss(self.detection, self.shadow_mask) * lambda_
        
        self.loss = self.loss_F
        self.loss.backward()

    def optimize_parameters(self):
        # self.netF.zero_grad()
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward()
        self.optimizer_D.step()
  
def save_detection(self, e, s, path='', eval=False):

        if not eval:

            if len(path) > 0:
                save_dir = os.path.join(self.save_dir, path)
            else:
                save_dir = self.save_dir

            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            output = self.detection
            gt = self.shadow_mask * 2 - 1
            
            
            if eval:
                img = self.detection[0, ...]
                filename = os.path.join(save_dir, self.imname[0])
            else:
                img = torch.cat([output, gt], axis=-1)[0, ...]
                filename = os.path.join(save_dir, "epoch_%d_step_%d_%s" % (e, s, self.imname[0]))

            img = T.ToPILImage()(img.to('cpu').ge(0.5).to(torch.float32))
            img.save(filename,quality=95)
            
            
            
            pre = T.ToPILImage()(output.to('cpu').ge(0.5).to(torch.float32)[0, ...])
            gt = T.ToPILImage()(gt.to('cpu').ge(0.5).to(torch.float32)[0,...])
            score, accuracy = BER(torch.from_numpy(np.array(pre)).float(), torch.from_numpy(np.array(gt)).float())
            
            f = open(os.path.join(self.save_dir, str(e) + '_metric.txt'), "a")
            f.write("e: {:<3} n: {:<4} ".format(e, s))
            f.write("im:{}  ".format(self.imname[0]))
            f.write("Accuracy: {:^8.4f}  ".format(accuracy))
            f.write("BER: {:^8.4f}\n".format(score))
            f.close()
            
        else:

            pre = self.detection
            gt = self.shadow_mask * 2 - 1

            basepath = os.path.join(path, str(e))
            if not os.path.isdir(basepath):
                os.mkdir(basepath)

            save_dir_pre = os.path.join(basepath, 'pre')
            if not os.path.isdir(save_dir_pre):
                os.mkdir(save_dir_pre)
            pre_filename = os.path.join(save_dir_pre, "%s" % (self.imname[0]))

            save_dir_gt = os.path.join(basepath, 'gt')
            if not os.path.isdir(save_dir_gt):
                os.mkdir(save_dir_gt)
            gt_filename = os.path.join(save_dir_gt, "%s" % (self.imname[0]))

            # score, accuracy = BER(gt, pre)

            pre = T.ToPILImage()(pre.to('cpu').ge(0.5).to(torch.float32)[0, ...])
            gt = T.ToPILImage()(gt.to('cpu').ge(0.5).to(torch.float32)[0,...])
            # pre.save(pre_filename, quality=95)
            # gt.save(gt_filename, quality=95)

            score, accuracy = BER(torch.from_numpy(np.array(pre)).float(), torch.from_numpy(np.array(gt)).float())

            f = open(os.path.join(path, str(e) + '_metric.txt'), "a")
            f.write("e: {:<3} n: {:<4} ".format(e, s))
            f.write("im:{}  ".format(self.imname[0]))
            f.write("Accuracy: {:^8.4f}  ".format(accuracy))
            f.write("BER: {:^8.4f}\n".format(score))
            f.close()
