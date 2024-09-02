import os
import cv2
import torch
import numpy as np
from . import networks
import torch.nn.functional as F
from .base_model import BaseModel
from .distangle_model import DistangleModel
from util.metric import psnr, ssim
from .loss_function import VGG16FeatureExtractor, style_loss, perceptual_loss


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.clip(image_numpy, 0, 255).astype(imtype)


class FusionModel(DistangleModel):
    def name(self):
        return 'Fusion Net'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='expo_param')
        
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_names = ['M']
        opt.output_nc = 3

        self.ks = ks = opt.ks
        self.n = n = opt.n
        self.shadow_loss = opt.shadow_loss
        if self.opt.HSV_removalH:
            if self.opt.input_mask:
                self.netM = networks.define_G(1 + 3 + (2*n+1), ((3 + 1 + 2*n)) * 3 * ks * ks, opt.ngf, 'unet_256', opt.norm, 
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netM = networks.define_G(3 + (2*n+1), ((3 + 1 + 2*n)) * 3 * ks * ks, opt.ngf, 'unet_256', opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netM = networks.define_G(1 + 3 + n * 3, ((1 + n) * 3) * 3 * ks * ks, opt.ngf, 'unet_256', opt.norm, 
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)        
        self.netM.to(self.device)
        print(self.netM)
        self.lossNet = VGG16FeatureExtractor()
        self.lossNet.to(self.device)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizers = []

            if opt.optimizer == 'adam':
                self.optimizer_M = torch.optim.Adam(self.netM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            elif opt.optimizer == 'sgd':
                self.optimizer_M = torch.optim.SGD(self.netM.parameters(), momentum=0.9, lr=opt.lr, weight_decay=1e-5)
            else:
                assert False
            self.optimizers.append(self.optimizer_M)
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)        
        self.img_HSV = input['D'].to(self.device) if self.opt.input_RGB else self.input_img
        self.shadow_mask = input['B'].to(self.device)    
        self.imname = input['imname']
        self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float) # * 2 - 1
        self.nim = self.input_img.shape[1]
        self.shadowfree_img = input['C'].to(self.device)  
        self.VWeights = self.opt.VWeights
        self.SWeights = self.opt.SWeights


    def forward(self):
        n = self.input_img.shape[0]
        w = self.input_img.shape[2]
        h = self.input_img.shape[3]

        shadow_output_list = []
        shadow_image = self.img_HSV.clone() / 2 + 0.5
        if self.opt.HSV_removalH:
            shadow_output_list_show = []
            shadow_output_list.append(shadow_image[:, 0, :, :])
            S_img, V_img= shadow_image[:, 1, :, :], shadow_image[:, 2, :, :]
            for s in self.SWeights:
                shadow_output_list.append(S_img*s)
            for v in self.VWeights:
                shadow_output_list.append(V_img*v)
            for i in range(len(self.SWeights)):
                img_show = torch.cat([shadow_output_list[0], shadow_output_list[i+1], shadow_output_list[i+6]], dim=0)
                shadow_output_list_show.append(img_show)
            shadow_output = torch.stack(shadow_output_list, dim=1) 
            self.lit = torch.cat(shadow_output_list_show, dim=-1) * 2 - 1
        else:
            shadow_param_pred = torch.tensor([1, self.SWeights[0], 0.5, 0, 0, 0]).repeat(n, 1)  
            base_shadow_param_pred = shadow_param_pred[:, :2 * 3] # shadow_param_pred.view((n, self.n * 3, 2, 1, 1))
            self.base_shadow_param_pred = base_shadow_param_pred   
            base_shadow_output = shadow_image * base_shadow_param_pred[:, :3].view((n, 3, 1, 1)).to(self.device) + \
                                base_shadow_param_pred[:, 3:].view((n, 3, 1, 1)).to(self.device)
            for i in range(1, self.n):  #1,2,3,4    0.85,1.3,0.55,1.6 >>     0.275,0.425,0.5,0.65,0.8
                if i % 2 == 0:
                    scale = 1 + i * 0.15
                else:
                    scale = 1 - i * 0.15
                s = self.SWeights[i]/self.SWeights[0]
                scale = torch.tensor([1, s, scale]).repeat(n, 1).view((n, 3, 1, 1)).to(self.device)
                shadow_output_list.append(base_shadow_output * scale)  
            self.lit = torch.cat([base_shadow_output] + shadow_output_list, dim=-1) * 2 - 1  # [-1,1]
            shadow_output = torch.cat([base_shadow_output] + shadow_output_list, dim=1)    

        shadow_output = shadow_output * 2 - 1  # [-1,1]
        self.shadow_output = shadow_output
        if self.opt.input_mask:
            inputM = torch.cat([self.input_img, shadow_output, self.shadow_mask], 1)
        else:
            inputM = torch.cat([self.input_img, shadow_output], 1)
        out = torch.cat([self.input_img, shadow_output], 1)
        out = out / 2 + 0.5  # [0,1]
        out_matrix = F.unfold(out, stride=1, padding=self.ks // 2, kernel_size=self.ks) # N, C x \mul_(kernel_size), L

        kernel = self.netM(inputM) # b, (3+1)*n * 3 * ks * ks, Tanh
        b, c, h, w = self.input_img.shape
        output = []
        for i in range(b):
            feature = out_matrix[i, ...]            # ((1 + n) * 3) * ks * ks, L
            weight = kernel[i, ...]                 # ((1 + n) * 3) * 3 * ks * ks, H, W
            feature = feature.unsqueeze(0)          # 1, C, L
            if self.opt.HSV_removalH:
                weight = weight.view((3, (3 + 1 + 2*self.n) * self.ks * self.ks,  h * w))
            else:
                weight = weight.view((3, (self.n + 1) * 3 * self.ks * self.ks,  h * w))
            weight = F.softmax(weight, dim=1)
            iout = feature * weight              # (3, C, L)
            iout = torch.sum(iout, dim=1, keepdim=False)
            iout = iout.view((1, 3, h, w))
            output.append(iout)
        self.final = torch.cat(output, dim=0) * 2 - 1  # [-1 1]

    def backwardloss(self):
        criterionL1 = self.criterionL1
        lambda_ = self.opt.lambda_L1
        self.detect_loss = torch.tensor([0])
        real_feature = self.lossNet(self.shadowfree_img) 
        fake_feature = self.lossNet(self.final)
        loss_style = style_loss(real_feature, fake_feature) * lambda_          
        loss_content = perceptual_loss(real_feature, fake_feature) * lambda_    
        L1loss = criterionL1(self.final, self.shadowfree_img) * lambda_  
        self.rescontruct_loss = L1loss  + loss_style*120/6 + loss_content*0.05/5
        self.total_loss = self.rescontruct_loss * 20

    def optimize_parameters(self):
        
        self.forward() 
        self.optimizer_M.zero_grad()
        self.backwardloss() 
        self.total_loss.backward()
        self.optimizer_M.step()

    def visRemoval(self, e, s, path='',  eval = False):
        save_dir = os.path.join(self.save_dir, 'removal')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if len(path) > 0:
            save_dir_img = os.path.join(save_dir, path)
            if not os.path.isdir(save_dir_img):
                os.mkdir(save_dir_img)

        shadow = self.input_img
        output = self.final
        gt = self.shadowfree_img

        if eval:
            img = self.final[0, ...]
            filename = os.path.join(save_dir_img, self.imname[0])
            img = tensor2im(img)
            if not self.opt.gt_RGB:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = img[:, :, ::-1] 
        else:
            filename = os.path.join(save_dir_img, "epoch_%d_%s" % (e, self.imname[0]))
            if self.opt.HSV:
                series = cv2.cvtColor(tensor2im(self.lit), cv2.COLOR_HSV2RGB)
                shadow = tensor2im(shadow[0,...])
                output = tensor2im(output[0,...])
                gt = tensor2im(gt[0,...])
                if not self.opt.input_RGB:
                    shadow = cv2.cvtColor(shadow, cv2.COLOR_HSV2RGB) 
                if not self.opt.gt_RGB:
                    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
                    gt = cv2.cvtColor(gt, cv2.COLOR_HSV2RGB)
                img = np.concatenate([shadow, output, gt, series], 1)[:, :, ::-1] 
            else:
                img = torch.cat([shadow, output, gt, self.lit], axis=-1)[0, ...]
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        cv2.imwrite(filename, img)


    def record_metric(self, e, s, path=''):
        if len(path) > 0:
            save_dir = os.path.join(self.save_dir, path)
        else:
            save_dir = self.save_dir

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        output_tensor = self.final[0, ...]
        gt_tensor = self.shadowfree_img[0, ...]

        output = tensor2im(output_tensor)
        gt = tensor2im(gt_tensor)

        if self.opt.HSV:
            output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
            gt = cv2.cvtColor(gt, cv2.COLOR_HSV2BGR)
        else:
            output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
            gt = cv2.cvtColor(gt, cv2.COLOR_LAB2BGR)

        file_name = "epoch " + str(e)

        psnr_val = psnr(gt, output)
        ssim_gray, ssim_color = ssim(gt, output)

        f = open(os.path.join(save_dir, file_name + '.txt'), "a")
        f.write("im:{} ".format(self.imname[0]))
        f.write("e: {:<3} n: {:<4} ".format(e,s))
        f.write("psnr: {: .3f}  ssim_gray: {: .3f}  ssim_color: {: .3f} \n".format(psnr_val, ssim_gray, ssim_color))

        f.close()

        file_name_psnr = "epoch " + str(e) + "_psnr"
        f = open(os.path.join(save_dir, file_name_psnr + '.txt'), "a")
        f.write("{:^8.4f}\n".format(psnr_val))
        f.close()

        file_name_ssimg = "epoch " + str(e) + "_ssimg"
        f = open(os.path.join(save_dir, file_name_ssimg + '.txt'), "a")
        f.write("{:^8.4f}\n".format(ssim_gray))
        f.close()

        file_name_ssimc = "epoch " + str(e) + "_ssimc"
        f = open(os.path.join(save_dir, file_name_ssimc + '.txt'), "a")
        f.write("{:^8.4f}\n".format(ssim_color))
        f.close()

    def vis_and_metric_Detection(self, e, s, path='', saveImg= False):
        return 0, 0

    def metricRemoval(self, e, s, path=''):

        save_dir = os.path.join(self.save_dir, 'removal')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        output_tensor = self.final[0, ...]
        gt_tensor = self.shadowfree_img[0, ...]

        output = tensor2im(output_tensor)   # HSV
        gt = tensor2im(gt_tensor)

        psnr_val = psnr(gt, output)  
        ssim_color = ssim(gt, output)

        if self.opt.HSV:
            if not self.opt.gt_RGB:
                output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
                gt = cv2.cvtColor(gt, cv2.COLOR_HSV2BGR)
        else:
            output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
            gt = cv2.cvtColor(gt, cv2.COLOR_LAB2BGR)

        psnr_val_RGB = psnr(gt, output)
        ssim_color_RGB = ssim(gt, output)

        file_name = "epoch " + str(e) + "psnr&ssim"

        f = open(os.path.join(save_dir, file_name + '.txt'), "a")
        f.write("im:{} ".format(self.imname[0]))
        f.write("e: {:<3} n: {:<4} ".format(e,s))
        f.write("psnr_RGB: {: .3f}  ssim_color_RGB: {: .3f}  psnr: {: .3f}  ssim_color: {: .3f} \n".format(psnr_val_RGB, ssim_color_RGB,
                                                                                                           psnr_val, ssim_color))

        f.close()
        return psnr_val_RGB, ssim_color_RGB