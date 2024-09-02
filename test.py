import os
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from options.print_option import print_options, print_options_in_log
from util.util import set_logger

def main(opt):
    
    logger = set_logger(opt)
    print_options(opt)
    print_options_in_log(opt, logger)

    opt.phase = 'test/test_'
    opt.batch_size = 1
    opt.serial_batches = True
    test_data_loader = CreateDataLoader(opt)
    test_dataset = test_data_loader.load_data()
    es = opt.batch_size / len(test_dataset) 
    model = create_model(opt) 
    
    model.setup(opt)  
    load_epoch = opt.load_epoch
    print('load fusion net from:', opt.load_dir)
    model.load_networks(load_epoch, opt.load_dir, logger)

    total_steps = 0
    model.eval()
    ber_sum = psnr_sum = ssim_sum = acc_sum = 0
    epoch = 0
    for i, data in enumerate(test_dataset):
        total_steps += opt.batch_size
        model.set_input(data)
        model.forward()   

        vis_path = "epoch_test_" + str(epoch)    
        model.visRemoval(epoch, i+1, vis_path, eval = True)      
        psnr_val, ssim_val = model.metricRemoval(epoch, i+1, vis_path) 
        ber, acc = model.vis_and_metric_Detection(epoch, i+1, vis_path, saveImg = True)
        psnr_sum += psnr_val
        ssim_sum += ssim_val
        ber_sum += ber
        acc_sum += acc

    logger.info('[Eval] [Epoch] %d | ber: %.5f| acc: %.5f | psnr: %.5f | ssim: %.5f '  % (epoch, ber_sum*es, acc_sum*es, psnr_sum*es, ssim_sum*es))

    f = open(os.path.join(model.save_dir,  'ber.txt'), "a")
    f.write("e: {:<3} ber: {:^8.4f}\n".format(epoch, ber_sum*es))
    f.close

    f = open(os.path.join(model.save_dir,  'psnr.txt'), "a")
    f.write("e: {:<3} psnr: {:^8.4f}\n".format(epoch, psnr_sum*es))
    f.close
    
    f = open(os.path.join(model.save_dir,  'ssim.txt'), "a")
    f.write("e: {:<3} ssim: {:^8.4f}\n".format(epoch, ssim_sum*es))
    f.close


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.PretrainD = False   
    opt.joint_train = False 
    opt.loadSize = 256     # scale images to this size
    opt.randomSize = True  
    opt.keep_ratio = True  
    opt.HSV = True   
    opt.HSV_removalH = True   
    opt.gt_RGB = True    
    opt.input_RGB = True    
    opt.input_mask = False    
    opt.isTrain = False  
    opt.mode = 'SVPC'              
    SWeights = [2, 1.8, 1.6, 1.4, 1.2]
    opt.SWeights = SWeights
    opt.VWeights = [0.8, 0.65, 0.5, 0.425, 0.275]

    opt.dataroot = '/your/dataset/path/of/ShiQ' 
    opt.fineSize = 256    # crop images to this size 
    opt.model = 'DF'      
    # opt.model = 'Fusion'     
    opt.batch_size = 24   # 24 
    opt.phase = 'train_'  
    opt.gpu_ids = [0]
    opt.lr = 0.001             
    opt.lambda_L1 = 100.0      
    opt.num_threads = 24
    opt.dataset_mode = 'expo_param' 
    opt.optmizer = 'adam'       
    opt.n = 5                  
    opt.ks = 3                 
    opt.lr_policy = 'step'      
    opt.lr_decay_iters = 80
    opt.gamma = 0.5            
    opt.load_epoch = 460
    opt.load_dir = 'None'   
    opt.save_epoch_freq = 100 
    opt.niter = 200     
    opt.niter_decay = 300  
    name = "{}Net_lr{}_{}_ks{}_PreD{}_Joint{}_gpu{}_{}_HSV{}_removalH_{}_gtRGB{}_input_RGB{}_input_mask{}".format(opt.model, opt.lr, opt.lr_policy, opt.ks, 
                                                                 opt.PretrainD, opt.joint_train, opt.gpu_ids[0], opt.mode, opt.HSV, opt.SWeights[0], opt.gt_RGB, opt.input_RGB, opt.input_mask) 
    opt.name = name
    opt.checkpoints_dir = './test_log'
    print_options(opt)
    main(opt)
