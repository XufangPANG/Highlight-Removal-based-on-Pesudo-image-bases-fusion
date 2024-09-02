import time
import os
from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from options.print_option import print_options, print_options_in_log
from util.util import set_logger

def main(opt):

    logger = set_logger(opt)
    print_options(opt)
    print_options_in_log(opt, logger)

    opt.phase = 'train/train_' 
    opt.serial_batches = False
    train_data_loader = CreateDataLoader(opt)
    train_dataset = train_data_loader.load_data()
    ts = opt.batch_size / len(train_dataset)            # for log train loss

    opt.phase = 'test/test_'
    opt.batch_size = 1
    opt.serial_batches = True
    test_data_loader = CreateDataLoader(opt)
    test_dataset = test_data_loader.load_data()
    es = opt.batch_size / len(test_dataset) 

    model = create_model(opt) 
    model.setup(opt) 

    load_epoch = 'latest' 
    if opt.load_dir and opt.load_dir != 'None': 
        print('load fusion net from:', opt.load_dir, load_epoch)
        model.load_networks(load_epoch, opt.load_dir, logger)

    total_steps = 0
    opt.epoch_count = 1
    epochs = opt.niter + opt.niter_decay+1
    best = tmp =  0
    for epoch in range(opt.epoch_count, epochs):
        
        epoch_start_time = time.time()
        epoch_iter = 0
        model.epoch = epoch
        model.train()
        
        d_loss_sum = r_loss_sum = t_loss_sum = 0 
        for i, data in enumerate(train_dataset):
            # iter_start_time = time.time()
            total_steps += 1
            model.set_input(data)  
            model.optimize_parameters()  
                        
            if opt.PretrainD:  
                detect_loss = model.detect_loss.detach().item()     
                if total_steps % 20 == 0:       
                # Do log
                    logger.info('[Train] [Epoch] %d | detect_loss : %.4f  '  % (epoch, detect_loss))
                d_loss_sum += detect_loss 
        
        
            else:   
                detect_loss, reconst_loss, total_loss = model.detect_loss.detach().item(), model.rescontruct_loss.detach().item(), model.total_loss.detach().item() 
                if total_steps % 20 == 0:
                    # Do log
                    logger.info('[Train] [Epoch] %d/%d | detect_loss : %.4f | reconst_loss: %.4f | total_loss: %.4f '  % (epoch, epochs, detect_loss, reconst_loss, total_loss))
                d_loss_sum += detect_loss
                r_loss_sum += reconst_loss
                t_loss_sum += total_loss

        if opt.PretrainD: 
            logger.info('[Train] [Epoch] %d/%d | detect_loss : %.5f '  % (epoch, epochs, d_loss_sum*ts))  
    
            f = open(os.path.join(model.save_dir,  'PretrainDloss.txt'), "a")  
            f.write("e: {:<3} d_loss: {:^8.4f} \n".format(epoch, d_loss_sum*ts))
            f.close

        else:
            logger.info('[Train] [Epoch] %d/%d | detect_loss_ts : %.5f | reconst_loss_ts: %.5f | total_loss_ts: %.5f '  % (epoch, epochs, d_loss_sum*ts, r_loss_sum*ts, t_loss_sum*ts)) # 打印信息

            f = open(os.path.join(model.save_dir,  'trainloss.txt'), "a") 
            f.write("e: {:<3} d_loss: {:^8.4f} r_loss: {:^8.4f} t_loss: {:^8.4f}\n".format(epoch, d_loss_sum*ts, r_loss_sum*ts, t_loss_sum*ts))
            f.close

        if (epoch >= (opt.niter + opt.niter_decay - 50) and epoch % opt.eval_freq == 0 ) or (epoch==1) :
            model.eval()
            d_loss_sum = r_loss_sum = t_loss_sum = 0
            ber_sum = psnr_sum = ssim_sum = acc_sum = 0
            saveImg_flag = epoch % 100 == 0 or (epoch == opt.niter + opt.niter_decay) or (epoch==1)

            for i, data in enumerate(test_dataset):
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.forward()      
                model.backwardloss() 

                if opt.PretrainD: 
                    detect_loss = model.detect_loss.detach().item()
                    d_loss_sum += detect_loss

                    vis_path = "epoch_" + str(epoch) 
                    ber, acc = model.vis_and_metric_Detection(epoch, i+1, vis_path, saveImg = saveImg_flag)  
                    ber_sum += ber
                    acc_sum += acc
                    
                else:  
                    detect_loss, reconst_loss, total_loss = model.detect_loss.detach().item(), model.rescontruct_loss.detach().item(), model.total_loss.detach().item()
                    d_loss_sum += detect_loss
                    r_loss_sum += reconst_loss
                    t_loss_sum += total_loss

                    vis_path = "epoch_" + str(epoch) 
                    ber, acc = model.vis_and_metric_Detection(epoch, i+1, vis_path,  saveImg = saveImg_flag)  
                    ber_sum += ber
                    acc_sum += acc

                    if saveImg_flag:
                        model.visRemoval(epoch, i+1, vis_path)      
                    psnr_val, ssim_val = model.metricRemoval(epoch, i+1, vis_path)
                    psnr_sum += psnr_val
                    ssim_sum += ssim_val

            if opt.PretrainD:    
                tmp = ber_sum*es
                logger.info('[Eval] [Epoch] %d | detect_loss : %.5f '  % (epoch, d_loss_sum*es))
                logger.info('[Eval] [Epoch] %d | ber: %.5f, accuracy: %.5f' % (epoch, ber_sum*es, acc_sum*es))

                f = open(os.path.join(model.save_dir,  'PreEVALloss.txt'), "a")
                f.write("e: {:<3} d_loss: {:^8.4f} \n".format(epoch, d_loss_sum*ts))
                f.close

                f = open(os.path.join(model.save_dir,  'Preber.txt'), "a")
                f.write("e: {:<3} ber: {:^8.4f}  accuracy: {:^8.4f}\n".format(epoch, ber_sum*es, acc_sum*es))
                f.close

            else: 
                tmp = psnr_sum*es
                logger.info('[Eval] [Epoch] %d/%d | detect_loss : %.5f | reconst_loss: %.5f | total_loss: %.5f '  % (epoch, epochs, d_loss_sum*es, 
                                                                                                                     r_loss_sum*es, t_loss_sum*es))
                logger.info('[Eval] [Epoch] %d/%d | ber: %.5f | acc: %.5f | psnr: %.5f | ssim: %.5f '  % (epoch, epochs, ber_sum*es, acc_sum*es, 
                                                                                                          psnr_sum*es, ssim_sum*es))

                f = open(os.path.join(model.save_dir,  'evalloss.txt'), "a")
                f.write("e: {:<3} d_loss: {:^8.4f} r_loss: {:^8.4f} t_loss: {:^8.4f}\n".format(epoch, d_loss_sum*es, r_loss_sum*es, t_loss_sum*es))
                f.close
        
                f = open(os.path.join(model.save_dir,  'ber.txt'), "a")
                f.write("e: {:<3} ber: {:^8.4f}  accuracy: {:^8.4f}\n".format(epoch, ber_sum*es, acc_sum*es))
                f.close

                f = open(os.path.join(model.save_dir,  'psnr.txt'), "a")
                f.write("e: {:<3} psnr: {:^8.4f}\n".format(epoch, psnr_sum*es))
                f.close
                
                f = open(os.path.join(model.save_dir,  'ssim.txt'), "a")
                f.write("e: {:<3} ssim: {:^8.4f}\n".format(epoch, ssim_sum*es))
                f.close

        save_path = os.path.join(model.save_dir, 'checkpoints')
        if (epoch and epoch % opt.save_freq == 0 and epoch>(opt.niter + opt.niter_decay-50) and tmp >= best) or (epoch == opt.niter + opt.niter_decay):
            logger.info('saving the model at the end of epoch %d/%d, iters %d, at %s' % (epoch, epochs, total_steps, save_path))
            model.save_networks(epoch)
            best = tmp
        
        model.save_networks("latest")
        logger.info('saving the model at the end of epoch %d/%d, iters %d,  at %s latest' % (epoch, epochs, total_steps, save_path))

        spt_time = time.time() - epoch_start_time
        lft_time = (opt.niter + opt.niter_decay - epoch) * spt_time
        
        model.update_learning_rate()
        lr = model.optimizers[0].param_groups[0]['lr']
        logger.info('End of epoch %d / %d |learning rate = %.7f| Time Taken: %d sec | eta %.2f hour' %(epoch, epochs, lr, spt_time, lft_time / 3600.0))

if __name__ == '__main__':

    opt = TrainOptions().parse()
    opt.PretrainD = False   
    opt.joint_train = True 
    opt.loadSize = 256     # scale images to this size
    opt.randomSize = True 
    opt.keep_ratio = True  
    opt.HSV = True   
    opt.HSV_removalH = True   
    opt.HSV_scale = False   
    opt.gt_RGB = True   
    opt.input_RGB = True   
    opt.train = True  
    opt.mode = 'SVPC'       # SVPC
    SWeights = [2, 1.8, 1.6, 1.4, 1.2]
    opt.SWeights = SWeights 
    opt.VWeights = [0.8, 0.65, 0.5, 0.425, 0.275]

    opt.dataroot = '/your/dataset/path/of/ShiQ' 
    opt.fineSize = 256    # crop images to this size 
    opt.model = 'DF'      # 
    # opt.model = 'Fusion'
    opt.batch_size = 24   # 24 
    opt.phase = 'train_'  
    opt.gpu_ids = [0] 
    opt.lr = 0.001              
    opt.lambda_L1 = 100.0      
    opt.num_threads = 24
    opt.dataset_mode = 'expo_param' 
    opt.optmizer = 'adam'       #'adam'
    opt.n = 5                   
    opt.ks = 3                  # ks（kenerl size）
    opt.lr_policy = 'step'       
    opt.lr_decay_iters = 80
    opt.gamma = 0.5              
    opt.load_dir = 'None'       
    opt.save_epoch_freq = 100 
    opt.niter = 200         
    opt.niter_decay = 300   
    name = "{}Net_lr{}_{}_ks{}_PreD{}_Joint{}_gpu{}_{}_HSV{}_removalH_{}_gtRGB{}_input_RGB{}_500".format(opt.model, opt.lr, opt.lr_policy, opt.ks, 
                                opt.PretrainD, opt.joint_train, opt.gpu_ids[0], opt.mode, opt.HSV, opt.SWeights[0] , opt.gt_RGB, opt.input_RGB)  
    opt.name = name
    opt.checkpoints_dir = './log'  
    main(opt)