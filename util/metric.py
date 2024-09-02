import numpy as np
import math
import cv2
import torch

# from skimage.measure import compare_ssim
#rom skimage.metrics import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim

def ber_accuracy(y_actual, y_hat):
    y_hat = y_hat.ge(128).float()
    y_actual = y_actual.ge(128).float()

    y_actual = y_actual.squeeze(1)
    y_hat = y_hat.squeeze(1)

    #output==1
    pred_p=y_hat.eq(1).float()
    #print(pred_p)
    #output==0
    pred_n = y_hat.eq(0).float()
    #print(pred_n)
    #TP
    pre_positive = float(pred_p.sum())
    pre_negtive = float(pred_n.sum())

    # FN
    fn_mat = torch.gt(y_actual, pred_p)
    FN = float(fn_mat.sum())

    # FP
    fp_mat = torch.gt(pred_p, y_actual)
    FP = float(fp_mat.sum())

    TP = pre_positive - FP
    TN = pre_negtive - FN

    #print(TP,TN,FP,FN)
    #tot=TP+TN+FP+FN
    #print(tot)
    pos = TP+FN
    neg = TN+FP

    #print(pos,neg)

    #print(TP/pos)
    #print(TN/neg)
    if(pos!=0 and neg!=0):
        BAC = (.5 * ((TP / pos) + (TN / neg)))
    elif(neg==0):
        BAC = (.5*(TP/pos))
    elif(pos==0):
        BAC = (.5 * (TN / neg))
    else:
        BAC = .5
    # print('tp:%d tn:%d fp:%d fn:%d' % (TP, TN, FP, FN))
    accuracy = (TP+TN)/(pos+neg)*100
    BER=(1-BAC)*100
    return BER, accuracy




def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps
    return 20*math.log10(255.0/rmse)


def ssim(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)

    # convert the images to grayscale BGR2GRAY
    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # (grayScore, diff) = compare_ssim(grayA, grayB, full=True)
    # diff = (diff * 255).astype("uint8")
    # print("gray SSIM: {}".format(grayScore))

    (score0, diffB) = compare_ssim(B1, B2, full=True)
    (score1, diffG) = compare_ssim(G1, G2, full=True)
    (score2, diffR) = compare_ssim(R1, R2, full=True)
    aveScore = (score0 + score1 + score2) / 3
    # print("BGR average SSIM: {}".format(aveScore))

    return aveScore

'''
image1 = cv2.imread('outcome/D/14001_D.png')
image2 = cv2.imread('outcome/D_pred/14001_D.png_pred.png')

psnrvalue = psnr(image1,image2)
# print(psnrvalue)

grayScore, aveScore = ssim(image1,image2)
# print(grayScore)
# print(aveScore)
'''



#
# path1 = 'outcome/D'
# paths1 = [os.path.join(path1, i)
#                    for i in os.listdir(path1)]
# paths1.sort()
#
# path2 = 'outcome/D_pred'
# paths2 = [os.path.join(path2, i)
#                    for i in os.listdir(path2)]
# paths2.sort()
#
# Names = os.listdir(path1)
# Names.sort()
#
#
# if len(paths1) != len(paths2):
#     raise Exception("numbers of images don't match")
#
# psnr_total = 0
# grayScore_total = 0
# aveScore_total = 0
#
# with open("Metric_record","w+",encoding= "gbk") as f:  #clear all the before contents first
#  for i in range(0 , int(len(paths1)) ):
#     image1 = cv2.imread(paths1[i])
#     image2 = cv2.imread(paths2[i])
#
#     psnr_temp = psnr(image1,image2)
#     psnr_total += psnr_temp
#
#     grayScore_temp,aveScore_temp = ssim(image1,image2)
#     grayScore_total += grayScore_temp
#     aveScore_total += aveScore_temp
#     text = [ Names[i], "  psnr: ",str(psnr_temp), "  ssim1: ",str(grayScore_temp), "  ssim2: ",str(aveScore_temp),"\n" ]
#     f.writelines(text)
#
# psnr_mean = psnr_total / len(paths1)
# grayScore_mean = grayScore_total / len(paths1)
# aveScore_mean = aveScore_total / len(paths1)
# with open("Metric_record","a+",encoding= "gbk") as f:
#   f.write("Mean psnr: {}".format(psnr_mean))
#   f.write("Mean ssim: {}(method1), {}(method2) ".format(grayScore_mean,aveScore_mean))
# f.close()
#
# print("The mean psnr is :{}".format(psnr_mean))
# print("The mean ssim is: {}(method1), {}(method2) ".format(grayScore_mean,aveScore_mean))
#
# print('END!!!!')