import os, cv2
import numpy as np
import torch

def BER(y_actual, y_hat):
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

if __name__ == "__main__":
    # get img file in a list
    '''
    img_list = os.listdir(pre_path)
    print(img_list)
    average_ber = 0.0
    average_accuracy = 0.0
    sum_ber = 0.0
    sum_accuracy = 0.0
    difficult = []
    for i,name in enumerate(img_list):
        if name.endswith('.png'):
            predict = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            label = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            #print(label)
            score, accuracy = BER(torch.from_numpy(label).float(), torch.from_numpy(predict).float())
            sum_ber = sum_ber + score
            average_ber = sum_ber/(i+1)
            sum_accuracy = sum_accuracy + accuracy
            average_accuracy = sum_accuracy/(i+1)
            print("name:%s , ber:%f, average_ber:%f, accuracy:%f, average_accuracy=%f" % (name, score,average_ber,accuracy,average_accuracy))
            if accuracy<96:
                difficult.append((name,accuracy))

    print(difficult)
    '''

    pre_path = './output_335/pre/14936.png'

    label_path ='./output_335/gt/14936.png'
    average_ber = 0.0
    average_accuracy = 0.0
    sum_ber = 0.0
    sum_accuracy = 0.0
    difficult = []
    # predict = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
    # print(predict)
    # label = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
    # print(label)
    predict = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    score, accuracy = BER(torch.from_numpy(label).float(), torch.from_numpy(predict).float())
    sum_ber = sum_ber + score
    average_ber = sum_ber / (0 + 1)
    sum_accuracy = sum_accuracy + accuracy
    average_accuracy = sum_accuracy / (0 + 1)
    # print("name:%s , ber:%f, average_ber:%f, accuracy:%f, average_accuracy=%f" % (
    # name, score, average_ber, accuracy, average_accuracy))
    if accuracy < 96:
        # difficult.append((name, accuracy))
        print(accuracy)