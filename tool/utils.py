#DataLoader
import torch
from datasets import SketchData
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import precision_score,average_precision_score,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
attribute_Dict = {"hair": ["w/ H", "w/o H"], "gender": ["male", "female"],
                      "earring": ["w/ E", "w/o E"], "smile": ["w/ S", "w/o S"],
                      "frontal_face": ["<=30 D", ">30 D"], }#"style": ["Style1", "Style2", "Style3"]
def Mask_GT_Preds(outputs,gts):#为了获取真实标签的置信度
    #outputs:[attributes_num,batch_size,value_dim], gts:[attributes_num,batch_size]
    gts = gts.long()
    Mask = torch.zeros(outputs.shape)
    if len(outputs.size())>2:#多属性
        for i in range(outputs.size(0)):#按属性
            for j in range(outputs.size(1)):#按batch_size
                Mask[i,j,gts[i,j]] = 1
    else:#单属性
        for i in range(outputs.size(0)):#按batch_size
            Mask[i,gts[i]] =1
    return outputs*Mask

def Load_SketchData(train_data=True,test_data=True,batch_size=16,shuffle=True):#输入torch.utils.data.Dataset类训练与测试数据集，输出torch.utils.data.DataLoader类训练与测试数据批次数据
    if train_data:
        train_data = SketchData('../data/train_anno.csv','../data/train',transform=torchvision.transforms.CenterCrop(size=250))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    if test_data:
        test_data = SketchData('../data/test_anno.csv','../data/test',transform=torchvision.transforms.CenterCrop(size=250))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    if train_data and test_data:
        return train_dataloader, test_dataloader
    elif train_data:
        return train_dataloader
    else:
        return test_dataloader
def Precision(preds,labels):#preds:[batch_size,value_dim], labels[batch_size]
    preds,labels = preds.cpu(),labels.cpu()
    preds = preds.max(dim=-1)[1]#按最后一维求最大值，其中result[0]为最大值的Tensor，result[1]为最大值对应的index的Tensor。
    #preds.shape:(batch_size)
    return precision_score(labels,preds,average=None)
def Average_Precision(preds,labels):#preds:[batch_size,value_dim], labels[batch_size]
    preds, labels = preds.cpu(), labels.cpu()
    preds = preds.max(dim=-1)[1].type_as(labels)# 按最后一维求最大值，其中result[0]为最大值的Tensor，result[1]为最大值对应的index的Tensor。
    # preds.shape:(batch_size)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)#等价于precision_score(labels,preds,average="micro")
def mAP(outputs,labels):#这个方法是求Average_Precision_score 不适合Style属性这种多分类
    #outputs:[attributes_num,batch_size,value_dim], labels:[attributes_num,batch_size]
    outputs = Mask_GT_Preds(outputs,labels)
    outputs = outputs.max(dim=-1)[0]#按最后一维求最大值，其中result[0]为最大值的Tensor，result[1]为最大值对应的index的Tensor。
    scores = outputs.cpu().detach().numpy()#scores:[attributes_num,batch_size]
    gts = labels.cpu().detach().numpy()#gts:[attributes_num,batch_size]
    #根据真实标签生成掩码矩阵Mask_Matrix * outputs -> 在求max(dim)[0]即可获得真实标签置信度
    AP = []
    for i in range(gts.shape[0]):#按属性计算AP
        AP.append(average_precision_score(gts[i],scores[i]))
    mAP = np.mean(AP)
    res = {}
    res['mAP'] = mAP
    res['AP'] ={}
    for name, val in zip(list(attribute_Dict.keys()),AP[:-1]):
        res['AP'][name] = val
    return res
def Confusion_Matrix_Plot(outputs,labels,style_output,style_label,epoch):#outputs:[attributes_num,batch_size,value_dim], labels:[attributes_num,batch_size]
    outputs = outputs.max(dim=-1)[1]  # 按最后一维求最大值，其中result[0]为最大值的Tensor，result[1]为最大值对应的index的Tensor。
    preds = outputs.cpu().detach().numpy()  # preds:[attributes_num,batch_size]
    gts = labels.cpu().detach().numpy()  # gts:[attributes_num,batch_size]
    style_pred = style_output.max(dim=-1)[1].cpu().detach().numpy()
    style_gt = style_label.cpu().detach().numpy()
    for i in range(len(attribute_Dict)):#按属性计算混淆矩阵
        pred,gt = preds[i],gts[i]#获取 预测值与真实标签
        attr = list(attribute_Dict.keys())[i]#获取 属性名
        attribute_selection = attribute_Dict[attr]#获取 取值含义
        CM = confusion_matrix(gt,pred)#normalize ='all'全部， 'true' for rows, 'predicted' for columns
        plt.matshow(CM, cmap=plt.cm.YlGn)#Reds
        plt.colorbar()#显示颜色条
        for i in range(len(CM)):
            for j in range(len(CM)):
                plt.annotate(CM[i, j], xy=(i, j), horizontalalignment='center',
                             verticalalignment='center')
                #plt.annotate("{:.4f}".format(CM[i,j]), xy=(i,j), horizontalalignment='center', verticalalignment='center')
        tick_marks = np.arange(len(attribute_selection))
        plt.xticks(tick_marks,attribute_selection)#rotation=25
        plt.yticks(tick_marks,attribute_selection,rotation=90)
        plt.tick_params(labelsize=13)#设置字体
        plt.ylabel('True Label', fontdict={'family': 'Times New Roman', 'size': 15})
        plt.xlabel('Predicted Label', fontdict={'family': 'Times New Roman', 'size': 15})
        # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
        # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
        title = 'Confusion Matrix for {}'.format(attr.replace('_',' '))
        plt.title(title,fontdict={'family': 'Times New Roman', 'size': 18})
        plt.savefig('../tmp/' + str(epoch) + '_' + attr + '.png')#需先保存图片再show()
        plt.show()

    #style属性单独拎出来
    CM = confusion_matrix(style_gt, style_pred, normalize='all')  # labels=attribute_selection
    attribute_selection = ['S1','S2','S3']
    plt.matshow(CM, cmap=plt.cm.YlGn)  # Reds
    plt.colorbar()  # 显示颜色条
    for i in range(len(CM)):
        for j in range(len(CM)):
            plt.annotate(('{:.4f}'.format(CM[i, j])), xy=(i, j), horizontalalignment='center', verticalalignment='center')
    tick_marks = np.arange(len(attribute_selection))
    plt.xticks(tick_marks, attribute_selection)  # rotation=25
    plt.yticks(tick_marks, attribute_selection, rotation=90)
    plt.tick_params(labelsize=13)  # 设置字体
    plt.ylabel('True Label', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.xlabel('Predicted Label', fontdict={'family': 'Times New Roman', 'size': 15})
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    title = 'Confusion Matrix for Style'
    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 18})
    plt.savefig('../tmp/' + str(epoch) + '_style.png')
    plt.show()

'''
style_preds = torch.rand((16,3))
style_labels = torch.cat((torch.ones((1,5)),torch.zeros((1,5)),torch.ones((1,6))+torch.ones((1,6))),dim=1)
preds = torch.rand((5,16,2))
labels = torch.cat((torch.ones((5,8)),torch.zeros((5,8))),dim=1)
Confusion_Matrix_Plot(preds,labels,style_preds,style_labels)
res = mAP(preds,labels)
'''