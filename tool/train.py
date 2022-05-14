import time
import argparse
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import os
from utils import Load_SketchData,Precision,Average_Precision,Confusion_Matrix_Plot
import sys
sys.path.append('../')
from model.ResNet import Load_Model_Resnet18



log_fp = open('../tmp/train_log.txt','w',encoding='utf8')

attribute_Dict = {"hair": ["w/ H", "w/o H"], "gender": ["male", "female"],
                      "earring": ["w/ E", "w/o E"], "smile": ["w/ S", "w/o S"],
                      "frontal_face": ["<=30 D", ">30 D"],"style": ["Style1", "Style2", "Style3"]}

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--pre_trained', action='store_true', default=True,
                    help='if use pre_trained model.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch_size.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--optim', type=str, default='Adam',
                    help='Select optimizer type[Adam,SGD,RMSprop].')
parser.add_argument('--model', type=str, default='CNN',
                    help='Select model type.')

args = parser.parse_args()
device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

#Load Data
train_dataloader,test_dataloader = Load_SketchData(batch_size=args.batch_size)#[1058,3,250,250] [1058,3,250,250]
#train_features, train_labels = next(iter(train_dataloader))
#train_features.shape:(64,3,250,250)(B,C,H,W) train_labels.shape:(6,64)(attribute_num,Batch_size)
#Load Model


model = Load_Model_Resnet18(pretrained=args.pre_trained)
''''''
#Load Optimizer
if args.optim == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == 'SGD':
    optimizer = optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
else:#即'RMSprop'
    optimizer = optim.RMSprop(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
#Load Loss_Function
loss_fn = nn.CrossEntropyLoss()
model = model.to(device)
loss_fn = loss_fn.to(device)

def train_loop(dataloader, model, loss_fn, optimizer,epoch):
    train_time = time.time()
    model.train()
    optimizer.zero_grad()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):#y.shape:(16,6)
        X,y = X.to(device),y.T.to(device)
        # Compute prediction and loss
        train_batch_time = time.time()
        if args.pre_trained:
            preds,style_pred = model(model,X)#preds.shape:(5,16,2) styple_pred.shape:(16,3)
        else:
            preds, style_pred = model(X)
        for i in range(preds.size(0)):
            if i==0:
                loss = loss_fn(preds[i], y[i])
            else:
                loss += loss_fn(preds[i], y[i])
        loss += loss_fn(style_pred,y[5])
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if batch % 100 == 0:
        if batch:
            loss, current = loss.item(), batch * args.batch_size
            print(f"Batch:{batch:>2d} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time:{time.time() - train_batch_time:.4f}")
    print(f" epoch:{epoch} train time: {time.time()-train_time:.4f}")
    log_fp.write(f"Epoch:{epoch} train time: {time.time()-train_time:.4f}\t")


def test_loop(dataloader, model, loss_fn,epoch):
    test_time = time.time()
    with torch.no_grad():
        for batch,(X, y) in enumerate(dataloader):#y.shape:(6,64)
            X, y = X.to(device), y.T.to(device)
            if args.pre_trained:
                preds, style_pred = model(model, X)  # preds.shape:(5,16,2) styple_pred.shape:(16,3)
            else:
                preds, style_pred = model(X)
            W_Preds = preds if batch==0 else torch.cat((W_Preds,preds),dim=1)
            WS_Preds = style_pred if batch==0 else torch.cat((WS_Preds,style_pred),dim=0)
            W_Y = y if batch==0 else torch.cat((W_Y,y),dim=1)
        #W_Preds.shape:(5,1058,2),WS_Preds.shape:(1058,3),W_Y.shape:(6,1058)
        for i in range(W_Preds.size(0)):
            if i==0:
                test_loss = loss_fn(W_Preds[i], W_Y[i])
            else:
                test_loss += loss_fn(W_Preds[i], W_Y[i])
        test_loss += loss_fn(WS_Preds,W_Y[5])
        avg_loss = test_loss.item()/batch
        P_list= []
        for i in range(W_Preds.size(0)):
            P_list.append(list(Precision(W_Preds[i],W_Y[i])))
        P_list.append(list(Precision(WS_Preds, W_Y[5])))
        AP_list = []
        for i in range(W_Preds.size(0)):
            AP_list.append(Average_Precision(W_Preds[i],W_Y[i]).item())
        AP_list.append(Average_Precision(WS_Preds,W_Y[5]).item())
    Test_res = "Epoch{}: \n P:".format(epoch) +str(P_list) +"\nAP:"+str(AP_list) +"\n loss: {:>8f}, time: {:.4f}s\n".format(avg_loss,time.time() - test_time)
    log_test = "P:".format(epoch) +str(P_list) +"\nAP:"+str(AP_list) +"\n loss: {:>8f}, time: {:.4f}s\n".format(avg_loss,time.time() - test_time)
    print(Test_res)
    log_fp.write(log_test)
    if epoch and (epoch%100==0 or epoch==args.epochs):#不为0且是100的倍数或已经到最后一个epochs
        Confusion_Matrix_Plot(W_Preds, W_Y[:5], WS_Preds, W_Y[5], epoch)
        torch.save(model,'../tmp/model_param/'+str(epoch)+'.pth')


for t in range(args.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer,t+1)
    test_loop(test_dataloader, model, loss_fn,t+1)
print("Done!")
log_fp.close()