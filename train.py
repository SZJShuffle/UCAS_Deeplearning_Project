import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
import argparse

from pre_processing import *
from encoding import *

import time
from sklearn.metrics import roc_curve, auc, roc_auc_score



if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')

class CNN(nn.Module):
    def __init__(self, nb_filter = 16, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 107, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1   
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1

        print ('out1_size',out1_size)  #98  (107+0-9-1)/1 + 1
        print ('maxpool_size',maxpool_size) #96   ((98+0-2)-1)/1 + 1

        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = (maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool2_size = (out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.drop1 = nn.Dropout(p=0.25)
        print ('out2_size',out2_size)   # 87
        print ('maxpool2_size',maxpool2_size) # 85         

        print ('maxpool2_size * nb_filter', maxpool2_size* nb_filter)  ##1360 (16*85)

        self.fc1 = nn.Linear(int(maxpool2_size*nb_filter), hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

class Estimator(object):
    def __init__(self,model,loader,optimizer,loss_func):
        ##clf = Estimator(cnn,train_loader,optimizer,loss_func)
        self.model = model   
        self.loader = loader 
        self.optimizer = optimizer
        self.loss_func = loss_func

    def fit(self,protein,nb_epoch = 1):
        for epoch in range(nb_epoch): 
            print ('epoch %s' % epoch)
            for step,(batch_x,batch_y) in enumerate(self.loader): 

                if cuda:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                output = self.model(batch_x)
                loss = self.loss_func(output,batch_y)

                global loss_history
                loss_history.append(loss.data.cpu().numpy())

                self.optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                self.optimizer.step()        # apply gradients
                
                if step %50 == 0:
                    print('-------------- start predicting -----------------')
                    global test_bags
                    global train_bags
                    #test_bags = torch.from_numpy(np.array(test_bags).astype(np.float32))
                    if cuda:
                        test_bags = test_bags.cuda()
                        pred = self.predict_prob(test_bags)

                        train_bags = train_bags.cuda()
                        trainset_pred = self.predict_prob(train_bags)
                    else:
                        pred = self.predict_prob(test_bags)                     
                        trainset_pred = self.predict_prob(train_bags)                      
                    print('-------------- predicting end -----------------') 
                    
                    acc = self.accuracy(protein,pred)           
                    auc = self.roc(protein,pred)  

                    trainset_acc = self.accuracy(protein,trainset_pred,trainset = True)           
                    trainset_auc = self.roc(protein,trainset_pred,trainset = True)  

                    print('Epoch: ', epoch)
                    print('trainset_accuracy %.4f ' % trainset_acc)
                    print('trainset_auc %.4f ' % trainset_auc)
                    print('testset_accuracy %.4f' %acc)
                    print('testset_auc %.4f' %auc)

    def predict_prob(self,X):
        self.model.eval()
        predict = self.model(X)
        #predict = predict.data.numpy()
        
        if cuda:
            predict = torch.max(predict,1)[1].cuda().data
        else:
            predict = torch.max(predict,1)[1].data.numpy()
        return predict

    def accuracy(self,protein,pred_y,trainset = False):
        if trainset:
            true_y = get_all_data(protein)[1]
        else:
            true_y = get_all_data(protein)[3]

        if cuda:
          #true_y = torch.from_numpy(np.array(true_y).astype(np.float32))
          #true_y = true_y.cuda().data
            pred_y = pred_y.cpu().data.numpy()
            acc = float((pred_y == true_y.astype(int)).sum())/float(true_y.size)
        else:
            acc = float((pred_y == true_y.astype(int)).sum())/float(true_y.size)

        return acc

    def roc(self,protein,pred_y,trainset = False):
        if trainset:
            true_y = get_all_data(protein)[1]
        else:
            true_y = get_all_data(protein)[3]
        if cuda:
            pred_y = pred_y.cpu().data.numpy()
            auc = roc_auc_score(true_y,pred_y)
        else:
            auc = roc_auc_score(true_y,pred_y)          
        return auc

loss_history=[]

def get_all_data(protein, channel = 7):
    data = load_graphprot_data(protein)
    test_data = load_graphprot_data(protein, train = False)
    #pdb.set_trace()
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data)
        test_bags, true_y = get_bag_data_1_channel(test_data)
    else:
        train_bags, label = get_bag_data(data)
        test_bags, true_y = get_bag_data(test_data)

    return train_bags, label, test_bags, true_y


# training and predict ---------------------------------------------------

def globalCNN(protein,nb_epoch):
    global test_bags
    global train_bags
    train_bags, train_labels,test_bags, true_y = get_all_data(protein, channel = 1)
    train_set = TensorDataset(torch.from_numpy(np.array(train_bags).astype(np.float32)),torch.from_numpy(np.array(train_labels).astype(np.float32)).long().view(-1))
    train_loader = Data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    
    test_bags = torch.from_numpy(np.array(test_bags).astype(np.float32))
    train_bags = torch.from_numpy(np.array(train_bags).astype(np.float32))
    
    cnn = CNN(channel = 1,labcounts = 4,window_size = 501+6)
    if cuda:
      cnn.cuda()
    print("--------globalCNN network architecture--------")
    print(cnn)
    
    
    '''
    ##reload parameters
    para_file = './'+protein+'_param/'+'globalCNN_'+protein+'.pkl'
    param = torch.load(para_file)
    cnn.load_state_dict(param)  
    '''


    ##optimizer
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=0.2)   # optimize all cnn parameters
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=0.2,momentum = 0.8)
    #optimizer= torch.optim.RMSprop(cnn.parameters(), lr=0.2,alpha = 0.9)
    #optimizer= torch.optim.Adam(cnn.parameters(), lr=0.2,betas= (0.9,0.99))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay = 0.001)
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    clf = Estimator(cnn,train_loader,optimizer,loss_func)
    clf.fit(protein,nb_epoch)       

    global_trainset_pred = clf.predict_prob(train_bags)
    global_testset_pred = clf.predict_prob(test_bags)
    
    ##save parameters
    isExist = os.path.exists('./'+protein+'_param')
    if not isExist:
        os.mkdir('./'+protein+'_param')
    para_file = './'+protein+'_param/'+'globalCNN_'+protein+'.pkl'
    torch.save(cnn.state_dict(),para_file)
    
    return global_trainset_pred,global_testset_pred
    '''
    ##predict 
    print('-------------- global CNN start predicting -----------------')
    test_bags = torch.from_numpy(np.array(test_bags).astype(np.float32))
    global_pred = clf.predict_prob(test_bags)
    print('-------------- global CNN predicting end -----------------')
    return global_pred
    '''

def localCNN(protein,nb_epoch):
    global test_bags 
    global train_bags
    train_bags, train_labels,test_bags, true_y  = get_all_data(protein,channel = 7)
    ##np.array(train_bags),np.array(train_labels)
    train_set = TensorDataset(torch.from_numpy(np.array(train_bags).astype(np.float32)),torch.from_numpy(np.array(train_labels).astype(np.float32)).long().view(-1))
    train_loader = Data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    
    test_bags = torch.from_numpy(np.array(test_bags).astype(np.float32))
    train_bags = torch.from_numpy(np.array(train_bags).astype(np.float32))
    
    cnn = CNN()
    if cuda:
      cnn.cuda()
    print("--------localCNN network architecture--------")
    print(cnn)
    
    '''
    #reload parameters
    para_file = './'+protein+'_param/'+'localCNN_'+protein+'.pkl'
    param = torch.load(para_file)
    cnn.load_state_dict(param)
    '''
    
    ##optimizer
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=0.2)   # optimize all cnn parameters
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=0.2,momentum = 0.8)
    #optimizer= torch.optim.RMSprop(cnn.parameters(), lr=0.2,alpha = 0.9)
    #optimizer= torch.optim.Adam(cnn.parameters(), lr=0.2,betas= (0.9,0.99))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay = 0.0001)
    loss_func = nn.CrossEntropyLoss()                     

    clf = Estimator(cnn,train_loader,optimizer,loss_func)
    clf.fit(protein,nb_epoch)

    local_trainset_pred = clf.predict_prob(train_bags)
    local_testset_pred = clf.predict_prob(test_bags)
    
    ##save parameters
    isExist = os.path.exists('./'+protein+'_param')
    if not isExist:
        os.mkdir('./'+protein+'_param')
    para_file = './'+protein+'_param/'+'localCNN_'+protein+'.pkl'
    torch.save(cnn.state_dict(),para_file)    
    
    return local_trainset_pred,local_testset_pred
    '''
    ##predict 
    print('-------------- local CNN start predicting -----------------')
    test_bags = torch.from_numpy(np.array(test_bags).astype(np.float32))
    local_pred = clf.predict_prob(test_bags)
    print('-------------- local CNN predicting end -----------------')
    return local_pred
    '''

def ensemble(protein,nb_epoch):
    global_trainset_pred,global_testset_pred = globalCNN(protein,nb_epoch)
    local_trainset_pred,local_testset_pred = localCNN(protein,nb_epoch)

    ensemble_trainset_pred = (global_trainset_pred + local_trainset_pred) / 2.0
    ensemble_testset_pred = (global_testset_pred + local_testset_pred) / 2.0

    if cuda:
        ensemble_trainset_pred  = ensemble_trainset_pred.cpu().numpy()
        ensemble_testset_pred = ensemble_testset_pred.cpu().numpy()

    train_bags, train_labels,test_bags, true_y = get_all_data(protein, channel = 1)
    print('-------------start evaluation of  ensemble method-----------')
    trainset_eval = ensemble_eval(train_labels,ensemble_trainset_pred)
    testset_eval = ensemble_eval(true_y,ensemble_testset_pred)
    print('-------------end evaluation of  ensemble method-----------')
    return trainset_eval,testset_eval

def ensemble_eval(true_y,pred_y):
    auc = roc_auc_score(true_y,pred_y)
    acc = float((pred_y == true_y.astype(int)).sum())/float(true_y.size)
    return auc,acc



def main(net,protein,nb_epoch):
      if net == 'ensemble':
          ## Not yet finished
          ensemble_pred = ensemble(protein,nb_epoch)
          print('-----------ensemble evaluation-------------')
          print('ensemble auc in trainingset: %.3f' % ensemble_pred[0][0])
          print('ensemble accuracy in trainingset: %.3f' % ensemble_pred[0][1])    
          print('ensemble auc in testset: %.3f' % ensemble_pred[1][0])
          print('ensemble accuracy in testset: %.3f' % ensemble_pred[1][1])
          
          return ensemble_pred
      if net == 'globalCNN':
          global_pred = globalCNN(protein,nb_epoch)
          return global_pred
      if net == 'localCNN':
          local_pred = localCNN(protein,nb_epoch)
          return local_pred     

def parser_args(parser):
    parser.add_argument('-model_type', type=str, default='ensemble', help='it supports the following deep network models:globalCNN,localCNN and ensemble, default=ensemble')    
    parser.add_argument('-protein',type=str,default = 'ALKBH5', help= 'input the protein you want to train ,default=ALKBH5')
    parser.add_argument('-epoch',type=int,default = '1', help= 'input the epoch you want to train ,default=1')

def run(args):
    model_type = args.model_type
    protein = args.protein
    nb_epoch = args.epoch
    start = time.process_time()
    
    main(model_type,protein,nb_epoch)
    print('-------------loss_history-----------')
    print(loss_history)
        
    print('--------------done------------------')   
    run_time = (time.process_time() - start)
    print("Time used:",run_time)


if __name__ == '__main__':
    ##QKI/IGF2BP1-3/ELAVL1H

    parser = argparse.ArgumentParser()
    parser_args(parser)
    args = parser.parse_args()
    print('----args-----')
    print(args)
    run(args)




