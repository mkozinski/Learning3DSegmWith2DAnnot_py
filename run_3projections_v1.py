import sys
sys.path.append("../")
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from net_v1 import UNet3d
from NetworkTraining_py.loggerBasic import LoggerBasic
from NetworkTraining_py.loggerF1 import LoggerF1
from NetworkTraining_py.loggerComposit import LoggerComposit
from NetworkTraining_py.dataset import Dataset
from NetworkTraining_py.datasetCrops import TestDataset
import os
import os.path
import torch.nn.functional as F
from shutil import copyfile
from NetworkTraining_py.trainer import trainer
from NetworkTraining_py.tester import tester
from random import randint, random
import math
from lossProjections_v1 import loss_projections_v1

def augmentTrain(img,lbl,cropSz,augmentParam):

  # random crop 
  maxstartind1=img.shape[0]-cropSz[0]
  maxstartind2=img.shape[1]-cropSz[1]
  maxstartind3=img.shape[2]-cropSz[2]
  startind1=randint(0,maxstartind1)
  startind2=randint(0,maxstartind2)
  startind3=randint(0,maxstartind3)
  img =img[startind1:startind1+cropSz[0],
           startind2:startind2+cropSz[1],
           startind3:startind3+cropSz[2]]
  lbl =[lbl[0][startind2:startind2+cropSz[1],
               startind3:startind3+cropSz[2]],
        lbl[1][startind1:startind1+cropSz[0],
               startind3:startind3+cropSz[2]],
        lbl[2][startind1:startind1+cropSz[0],
               startind2:startind2+cropSz[1]],
       ]

  # flip
  if random()>0.5 :
    img=np.flip(img,0)
    lbl[1]=np.flip(lbl[1],0)
    lbl[2]=np.flip(lbl[2],0)
  if random()>0.5 :
    img=np.flip(img,1)
    lbl[0]=np.flip(lbl[0],0)
    lbl[2]=np.flip(lbl[2],1)
  if random()>0.5 :
    img=np.flip(img,2)
    lbl[0]=np.flip(lbl[0],1)
    lbl[1]=np.flip(lbl[1],1)

  it=torch.from_numpy(np.copy(img)).unsqueeze(0)
  #lt=torch.from_numpy(np.stack([lbl[0].astype(np.long),
  #                              lbl[1].astype(np.long),
  #                              lbl[2].astype(np.long)],axis=0))
  lt=[torch.from_numpy(lbl[0].astype(np.long)),
      torch.from_numpy(lbl[1].astype(np.long)),
      torch.from_numpy(lbl[2].astype(np.long))]

  return it, lt

def preproc(img,lbl):
  # load data to gpu in the training loop
  return img.cuda(), [lbl[0].cuda(),lbl[1].cuda(),lbl[2].cuda()]

def augmentTest(img,lbl):
  lbl=torch.from_numpy(lbl)
  img=torch.from_numpy(img).unsqueeze(0)
  return img,lbl

def preprocTest(img,lbl):
  # load data to gpu for test
  return img.cuda(), lbl.cuda()

def test_preproc(o,t):
  # preprocessing for test logger
  idx=torch.Tensor.mul_(t<=1, t>=0).reshape(t.numel())
  o=o[:,1,:,:,:]
  oo=o.reshape(t.numel())[idx]
  tt=t.reshape(t.numel())[idx]
  o=torch.pow(torch.exp(-oo)+1,-1)
  return o,tt

datadir="../MRAdata_py/"
exec(open("../MRAdata_py/trainFiles.txt").read())
exec(open("../MRAdata_py/testFiles.txt").read())
trainimgs=[]
trainlbls=[]
for f in trainFiles:
  img =np.load(os.path.join(datadir,f[0]))
  lbl =[np.load(os.path.join(datadir,f[3][0])).astype(np.uint8),
        np.load(os.path.join(datadir,f[3][1])).astype(np.uint8),
        np.load(os.path.join(datadir,f[3][2])).astype(np.uint8)]
  trainimgs.append(img.astype(np.float32))
  trainlbls.append(lbl)
testimgs=[]
testlbls=[]
for f in testFiles:
  img   =np.load(os.path.join(datadir,f[0]))
  lbl   =np.load(os.path.join(datadir,f[2]))
  testimgs.append(img.astype(np.float32))
  testlbls.append(lbl-1)

log_dir="log_3projections_v1"
os.makedirs(log_dir)
copyfile(__file__,os.path.join(log_dir,"setup.txt"))

net = UNet3d().cuda()
#prev_log_dir="log_baseline_v1"
#saved_net=torch.load(os.path.join(prev_log_dir,"net_last.pth"))
#net.load_state_dict(saved_net['state_dict'])
net.train()

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)

logger= LoggerBasic(log_dir,"Basic",100)

train_dataset=Dataset(trainimgs,trainlbls,
                 lambda i,l: augmentTrain(i,l,np.array([64,64,64]),0))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                  shuffle=True, num_workers=6, drop_last=True)

loss = loss_projections_v1().cuda() 

test_dataset = TestDataset(testimgs,testlbls,np.array([80,80,80]),
                  [22,22,22], augmentTest)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                  shuffle=False, num_workers=1, drop_last=False)

logger_test=LoggerF1(log_dir,"Test",test_preproc, nBins=10000, saveBest=True)
tstr=tester(test_loader,logger_test,preprocTest)

lr_lambda=lambda e: 1/(1+e*1e-5)
lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
trn=trainer(net, train_loader, optimizer, loss, logger, tstr, 100,
            lr_scheduler=lr_scheduler,preprocImgLbl=preproc)

if __name__ == '__main__':
  print(log_dir)
  trn.train(100000)
