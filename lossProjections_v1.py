import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def visualHullFilter(lbl1,lbl2,lbl3):
  h=torch.full((lbl1.size(0),lbl2.size(1),lbl1.size(1),lbl2.size(2)),1,dtype=lbl1.dtype)
  h[lbl1.reshape(lbl1.size(0),1,lbl1.size(1),lbl1.size(2)).expand_as(h)==0]=0
  h[lbl2.reshape(lbl1.size(0),lbl2.size(1),1,lbl2.size(2)).expand_as(h)==0]=0
  h[lbl3.reshape(lbl1.size(0),lbl2.size(1),lbl1.size(1),1).expand_as(h)==0]=0
  m1,_=torch.max(h,1)
  m2,_=torch.max(h,2)
  m3,_=torch.max(h,3)
  l1=lbl1.clone()
  l1[m1==0]=0
  l2=lbl2.clone()
  l2[m2==0]=0
  l3=lbl3.clone()
  l3[m3==0]=0
  return l1,l2,l3

class loss_projections_v1(nn.Module):

  def __init__(self,ignoredIndex=255):
    super(loss_projections_v1,self).__init__()
    self.ce1=nn.CrossEntropyLoss(ignore_index=ignoredIndex)
    self.ce2=nn.CrossEntropyLoss(ignore_index=ignoredIndex)
    self.ce3=nn.CrossEntropyLoss(ignore_index=ignoredIndex)

  def forward(self, logits, targets):
    # 0- batch, 1- channel, 2- height, 3- width, 4- depth
    projection1,_ = logits.max(2)
    projection1=projection1.reshape(logits.size(0),logits.size(1),
                                    logits.size(3),logits.size(4))
    projection2,_ = logits.max(3)
    projection2=projection2.reshape(logits.size(0),logits.size(1),
                                    logits.size(2),logits.size(4))
    projection3,_ = logits.max(4)
    projection3=projection3.reshape(logits.size(0),logits.size(1),
                                    logits.size(2),logits.size(3))
    ## the visual hull filtering  makes sense 
    ## if annotations are not consistent across views 
    ## and contain the ignored class around positive centerline annotations
    #l1,l2,l3=visualHullFilter(targets[0], targets[1], targets[2])
    l1,l2,l3=targets[0],targets[1],targets[2]
    loss1=self.ce1(projection1,l1) 
    loss2=self.ce2(projection2,l2) 
    loss3=self.ce3(projection3,l3) 
    return loss1+loss2+loss3
           


