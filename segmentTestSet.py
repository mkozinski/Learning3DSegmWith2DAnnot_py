import sys
sys.path.append("../")
import numpy as np
import torch
from net_v1 import UNet3d
import os
from NetworkTraining_py.forwardOnBigImages import processChunk

datadir="../MRAdata_py"
exec(open("../MRAdata_py/testFiles.txt_uncut").read())
exec(open("../MRAdata_py/trainFiles.txt_uncut").read())

log_dir="log_3projections_v1"
net = UNet3d().cuda()
saved_net=torch.load(os.path.join(log_dir,"net_last.pth")) #
net.load_state_dict(saved_net['state_dict'])
net.eval();

out_dir="output_last_test_v1" 

def process_output(o):
    e=np.exp(o[0,1,:,:,:])
    prob=e/(e+1)
    return prob
  
outdir=os.path.join(log_dir,out_dir)
os.makedirs(outdir)

for f in testFiles: 
  img=np.load(os.path.join(datadir,f[0])).astype(np.float32)
  inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
  oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
  prob=process_output(oup)
  np.save(os.path.join(outdir,os.path.basename(f[0])),prob)
