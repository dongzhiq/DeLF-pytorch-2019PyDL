import pickle
import torch
import numpy as np
from scipy.cluster.vq import vq,kmeans2
from sklearn.cluster.k_means_ import k_means
from dataset import configdataset
import os, sys, time
sys.path.append('../')
sys.path.append('../train')
sys.path.append('../helper')
from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from feeder import Feeder
import matcher

from evaluate import compute_map
def calEuclideanDistance(vec1,vec2):  
    dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))  
    return dist 
K=65536
print(K)
with open('ox5kdelf-full', 'rb') as file:
         b=pickle.load(file)
with open('ox5kdelfquery-full', 'rb') as file:
         a=pickle.load(file)
print(a[0]['filename'],b[30]['filename'],b[69]['filename'],b[75]['filename'])
c=[]
print(type(b[0]['descriptor_np_list'][0]))
for i in range(len(b)):
    for j in range(len(b[i]['descriptor_np_list'])):
        c.append(b[i]['descriptor_np_list'][j])
c=np.array(c)
codewords, _ ,_ ,_= k_means(c, K,max_iter=20,return_n_iter=True)
code=[]
query=[]
'''
i=0
gd=np.zeros((K,40), dtype=np.float32)
for j in range(len(a[i]['descriptor_np_list'])):
    x=a[i]['descriptor_np_list'][j].reshape(1,40)
    tmp,_=vq(x,codewords)
    
    gd[tmp]+=x-codewords[tmp]
gd0=gd.reshape(1,-1)
print(gd0)

i=30
gd=np.zeros((K,40), dtype=np.float32)
for j in range(len(b[i]['descriptor_np_list'])):
    x=b[i]['descriptor_np_list'][j].reshape(1,40)
    tmp,_=vq(x,codewords)
    
    gd[tmp]+=x-codewords[tmp]
gd1=gd.reshape(1,-1)
print(gd1)

i=69
gd=np.zeros((K,40), dtype=np.float32)
for j in range(len(b[i]['descriptor_np_list'])):
    x=b[i]['descriptor_np_list'][j].reshape(1,40)
    tmp,_=vq(x,codewords)
    
    gd[tmp]+=x-codewords[tmp]
gd2=gd.reshape(1,-1)
print(gd2)

i=75
gd=np.zeros((K,40), dtype=np.float32)
for j in range(len(b[i]['descriptor_np_list'])):
    x=b[i]['descriptor_np_list'][j].reshape(1,40)
    tmp,_=vq(x,codewords)
    
    gd[tmp]+=x-codewords[tmp]
gd3=gd.reshape(1,-1)
dist0 = np.linalg.norm(gd0 - gd1) 
dist1 = np.linalg.norm(gd0 - gd2) 
dist2 = np.linalg.norm(gd1 - gd2) 
dist3 = np.linalg.norm(gd0 - gd3)
print(dist0,dist1,dist2,dist3)
'''
for i in range(len(a)):
    print(i)
    gd=np.zeros((K,40), dtype=np.float32)
    for j in range(len(a[i]['descriptor_np_list'])):
        x=a[i]['descriptor_np_list'][j].reshape(1,40)
        tmp,_=vq(x,codewords)
        
        gd[tmp]+=x-codewords[tmp]
    gd=gd.reshape(1,-1)
    print(gd.shape)
    if i==0:
       query=gd
       print(111)
    else:
       
       query=np.concatenate((query,gd),axis=0)
query=np.array(query)
query =query - np.mean(query,axis=0) 
query /= np.std(query, axis=0) 


for i in range(len(b)):
    print(i)
    gd=np.zeros((K,40), dtype=np.float32)
    if len(b[i]['descriptor_np_list'])==0:
       gd=gd.reshape(1,-1)
       gd[0]=1
       print(gd,i)
    else:
        for j in range(len(b[i]['descriptor_np_list'])):
            
            x=b[i]['descriptor_np_list'][j].reshape(1,40)
            tmp,_=vq(x,codewords)
            
            gd[tmp]+=x-codewords[tmp]
        gd=gd.reshape(1,-1)
    
    if i==0:
       code=gd
       print(111)
    else:
       
       code=np.concatenate((code,gd),axis=0)
code =np.array(code)
code -= np.mean(code, axis=0)
code /= np.std(code, axis=0) 
print(code.shape)
sim = np.dot(code, query.T)
ranks = np.argsort(-sim, axis=0)
dataset='roxford5k'
INPUT_PATH = '/home/yangyc/revisitop-master/data/datasets/'
cfg = configdataset(dataset,INPUT_PATH)

gnd = cfg['gnd']

# evaluate ranks
ks = [1, 5, 10]

# search for easy
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])

    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

# search for easy & hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

# search for hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

