import numpy as np

import torch

import time

def forloopdists(feats,protos):
    N, D = feats.shape
    P, D_proto = protos.shape

    distances = np.zeros((N, P))
    for i in range(N):
        for j in range(P):
          diff = feats[i, :] - protos[j, :]
          distances[i, j] = (np.sum(diff**2))

    return distances

def numpydists(feats,protos):
    feats = np.sum(feats**2, axis=1, keepdims=True)
    protos = np.sum(protos**2, axis=1, keepdims=True)

    distances = (feats - 2 * np.matmul(feats, protos.T) + protos.T )
    return distances

def pytorchdists(feats, protos, device):
    feats = torch.tensor(feats).to(device)
    protos = torch.tensor(protos).to(device)

    feats = torch.sum(feats**2, dim=1, keepdim=True)
    protos = torch.sum(protos**2, dim=1, keepdim=True)
    
    distances = (feats - 2 * torch.mm(feats, protos.t()) + protos.t() )
    return distances

def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  feats=np.random.normal(size=(250000,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))

# Using pytorch gpu
  device=torch.device('cuda:0')
  since = time.time()

  dists0=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Pytorch GPU Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists0.shape)

# Using pytorch cpu
  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Pytorch CPU Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)



# Using numpy
  since = time.time()

  dists2=numpydists(feats,protos)


  time_elapsed=float(time.time()) - float(since)

  print('Numpy Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1.numpy()-dists2)))

# Using for loops
  since = time.time()

  dists3=forloopdists(feats,protos)


  time_elapsed=float(time.time()) - float(since)

  print('For Loops Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists3.shape)


if __name__=='__main__':
  run()