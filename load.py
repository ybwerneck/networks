import numpy as np
import matplotlib.pyplot as plt

##mostra que ele cria uma cond inicial novamente para cada janela
import vtk
import matplotlib.pyplot as plt
import scipy  

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os


def read_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput().GetPointData()
   # print(data)
    field_count = data.GetNumberOfArrays()
    return {data.GetArrayName(i): vtk_to_numpy(data.GetArray(i)) for i in range(field_count)}

def r(path):
    
    data=read_vtp(path)
    #print(data["t"])
    return data["t"],data["x1"],data["w"],data["K"]

    
def find(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

def f(T,d):
    for dado in d:
        T.append(dado)
def load(cells):
    from os import listdir
    from os.path import isfile, join
    mypath="."
    onlyfiles = [f for f in listdir(mypath)]
    p=lambda s:'outputs/fhn1P/window'+str(s)+'/inferencers/inferencer.vtp'
    #plot('outputs/fhn2eqv2/initial_conditions/validators/validator.vtp',True)


    t,x,w,k= r('outputs/fhn1P/initial_conditions/inferencers/inferencer.vtp')
    n_w=1
 
    
    n=np.shape(x)[0]
    T=[]#np.zeros(n*n_w)
    X=[]#np.zeros(n*n_w)
    W=[]#np.zeros(n*n_w)
    K=[]
    
    f(T,t)
    f(X,x)
    f(W,w)
    f(K,k)
    #T[0:n]=t
    #X[0:n]=x
    #W[0:n]=w

    for i in range(1,n_w):
        #print(i)
        d=r(p(i))

       # T[n*i:n*(i+1)]=d[0]+i
       # X[n*i:n*(i+1)]=d[1]
       # W[n*i:n*(i+1)]=d[2]
        f(T,d[0]+i)
        f(X,d[1])
        f(W,d[2])
        f(K,d[3])
    cells = list(sorted(set(K)))
    n_cell=len(cells)
    print("ncell",n_cell)
    n=int(len(T)/n_cell)
    print("len",len(X))
    U=np.zeros((int(n_cell)*int(n)))
    o=0
    for cell in cells:
        indexes = [i for i in range(len(K)) if K[i] == cell]
        U[o*n: (o+1)*n]=[X[i] for i in indexes]
        o=o+1
    
    e=lambda x:np.expand_dims(x,axis=1)

    invar={"t":e(np.array(T).T )   }
    out={"x1":e(np.array(X).T), "w":e(np.array(W).T)      }
    return U,np.array(list(sorted((T))))


