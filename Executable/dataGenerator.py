import numpy as np
import vaex
import illustris_python as il
import time
from DataCore import localizeDataFrame


Training = False
n = 256

path = "/net/virgo01/data/users/spirov/Nexus Project/"
subhalos = il.groupcat.load(path,135)['subhalos']

subHaloPoses = subhalos['SubhaloCM']
subHaloHalfRads = subhalos["SubhaloHalfmassRad"]
subhaloVels = subhalos["SubhaloVel"]

Xs = subHaloPoses[:,0]
Ys = subHaloPoses[:,1]
Zs = subHaloPoses[:,2]
Vx = subhaloVels[:,0]
Vy = subhaloVels[:,1]
Vz = subhaloVels[:,2]

df = vaex.from_arrays(X=Xs,Y=Ys,Z=Zs,Vx=Vx,Vy=Vy,Vz=Vz)


default = "/Users/users/spirov/ThesisProject/Data/Training/"
path = default
if not Training:
    path = "/Users/users/spirov/ThesisProject/Data/Testing/"


for i in range(n):
    x = np.random.uniform(0,75000)
    y = np.random.uniform(0,75000)
    z = np.random.uniform(0,75000)

    rdf = localizeDataFrame(df, x,y,z)
    t = str(time.time())
    rdf.export_hdf5(path+t+".hdf5")