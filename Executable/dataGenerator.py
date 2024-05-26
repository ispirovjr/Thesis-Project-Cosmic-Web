import numpy as np
import vaex
import illustris_python as il
import time

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

@vaex.register_function()
def correct(x):
    L = 75000
    return np.where(np.abs(x)>L/2, x-np.sign(x)*L,x)

def localizeDataFrame(absDf, x,y,z):
    rx = absDf.X - x
    ry = absDf.Y - y
    rz = absDf.Z - z
    
    L = 75000
    
    relx = rx.correct()
    rely = ry.correct()
    relz = rz.correct()
    
    
    relR = np.sqrt(relx**2 + rely**2+relz**2)
    
    theta = np.arccos(relz/relR)
    fi = np.arctan2(rely,relx)
    
    relDf = absDf.copy()
    
    Vaway = (relx*absDf.Vx+rely*absDf.Vy+relz*absDf.Vz)/relR
    
    
    relDf["R"] = relR
    relDf["Th"] = theta
    relDf["Fi"] = fi
    
    relDf["CZ"] = relR+Vaway*10
    
    return relDf.drop("X").drop("Y").drop("Z").drop("Vx").drop("Vy").drop("Vz")
    
    
x = np.random.uniform(0,75000)
y = np.random.uniform(0,75000)
z = np.random.uniform(0,75000)


default = "/Users/users/spirov/ThesisProject/Data/Training/"

rdf = localizeDataFrame(df, x,y,z)

path = default

t = str(time.time())

rdf.export_hdf5(path+t+".hdf5")
    
