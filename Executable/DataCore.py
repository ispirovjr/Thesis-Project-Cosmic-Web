import vaex
import numpy as np

snapshotPath ="/Users/users/spirov/ThesisProject/Snapshots/modelSnapshot.pt"

L = 75000
DataSizeLimit = 111992#37000
h = 0.704

@vaex.register_function()
def correct(x):
    return np.where(np.abs(x) > L / 2, x - np.sign(x) * L, x)


def localizeDataFrame(absDf, x, y, z):
    rx = absDf.X - x
    ry = absDf.Y - y
    rz = absDf.Z - z

    relx = rx.correct()
    rely = ry.correct()
    relz = rz.correct()

    relR = np.sqrt(relx ** 2 + rely ** 2 + relz ** 2)

    theta = np.arccos(relz / relR)
    fi = np.arctan2(rely, relx)

    relDf = absDf.copy()

    Vaway = (relx * absDf.Vx + rely * absDf.Vy + relz * absDf.Vz) / relR

    relR *=h
    
    relDf["R"] = relR
    relDf["Th"] = theta
    relDf["Fi"] = fi

    CZ = relR + Vaway * 10/h

    cz = CZ#.correct()

    relDf["CZ"] = cz

    #relDf = relDf[relR<L/2]

    return relDf.drop("X").drop("Y").drop("Z").drop("Vx").drop("Vy").drop("Vz")[:DataSizeLimit]

