import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paraatm.io.iff import read_iff_file
data = read_iff_file('IFF_SFO_ASDEX_ABC123.csv')

from SafetyBound import BoundAircraft

lhs = []
lps = []
rs = []
for i in range(len(data)-1):

    dt = pd.Timedelta(data.time[i+1]-data.time[i]).total_seconds()
    tas = data.tas[i+1]
    acc = (tas-data.tas[i])/dt
    hdg = data.heading[i+1]
    posAccur = data.coord1Accur[i+1]
    vwind0 = 0.1
    alphaw0 = 0.1

    b = BoundAircraft(tas,acc,dt,posAccur,hdg,vwind0,alphaw0)
    b.size()
    lh = b.lh
    lp = b.lp
    r = b.r

    lhs.append(lh)
    lps.append(lp)
    rs.append(r)