import numpy as np
import random
######################################### safety bound ############################################################
class Bound:
    def __init__(self, cfg, vwind0, alphaw0):
        self.vuav = cfg['vuav']
        self.a = cfg['a']
        self.tt = cfg['tt']
        self.ori = cfg['ori']
        self.des = cfg['des']
        self.vwind0 = vwind0
        self.alphaw0 = alphaw0

    # sensor errors
    def winderror(self):
        self.evwind = np.random.normal(0, 0.05 * self.vwind0)
        self.ealphaw = np.random.normal(0, 0.05 * self.alphaw0)

    def uaverror(self):
        self.epGPS = np.random.uniform(-1.5, 1.5)
        self.evuav = np.random.normal(0, 0.05 * self.vuav)
        self.ea = np.random.normal(0, 0.05 * self.a)
        self.ett = np.random.normal(0, 0.05 * self.tt)

    ######################### safety bound size #######################################################################
    def size(self):
        self.uaverror()
        self.winderror()
        ### angle between uav and wind
        self.vecwind = ((self.vwind0 + self.evwind) * np.cos(self.alphaw0 + self.ealphaw), (self.vwind0 + self.evwind) * np.sin(self.alphaw0 + self.ealphaw))
        self.uavdir = np.subtract(self.des, self.ori)
        self.uavvvec = self.vuav / np.linalg.norm(self.uavdir) * self.uavdir
        self.alphaw = np.arccos(np.dot(self.uavdir, self.vecwind)/(np.linalg.norm(self.uavdir)*np.linalg.norm(self.vecwind)))

        ### size
        self.valong = self.vuav + self.evuav + (self.vwind0 + self.evwind) * (np.cos(self.alphaw))
        self.vper = (self.vwind0 + self.evwind) * (np.sin(self.alphaw))

        self.lh = abs(self.valong) * (self.tt + self.ett) + (self.valong) ** 2 / 2 / (self.a + self.ea)
        self.lp = abs(self.vper) * (self.tt + self.ett) + (self.vper) ** 2 / 2 / (self.a + self.ea) + self.epGPS
        self.r = self.lh + self.lp

cfg = {'vuav': 16,
        'a': 4.95,
        'tt': 1,
        'ori': (0, 0),
        'des': (10, 0)} # up: (0, 10) down (0, -10) left (-10, 0) right (10, 0)

np.random.seed(0)
random.seed(0)
vwind0 = 4
alphaw0 = (np.pi) / 2.0
N = 1000
lh = np.zeros(N)
lp = np.zeros(N)
b = Bound(cfg, vwind0, alphaw0)

for i in range(N):
    b.size()
    lh[i] = b.lh
    lp[i] = b.lp
print(np.sort(lh)[950], np.sort(lp)[950])
