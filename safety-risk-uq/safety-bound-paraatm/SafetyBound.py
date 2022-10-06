import numpy as np
import random
######################################### safety bound ############################################################
class BoundAircraft:
    def __init__(self, tas,acc,dt,posAccur,hdg,vwind0,alphaw0):
        self.vuav = tas #m/s -knots
        self.a = acc #m/s2 - 
        self.tt = dt #seconds
        self.heading = hdg
        self.vwind0 = vwind0
        self.alphaw0 = alphaw0 #relative to 0
        self.posAccur = posAccur

    # sensor errors
    def winderror(self):
        self.evwind = np.random.normal(0, 0.05 * self.vwind0)
        self.ealphaw = np.random.normal(0, 0.05 * self.alphaw0)

    def uaverror(self):
        self.epGPS = np.random.uniform(-self.posAccur, self.posAccur)
        self.evuav = np.random.normal(0, 0.01 * self.vuav)
        self.ea = np.random.normal(0, 0.05 * np.abs(self.a))
        self.ett = np.random.normal(0, 0.01 * self.tt)

    ######################### safety bound size #######################################################################
    def size(self):
        self.uaverror()
        self.winderror()
        ### angle between uav and wind
        self.vecwind = ((self.vwind0 + self.evwind) * np.cos(self.alphaw0 + self.ealphaw), (self.vwind0 + self.evwind) * np.sin(self.alphaw0 + self.ealphaw))
        self.uavdir = self.heading
        self.uavvvec = self.vuav / np.linalg.norm(self.uavdir) * self.uavdir
        self.alphaw = np.arccos(np.dot(self.uavdir, self.vecwind)/(np.linalg.norm(self.uavdir)*np.linalg.norm(self.vecwind)))

        ### size
        self.valong = self.vuav + self.evuav + (self.vwind0 + self.evwind) * (np.cos(self.alphaw))
        self.vper = (self.vwind0 + self.evwind) * (np.sin(self.alphaw))

        self.lh = abs(self.valong) * (self.tt + self.ett) + (self.valong) ** 2 / 2 / (self.a + self.ea)
        self.lp = abs(self.vper) * (self.tt + self.ett) + (self.vper) ** 2 / 2 / (self.a + self.ea) + self.epGPS
        self.r = self.lh + self.lp

class BoundUAV:
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