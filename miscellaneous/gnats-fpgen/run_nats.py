
import numpy as np
import pandas as pd
import time
import os
from gnats_gate_to_gate import GateToGate

# Initialize GNATS simulation using wrapper class. This provides access to GNATS simulation functions and is passed to several functions in this module.
natsSim = GateToGate()

#Filenames to write modified trx and mfl files to. Need full path because GNATS wrapper changes the directory to the GNATS directory.
home_env = os.environ.get('HOME')
trx_dir = home_env+'/para-atm-collection/miscellaneous/gnats-fpgen'
results_dir = home_env+'/para-atm-collection/miscellaneous/gnats-fpgen'

trx_fname = '/iff_to_gnats_geo_SFO_test' # /iff_to_gnats_geo, iff_to_gnats_geo_SFO, SFO_mod
mfl_file= trx_dir+trx_fname+'_mfl.trx'
trx_file = trx_dir+trx_fname+'.trx'
results_file = results_dir+trx_fname+'.csv'

simResults = natsSim.simulation(trx_file,mfl_file)
print(simResults.columns)
#data=natsSim(trx_name=trx_file,mfl_name=mfl_file)['trajectory']
natsSim.write_output(results_file)
natsSim.cleanup()

with open(trx_file, 'r') as rFile:
    rLines = rFile.readlines()
    
trxCallsigns = [line.split(' ')[1] for line in rLines if line.split(' ')[0]=='TRACK']
trxLatitudes = [line.split(' ')[3] for line in rLines if line.split(' ')[0]=='TRACK']
trxLongitudes = [line.split(' ')[4] for line in rLines if line.split(' ')[0]=='TRACK']

print(trxCallsigns)
print(trxLatitudes)
print(trxLongitudes)

for no, cs in enumerate(trxCallsigns[0]):
    simData = data.loc[data.callsign==cs]
    print(simData)
    simLatitudes = simData['latitude']
    simLongitudes = simData['longitude']
    print(simData.columns)
    
    trxLat, trxLon = trxLatitudes[no], trxLongitudes[no]
    print(trxLat, trxLon)
    print(simLatitudes, simLongitudes)
    1/0