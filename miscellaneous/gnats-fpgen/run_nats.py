
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

trx_fname = '/iff_to_gnats_geo'
mfl_file= trx_dir+trx_fname+'_mfl.trx'
trx_file = trx_dir+trx_fname+'.trx'
results_file = results_dir+trx_fname+'.csv'

natsSim.simulation(trx_file,mfl_file)
natsSim.write_output(results_file)
natsSim.cleanup()

