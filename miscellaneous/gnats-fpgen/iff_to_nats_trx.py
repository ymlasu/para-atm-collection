
import numpy as np
import pandas as pd
import os

from paraatm.io import iff, gnats, utils
from paraatm.io.iff import read_iff_file

from gnats_gate_to_gate import GateToGate
from iff_functions import (get_departure_airport_from_iff,
                           get_arrival_airport_from_iff,
                           check_if_flight_has_departed,
                           create_gate_to_runway_from_iff)

import trx_tools as tt

from FlightPlanSelector import FlightPlanSelector

#Load IFF data and get unique callsigns
#iff_fname = 'IFF_SFO+ASDEX_20191221_080022_86363.csv'
#iff_fname = 'IFF_SFO+ASDEX_20190511_080104_86221.csv'
iff_fname = 'IFF_SFO_ASDEX_ABC123.csv'
print('Loading IFF file {} ...'.format(iff_fname))

iff_data =  read_iff_file(iff_fname,record_types=[2,3,4,8])

callsigns = iff_data[3].callsign.unique()
print('Loaded IFF file with {} callsigns...'.format(len(callsigns)))

# Initialize GNATS simulation using wrapper class. This provides access to GNATS simulation functions and is passed to several functions in this module.
gnatsSim = GateToGate()

dirPath = gnatsSim.DIR_share
fpath = dirPath + '/tg/trx/TRX_07132005_noduplicates_crypted'
f=FlightPlanSelector(gnatsSim,fname=fpath)

#Filenames to write modified trx and mfl files to. Need full path because GNATS wrapper changes the directory to the GNATS directory.
trx_dir = '/home/edecarlo/para-atm-collection/miscellaneous/gnats-fpgen/trx'
results_dir = '/home/edecarlo/para-atm-collection/miscellaneous/gnats-fpgen/results'

trx_fname = '/iff_to_gnats_geo'
mfl_file= trx_dir+trx_fname+'_mfl.trx'
trx_file = trx_dir+trx_fname+'.trx'
results_file = results_dir+trx_fname+'.csv'

if os.path.exists(trx_file):
    os.remove(trx_file)
    print('Previously generated trx file removed...')

if os.path.exists(mfl_file):
    os.remove(mfl_file)
    print('Previously generated mfl file removed...')

#For each callsign perform the following
for i,callsign in enumerate(callsigns):
    print (callsign, i)
    departureAirport, arrivalAirport = "", ""

    if callsign != 'UNKN':
        #Get departure airport. If none is known in the IFF+ASDEX file (i.e., it is not the airport whose name is in the iff_fname), then set departure airport as closest airport to the first lat/lon in the dataset. In the future, would like to use IFF_USA to determine departureAirport in this case
        departureAirport = get_departure_airport_from_iff(iff_data,callsign,gnatsSim)

        if departureAirport[-3:] is 'SFO':
            #departureRwy = get_departure_runway_from_iff(iff_data,callsign,gnatsSim) # doesn't exist
            #departureGate = get_departure_gate_from_iff(iff_data,callsign,gnatsSim) # doesn't exist
            print('callsign:',callsign,'departure airport:',departureAirport)
    
            #Get arrival airport. If none is known in the IFF+ASDEX file, currently getting closest airport (that is not departure airport) to the final lat/lon in the dataset. In the future, would like to use IFF_USA to determine arrivalAirport
            arrivalAirport = get_arrival_airport_from_iff(iff_data,callsign,gnatsSim,departureAirport,f.flmap)
            #arrivalRwy = get_arrival_runway_from_iff(iff_data,callsign,gnatsSim) #doesn't exist
            #arrivalGate = get_arrival_gate_from_iff(iff_data,callsign,gnatsSim) #doesn't exist
            result_generated3 = f.generate(3, departureAirport, arrivalAirport, "", "", "", "");

        #result_generated1 = f.generate(1, departureAirport, arrivalAirport, departureGate, arrivalGate, departureRwy, arrivalRwy);
        #fp_route=tt.write_trx_geo(iff_data,callsign,departureAirport,arrivalAirport,gnatsSim,trx_file,mfl_file)

# gnatsSim.setupAircraft(trx_file=trx_file,mfl_file=mfl_file)
# # gnatsSim.simulation()
# # gnatsSim.write_output(results_file)
# gnatsSim.cleanup()

