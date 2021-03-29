
import numpy as np
import pandas as pd
import os
import time

from paraatm.io import iff, gnats, utils
from paraatm.io.iff import read_iff_file

from gnats_gate_to_gate import GateToGate
from iff_functions import (get_departure_airport_from_iff,
                           get_arrival_airport_from_iff,
                           get_departure_gate_and_rwy_from_iff,
                           get_arrival_gate_and_rwy_from_iff,
                           random_airport_gate_and_rwy,
                           check_if_flight_has_departed,
                           check_if_flight_landing_in_dataset)

import trx_tools as tt

from FlightPlanSelector import FlightPlanSelector

#Load IFF data and get unique callsigns
#iff_fname = 'IFF_SFO+ASDEX_20191221_080022_86363.csv'
iff_fname = 'IFF_SFO+ASDEX_20190511_080104_86221.csv'
#iff_fname = 'IFF_SFO_ASDEX_ABC123.csv'
print('Loading IFF file {} ...'.format(iff_fname))
iffBaseAirport= "KSFO"
iff_data =  read_iff_file(iff_fname,record_types=[2,3,4,8])

callsigns = iff_data[3].callsign.unique()
print('Loaded IFF file with {} callsigns...'.format(len(callsigns)))

# Initialize GNATS simulation using wrapper class. This provides access to GNATS simulation functions and is passed to several functions in this module.
gnatsSim = GateToGate()

dirPath = gnatsSim.DIR_share
fpath = dirPath + '/tg/trx/TRX_07132005_noduplicates_crypted'
f=FlightPlanSelector(gnatsSim,fname=fpath)

#Filenames to write modified trx and mfl files to. Need full path because GNATS wrapper changes the directory to the GNATS directory.
trx_dir = '/home/ghaikal/para-atm-collection/miscellaneous/gnats-fpgen/trx'
results_dir = '/home/ghaikal/para-atm-collection/miscellaneous/gnats-fpgen/results'

trx_fname = '/iff_to_gnats_geo'
mfl_file= trx_dir+trx_fname+'_mfl.trx'
trx_file = trx_dir+trx_fname+'.trx'
results_file = results_dir+trx_fname+'.csv'

# if os.path.exists(trx_file):
#     os.remove(trx_file)
#     print('Previously generated trx file removed...')

# if os.path.exists(mfl_file):
#     os.remove(mfl_file)
#     print('Previously generated mfl file removed...')

cs = callsigns.tolist()
cs.remove('UNKN')
cs = [x for x in cs if (str(x) != 'nan')]
cs = [x for x in cs if not x.startswith('OPS')]
cs.remove('1200')

aircraftType = "B733"
TRACK_TIME = time.time()

# #For each callsign perform the following
# for i,callsign in enumerate(cs[:2]):
#     flightInAir = check_if_flight_has_departed(iff_data,callsign,gnatsSim,iffBaseAirport)
#     flightLandingInData = check_if_flight_landing_in_dataset(iff_data,callsign,gnatsSim,iffBaseAirport)
    
#     arrivalAirport = ''
#     departureAirport=''
#     arrivalRwy = ''
#     arrivalGate = ''
#     departureRwy =''
#     departureGate = ''

#     if flightInAir and flightLandingInData:
#         #Get departure airport. If none is known in the IFF+ASDEX file (i.e., it is not the airport whose name is in the iff_fname), then set departure airport as closest airport to the first lat/lon in the dataset. In the future, would like to use IFF_USA to determine departureAirport in this case
#         print (callsign, i)
#         arrivalAirport = 'KSFO'
#         departureAirport = get_departure_airport_from_iff(iff_data,callsign,gnatsSim,arrivalAirport=iffBaseAirport,flmap=f.flmap)
#         arrivalRwy,arrivalGate= get_arrival_gate_and_rwy_from_iff(iff_data,callsign,gnatsSim,arrivalAirport)
#         result_generated4 = f.generate(4, departureAirport, arrivalAirport, "", arrivalGate, "", arrivalRwy)
#         print(result_generated4[0])
#         print(result_generated4[1])
#         print(result_generated4[2])

#         TRACK_TIME += 10
#         with open(trx_file,'a') as trxFile:
#             tstr = '%s %s %s %.4f %.4f %d %.2f %d %s %s' % ('TRACK',callsign,aircraftType,float(result_generated4[1]),float(result_generated4[2]),200,100,28,'ZOA','ZOA46')
#             trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
#             trxFile.write( tstr + '\n')
#             trxFile.write('    FP_ROUTE ' + result_generated4[0] + '\n\n')

#         with open(mfl_file,'a') as mflFile:
#             mflFile.write(callsign + ' ' + '330' + '\n')

#     if  not flightInAir and not flightLandingInData:
#         print (callsign, i)
#         departureAirport = 'KSFO'
#         arrivalAirport = get_arrival_airport_from_iff(iff_data,callsign,gnatsSim,departureAirport,f.flmap)
#         departureRwy,departureGate = get_departure_gate_and_rwy_from_iff(iff_data,callsign,gnatsSim,departureAirport) #doesn't exist
#         arrivalRwy,arrivalGate = random_airport_gate_and_rwy(gnatsSim,arrivalAirport)
#         result_generated1 = f.generate(1, departureAirport, arrivalAirport, departureGate, arrivalGate, departureRwy, arrivalRwy)
#         print(result_generated1[0])
#     #     #fp_route=tt.write_trx_geo(iff_data,callsign,departureAirport,arrivalAirport,gnatsSim,trx_file,mfl_file)

#         airportInstance = gnatsSim.airportInterface.select_airport(departureAirport)
#         elev = airportInstance.getElevation()/100.0   

#         TRACK_TIME += 10
#         with open(trx_file,'a') as trxFile:
#             tstr = '%s %s %s %.4f %.4f %d %.2f %d %s %s' % ('TRACK',callsign,aircraftType,float(result_generated1[1]),float(result_generated1[2]),0,elev,28,'ZOA','ZOA46')
#             trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
#             trxFile.write( tstr + '\n')
#             trxFile.write('    FP_ROUTE ' + result_generated1[0] + '\n\n')

#         with open(mfl_file,'a') as mflFile:
#             mflFile.write(callsign + ' ' + '330' + '\n')

gnatsSim.setupAircraft(trx_file=trx_file,mfl_file=mfl_file)
gnatsSim.simulation()
gnatsSim.write_output(results_file)
gnatsSim.cleanup()

