###################################################################################
"""
Step 1a: Read in IFF Data using PARA-ATM
"""
###################################################################################
from paraatm.io import iff, gnats, utils
from paraatm.io.iff import read_iff_file

iff_fname = 'IFF_SFO_ASDEX_ABC456.csv'

iff_data =  read_iff_file(iff_fname,record_types=[2,3,4,8])
callsigns = iff_data[3].callsign.unique()
cs = callsigns.tolist()

###################################################################################
"""
Step 1b: Clean callsign list of UNKN, nan, 1200, and ground operations vehicles
"""
###################################################################################
if 'UNKN' in cs: cs.remove('UNKN') # remove 'UNKN'
cs = [x for x in cs if (str(x) != 'nan')] #remove nan
cs = [x for x in cs if not x.startswith('OPS')] #remove ground operation vehicles
if '1200' in cs: cs.remove('1200')

###################################################################################
"""
Step 2: Start NATS Simulation in the background
"""
###################################################################################
from gnats_gate_to_gate import GateToGate
natsSim = GateToGate()

###################################################################################
"""
Step 3: Create FlightPlanSelector object
"""
###################################################################################
from paraatm.fpgen import FlightPlanSelector
dirPath = natsSim.DIR_share
fpath = dirPath + '/tg/trx/TRX_07132005_noduplicates_crypted'
f=FlightPlanSelector(natsSim,fname=fpath)

###################################################################################
"""
Step 4: Remove previously generated TRX and MFL files
"""
###################################################################################
#Filenames to write modified trx and mfl files to. Need full path because GNATS wrapper changes the directory to the GNATS directory.
import os

home_env = os.environ.get('HOME')
trx_dir = '/home/edecarlo/para-atm-collection/miscellaneous/gnats-fpgen'
results_dir = '/home/edecarlo/para-atm-collection/miscellaneous/gnats-fpgen'

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

###################################################################################
"""
Step 5 Generate NATS TRX and MFL files
"""
###################################################################################

import time
import numpy as np
import pandas as pd
from jpype import JPackage

from paraatm.fpgen import (get_departure_airport_from_iff,
                           get_arrival_airport_from_iff,
                           get_gate_from_iff,
                           get_rwy_from_iff,
                           check_if_flight_has_departed_from_iff,
                           check_if_flight_landing_from_iff,
                           get_gate_lat_lon_from_nats,
                           get_random_gate,
                           get_random_runway)

iffBaseAirport= "KSFO"
clsGeometry = JPackage('com').osi.util.Geometry
TRACK_TIME = time.time()

for i,callsign in enumerate(cs[:10]):
    flightInAir = check_if_flight_has_departed_from_iff(iff_data,callsign,natsSim,iffBaseAirport)
    flightLandingInData = check_if_flight_landing_from_iff(iff_data,callsign,natsSim,iffBaseAirport)
    
    arrivalAirport = ''
    departureAirport=''
    arrivalRwy = ''
    arrivalGate = ''
    departureRwy =''
    departureGate = ''
    result_generated = ''

    if flightInAir and flightLandingInData:
        #Get departure airport. If none is known in the IFF+ASDEX file (i.e., it is not the airport whose name is in the iff_fname), then set departure airport as closest airport to the first lat/lon in the dataset. In the future, would like to use IFF_USA to determine departureAirport in this case
        arrivalAirport = 'KSFO'
        departureAirport = get_departure_airport_from_iff(iff_data,callsign,natsSim,arrivalAirport=iffBaseAirport,flmap=f.flmap)
        arrivalGate= get_gate_from_iff(iff_data,callsign,natsSim,arrivalAirport,arrival=True)
        arrivalRwy= get_rwy_from_iff(iff_data,callsign,natsSim,arrivalAirport,arrival=True)
        departureGate = get_random_gate(natsSim,arrivalAirport)
        departureRwy = get_random_runway(natsSim,arrivalAirport,arrival=False)
        result_generated = f.generate(4, departureAirport, arrivalAirport, "", arrivalGate, "", arrivalRwy)
        
        timestamp = iff_data[3].loc[iff_data[3].callsign==callsign,'time'].iloc[0]
        lat,lon = list(natsSim.airportInterface.getLocation(departureAirport))
        latstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(lat))
        lonstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(lon))
        elev = np.max([iff_data[3].loc[iff_data[3].callsign==callsign,'altitude'].iloc[0]/100.,100.])
        spd = iff_data[3].loc[iff_data[3].callsign==callsign,'tas'].iloc[0]
        hdg = iff_data[3].loc[iff_data[3].callsign==callsign,'heading'].iloc[0]
        aircraftType = 'B744'
        
    if  not flightInAir and not flightLandingInData:
        departureAirport = 'KSFO'
        arrivalAirport = get_arrival_airport_from_iff(iff_data,callsign,natsSim,departureAirport,f.flmap)
        departureGate = get_gate_from_iff(iff_data,callsign,natsSim,departureAirport,arrival=False)
        departureRwy = get_rwy_from_iff(iff_data,callsign,natsSim,departureAirport,arrival=False)
        arrivalGate = get_random_gate(natsSim,arrivalAirport)
        arrivalRwy = get_random_runway(natsSim,arrivalAirport,arrival=True)
        result_generated = f.generate(1, departureAirport, arrivalAirport, departureGate, arrivalGate, departureRwy, arrivalRwy)
        
        timestamp = iff_data[3].loc[iff_data[3].callsign==callsign,'time'].iloc[0]
        airportInstance = natsSim.airportInterface.select_airport(departureAirport)
        elev = airportInstance.getElevation()/100.0  
        lat,lon = get_gate_lat_lon_from_nats(natsSim,departureGate,departureAirport)
        latstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(lat))
        lonstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(lon))
        aircraftType = 'B744'
        spd = 0
        hdg = 28
    
    if result_generated:
        #TRACK_TIME = (timestamp-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        TRACK_TIME +=10
        track_string = '%s %s %.4f %.4f %d %.2f %d %s %s' %(callsign,aircraftType,float(latstr),float(lonstr),spd,elev,hdg,'ZOA','ZOA46')
        fp_route = result_generated[0]
        fp_validated = natsSim.aircraftInterface.validate_flight_plan_record(track_string,fp_route,330)
        if fp_validated: print('Validated Flight Plan:',callsign)
        fp_validated = True
        if fp_validated:
            print(result_generated)
            with open(trx_file,'a') as trxFile:
                tstr = '%s %s' % ('TRACK',track_string)
                trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
                trxFile.write( tstr + '\n')
                trxFile.write('    FP_ROUTE ' + fp_route + '\n\n')

            with open(mfl_file,'a') as mflFile:
                mflFile.write(callsign + ' ' + '330' + '\n')
        elif not fp_validated:
            print('This flight plan was not validated:\n', result_generated[0])
                       