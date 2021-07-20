###################################################################################
"""
Step 1a: Read in IFF Data using PARA-ATM
"""
###################################################################################
from paraatm.io import iff, gnats, utils
from paraatm.io.iff import read_iff_file
import pandas as pd

iff_fname = 'IFF_SFO+ASDEX_20190511_080104_86221.csv'

iff_data =  read_iff_file(iff_fname,record_types=[2,3,4,8])

###################################################################################
"""
Step 1b: Select IFF data within the specified timeframe
"""
###################################################################################
time_interval = 1200

start_time = iff_data[2]['time'].iloc[0] 
end_time =  pd.to_datetime(start_time.value*10**(-9) + time_interval, unit='s')

iff_data[2] = iff_data[2].loc[iff_data[2].time <= end_time,:]
iff_data[3] = iff_data[3].loc[iff_data[3].time <= end_time,:]
iff_data[4] = iff_data[4].loc[iff_data[4].time <= end_time,:]
iff_data[8] = iff_data[8].loc[iff_data[8].time <= end_time,:]

###################################################################################
"""
Step 1c:    Select callsign list with both recType=2 and recType=3 entries
            Clean callsign list of UNKN, nan, 1200, and ground operations vehicles
"""
###################################################################################

# retrieve callsigns with recType=3 entries
callsigns_iff3 = iff_data[3].callsign.unique()
callsigns_iff3 = callsigns_iff3.tolist()

# clean entries list
if 'UNKN' in callsigns_iff3: callsigns_iff3.remove('UNKN') # remove 'UNKN'
callsigns_iff3 = [x for x in callsigns_iff3 if (str(x) != 'nan')] #remove nan
callsigns_iff3 = [x for x in callsigns_iff3 if not x.startswith('OPS')] #remove ground operation vehicles
if '1200' in callsigns_iff3: callsigns_iff3.remove('1200')

# cross match callsigns with recType=2 and recType=3 entries 
flightDataLocs =[i for i in range(len(iff_data[2])) if (iff_data[2].iloc[i].callsign in callsigns_iff3)]
flightData = iff_data[2].iloc[flightDataLocs]

cs = flightData.callsign.tolist()

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
import os
import paraatm
from paraatm.fpgen import FlightPlanSelector

dirParaatm = os.path.dirname(paraatm.__file__)
fname = dirParaatm + '/fpgen/TRX_07132005_noduplicates_crypted_SFO_clean'
f=FlightPlanSelector(natsSim,fname=fname)

###################################################################################
"""
Step 4: Remove previously generated TRX and MFL files
"""
###################################################################################
#Filenames to write modified trx and mfl files to. Need full path because GNATS wrapper changes the directory to the GNATS directory.
import os

home_env = os.environ.get('HOME')
trx_dir = '/home/ghaikal/para-atm-collection/miscellaneous/gnats-fpgen'
results_dir = '/home/ghaikal/para-atm-collection/miscellaneous/gnats-fpgen'

trx_fname = '/iff_to_gnats_geo_SFO_test3'
mfl_file= trx_dir+trx_fname+'_mfl.trx'
trx_file = trx_dir+trx_fname+'.trx'
coord_file = trx_dir+trx_fname+'.crd'
results_file = results_dir+trx_fname+'.csv'

if os.path.exists(trx_file):
    os.remove(trx_file)
    print('Previously generated trx file removed...')

if os.path.exists(mfl_file):
    os.remove(mfl_file)
    print('Previously generated mfl file removed...')

if os.path.exists(coord_file):
    os.remove(coord_file)
    print('Previously generated coordinate file removed...')

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
#TRACK_TIME = time.time()

for i,callsign in enumerate(cs):
    
    arrivalAirport = ''
    departureAirport=''
    arrivalRwy = ''
    arrivalGate = ''
    departureRwy =''
    departureGate = ''
    result_generated = ''

    bcnCode, opsType = flightData.iloc[i].bcnCode, flightData.iloc[i].opsType
    cs_iff_data = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode)]
    recordLength = len(cs_iff_data)

    if recordLength==0: continue

    flightDepartureInData = check_if_flight_has_departed_from_iff(iff_data,callsign,bcnCode,natsSim,iffBaseAirport)
    flightLandingInData = check_if_flight_landing_from_iff(iff_data,callsign,bcnCode,natsSim,iffBaseAirport)

    if opsType=='A' and flightLandingInData:

        timestamp = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'time'].iloc[0]
        lat = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'latitude'].iloc[0]
        lon = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'longitude'].iloc[0] 

        #Get departure airport. If none is known in the IFF+ASDEX file (i.e., it is not the airport whose name is in the iff_fname), then set departure airport as closest airport to the first lat/lon in the dataset. In the future, would like to use IFF_USA to determine departureAirport in this case
        arrivalAirport = 'KSFO'
        departureAirport = get_departure_airport_from_iff(iff_data,callsign,bcnCode,lat,lon,natsSim,arrivalAirport=iffBaseAirport,flmap=f.flmap)
        arrivalGate= get_gate_from_iff(iff_data,callsign,bcnCode,natsSim,arrivalAirport,arrival=True)
        arrivalRwy= get_rwy_from_iff(iff_data,callsign,bcnCode,natsSim,arrivalAirport,arrival=True)
        departureGate = get_random_gate(natsSim,arrivalAirport)
        departureRwy = get_random_runway(natsSim,arrivalAirport,arrival=False)
        result_generated = f.generate(4, departureAirport, arrivalAirport, "", arrivalGate, "", arrivalRwy)
        
        timestamp = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'time'].iloc[0]
        lat = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'latitude'].iloc[0]
        lon = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'longitude'].iloc[0]        
        simLat,simLon = list(natsSim.airportInterface.getLocation(departureAirport))

        latstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(simLat))
        lonstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(simLon))
        elev = np.max([iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'altitude'].iloc[0]/100.,100.])
        spd = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'tas'].iloc[0]
        hdg = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'heading'].iloc[0]
        aircraftType = 'B744'

    # if  opsType=='D' and flightDepartureInData:
    #     departureAirport = 'KSFO'
    #     arrivalAirport = get_arrival_airport_from_iff(iff_data,callsign,bcnCode,natsSim,departureAirport,f.flmap)     
    #     departureGate = get_gate_from_iff(iff_data,callsign,bcnCode,natsSim,departureAirport,arrival=False)
    #     departureRwy = get_rwy_from_iff(iff_data,callsign,bcnCode,natsSim,departureAirport,arrival=False)
    #     arrivalGate = get_random_gate(natsSim,arrivalAirport)
    #     arrivalRwy = get_random_runway(natsSim,arrivalAirport,arrival=True)
    #     result_generated = f.generate(5, departureAirport, arrivalAirport, departureGate, "", departureRwy, "")
        
    #     timestamp = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'time'].iloc[0]
    #     airportInstance = natsSim.airportInterface.select_airport(departureAirport)
    #     elev = airportInstance.getElevation()/100.0  
    #     lat = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'latitude'].iloc[0]
    #     lon = iff_data[3].loc[(iff_data[3].callsign==callsign) & (iff_data[3].bcnCode==bcnCode),'longitude'].iloc[0]
    #     #lat,lon = get_gate_lat_lon_from_nats(natsSim,departureGate,departureAirport)
    #     latstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(lat))
    #     lonstr = clsGeometry.convertLatLonDeg_to_degMinSecString(str(lon))
    #     aircraftType = 'B744'
    #     spd = 0
    #     hdg = 28

    if not result_generated:
        print("Flight " + callsign + " result not generated")
    else:
        TRACK_TIME = (timestamp-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        #TRACK_TIME +=10
        track_string = '%s %s %.4f %.4f %d %.2f %d %s %s' %(callsign,aircraftType,float(latstr),float(lonstr),spd,elev,hdg,'ZOA','ZOA46')
        fp_route = result_generated[0]
        
        fp_validated = True #natsSim.aircraftInterface.validate_flight_plan_record(track_string,fp_route,330)

        if fp_validated:

            print('Validated Flight Plan:',callsign)
            with open(trx_file,'a') as trxFile:
                tstr = '%s %s' % ('TRACK',track_string)
                trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
                trxFile.write( tstr + '\n')
                trxFile.write('    FP_ROUTE ' + fp_route + '\n\n')

            with open(mfl_file,'a') as mflFile:
                mflFile.write(callsign + ' ' + '330' + '\n')

            with open(coord_file,'a') as coordFile:
                coordFile.write(callsign + ' ' + str(TRACK_TIME) + ' ' + str(lat) + ' ' + str(lon) + '\n')

        elif not fp_validated:
            print('This flight plan was not validated:\n', result_generated[0])

#list(natsSim.terminalAreaInterface.getAllStars('KLAX'))                       