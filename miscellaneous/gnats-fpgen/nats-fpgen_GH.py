
from GNATS_Python_Header_standalone import *

from FlightPlanSelector import FlightPlanSelector

clsGeometry = JPackage('com').osi.util.Geometry;

dirPath = '../GNATS_Server/share/tg/trx/'
fpath = dirPath + 'TRX_07132005_noduplicates_crypted'

f = FlightPlanSelector(fpath);

#orgAirportList = ["PHX","MDW","LAS","BOS","MEM","SEA","SLC","STL","LAX","SAN","ORD"]
#destAirportList = ["MCO","MEM","DEN","IAH","PIT","ORD","JFK","IAD","ATL"]

orgAirportList = ["KBOS"] #["KPHX","KMDW","KLAS","KBOS","KMEM","KSEA","KSLC","KSTL","KLAX","KSAN","KORD"]
destAirportList = ["KPHX"] #["KMCO","KMEM","KDEN","KIAH","KPIT","KORD","KJFK","KIAD","KATL"]
aircraftType = "B733"
isOrgSFO = [False, False]

airlines=['SWA','DAL','AAL','UAL']
import random
import os

trxFilename = dirPath + '/TRX_' + "KBOS" + '_' + "KSFO" + '_GateToGate.trx'
if os.path.exists(trxFilename): 
    os.remove (trxFilename)

mflFilename = dirPath + '/TRX_' + "KBOS" + '_' + "KSFO" + '_mfl.trx'
if os.path.exists(mflFilename): 
    os.remove (mflFilename)

TRACK_TIME = time.time()
counter = 1000  

for i in range(1):

    orgSFO = random.choice(isOrgSFO)  
    orgAirport = "KSFO" if orgSFO else random.choice(orgAirportList)
    destAirport = "KSFO" if not(orgSFO) else random.choice(destAirportList)
    print(orgAirport, destAirport)

    airportInstance = airportInterface.select_airport(orgAirport)
    elev = airportInstance.getElevation()/100.0   

    orgGates = airportInterface.getAllGates(orgAirport)
    destGates = airportInterface.getAllGates(destAirport)

    myOrgGate = random.choice(orgGates)
    myDestGate = random.choice(destGates)

    orgRwys = list(airportInterface.getAllRunways(orgAirport))
    destRwys = list(airportInterface.getAllRunways(destAirport))

    orgRwys = [rwy[0] for rwy in orgRwys]
    destRwys = [rwy[0] for rwy in destRwys]

    myOrgRwy = "RW15R" #random.choice(orgRwys)#[:4]
    myDestRwy = "RW28R" #random.choice(destRwys)[:4]
    print(myOrgRwy,myDestRwy)

    if not orgSFO:
        counter +=1
        flightID = str(random.choice(airlines)) + str(int(counter))

        print("Flight type 1", flightID);
        #try:
        result_generated1 = f.generate(1, orgAirport, destAirport, myOrgGate, myDestGate, myOrgRwy, myDestRwy);
        print("Result flight plan = ", result_generated1[0]);
        print("Starting point latitude/longitude = ", result_generated1[1], ",", result_generated1[2]);

        TRACK_TIME += 10
        with open(trxFilename,'a') as trxFile:
            tstr = '%s %s %s %.4f %.4f %d %.2f %d %s %s' % ('TRACK',flightID,aircraftType,float(result_generated1[1]),float(result_generated1[2]),0,elev,28,'ZOA','ZOA46')
            trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
            trxFile.write( tstr + '\n')
            trxFile.write('    FP_ROUTE ' + result_generated1[0] + '\n\n')

        with open(mflFilename,'a') as mflFile:
            mflFile.write(flightID + ' ' + '330' + '\n')
        # except(TypeError): 
        #     pass

    # print()

    # counter +=1
    # flightID = str(random.choice(airlines)) + str(int(counter))
    
    # print("Flight type 2", flightID);
    # result_generated2 = f.generate(2, orgAirport, destAirport, "", "", myOrgRwy, myDestRwy);
    # print("Result flight plan = ", result_generated2[0]);
    # print("Starting point latitude/longitude = ", result_generated2[1], ",", result_generated2[2]);

    # TRACK_TIME += 10
    # with open(trxFilename,'a') as trxFile:
    #     tstr = '%s %s %s %.4f %.4f %d %.2f %d %s %s' % ('TRACK',flightID,aircraftType,float(result_generated2[1]),float(result_generated2[2]),0,elev,28)
    #     trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
    #     trxFile.write( tstr + '\n')
    #     trxFile.write('    FP_ROUTE ' + result_generated2[0] + '\n\n')

    # with open(mflFilename,'a') as mflFile:
    #     mflFile.write(flightID + ' ' + '330' + '\n')

    # print()

    # if not(orgSFO):
    #     counter +=1
    #     flightID = str(random.choice(airlines)) + str(int(counter))
        
    #     print("Flight type 3", flightID);
    #     result_generated3 = f.generate(3, orgAirport, destAirport, "", "", "", "");
    #     print("Result flight plan = ", result_generated3[0]);
    #     print("Starting point latitude/longitude = ", result_generated3[1], ",", result_generated3[2]);

    #     newLat = float(result_generated3[1]) + 10000
    #     newLong = float(result_generated3[2]) + 10000
    #     TRACK_TIME += 10
    #     with open(trxFilename,'a') as trxFile:
    #         tstr = '%s %s %s %.4f %.4f %d %.2f %d %s %s' % ('TRACK',flightID,aircraftType,newLat,newLong,200,100,28,'ZOA','ZOA46')
    #         trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
    #         trxFile.write( tstr + '\n')
    #         trxFile.write('    FP_ROUTE ' + result_generated3[0] + '\n\n')

    #     with open(mflFilename,'a') as mflFile:
    #         mflFile.write(flightID + ' ' + '330' + '\n')

    # print()

    # if not(orgSFO):
    #     counter +=1
    #     flightID = str(random.choice(airlines)) + str(int(counter))
        
    #     print("Flight type 4",flightID);
    #     try:
    #         result_generated4 = f.generate(4, orgAirport, destAirport, "", myDestGate, "", myDestRwy);
    #         print("Result flight plan = ", result_generated4[0]);
    #         print("Starting point latitude/longitude = ", result_generated4[1], ",", result_generated4[2]);

    #         TRACK_TIME += 10
    #         newLat = 373704.2095999999889955 - 10000. # float(result_generated4[1]) + 10000.
    #         newLong = -1222325.53360000001817 + 10000. # float(result_generated4[2]) + 10000.

    #         with open(trxFilename,'a') as trxFile:
    #             tstr = '%s %s %s %.4f %.4f %d %.2f %d %s %s' % ('TRACK',flightID,aircraftType,newLat,newLong,200,100,28,'ZOA','ZOA46')
    #             trxFile.write('%s %d' % ('TRACKTIME', TRACK_TIME) + '\n')
    #             trxFile.write( tstr + '\n')
    #             trxFile.write('    FP_ROUTE ' + result_generated4[0] + '\n\n')
    #         with open(mflFilename,'a') as mflFile:
    #             mflFile.write(flightID + ' ' + '330' + '\n')
    #     except(TypeError):
    #         pass

gnatsStandalone.stop();

shutdownJVM()