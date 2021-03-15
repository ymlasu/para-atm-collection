from jpype import *
from array import *

import os
import time
import sys

import platform

from shutil import copyfile
from shutil import rmtree

import PostProcessor as pp

import PathVisualizer

print("!!!!!!!!!!!!!!!!!!")
print("Notice")
print("Some GNATS samples use Google map for presentation.  It requires Google map API key.  The key may exceed the maximum quota due to enormous NATS users' activities.")
print("If you experience Google map failure, please edit PathVisualizer.py with a new Google map key obtained from developers.google.com/maps/documentation/javascript/get-api-key")
print("!!!!!!!!!!!!!!!!!!")
print("")

env_GNATS_HOME = os.environ.get('GNATS_HOME')

str_GNATS_HOME = ""

if not(env_GNATS_HOME is None) :
	str_GNATS_HOME = env_GNATS_HOME + "/"

if platform.system() == "Windows" :
	classpath = str_GNATS_HOME + "dist/gnats-standalone.jar"
	classpath = classpath + ";" + str_GNATS_HOME + "../GNATS_Client/dist/gnats-client.jar"
	classpath = classpath + ";" + str_GNATS_HOME + "../GNATS_Client/dist/gnats-shared.jar"
	classpath = classpath + ";" + str_GNATS_HOME + "../GNATS_Client/dist/json.jar"
	classpath = classpath + ";" + str_GNATS_HOME + "../GNATS_Client/dist/commons-logging-1.2.jar"
else :
	classpath = str_GNATS_HOME + "dist/gnats-standalone.jar"
	classpath = classpath + ":" + str_GNATS_HOME + "../GNATS_Client/dist/gnats-client.jar"
	classpath = classpath + ":" + str_GNATS_HOME + "../GNATS_Client/dist/gnats-shared.jar"
	classpath = classpath + ":" + str_GNATS_HOME + "../GNATS_Client/dist/json.jar"
	classpath = classpath + ":" + str_GNATS_HOME + "../GNATS_Client/dist/commons-logging-1.2.jar"

DIR_share = "../GNATS_Server/share";

startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % classpath)

# Flight phase value definition
# You can detect and know the flight phase by checking its value
FLIGHT_PHASE_PREDEPARTURE = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_PREDEPARTURE;
FLIGHT_PHASE_ORIGIN_GATE = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_ORIGIN_GATE;
FLIGHT_PHASE_PUSHBACK = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_PUSHBACK;
FLIGHT_PHASE_RAMP_DEPARTING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_RAMP_DEPARTING;
FLIGHT_PHASE_TAXI_DEPARTING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_TAXI_DEPARTING;
FLIGHT_PHASE_RUNWAY_THRESHOLD_DEPARTING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_RUNWAY_THRESHOLD_DEPARTING;
FLIGHT_PHASE_TAKEOFF = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_TAKEOFF;
FLIGHT_PHASE_CLIMBOUT = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_CLIMBOUT;
FLIGHT_PHASE_HOLD_IN_DEPARTURE_PATTERN = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_HOLD_IN_DEPARTURE_PATTERN;
FLIGHT_PHASE_CLIMB_TO_CRUISE_ALTITUDE = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_CLIMB_TO_CRUISE_ALTITUDE;
FLIGHT_PHASE_TOP_OF_CLIMB = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_TOP_OF_CLIMB;
FLIGHT_PHASE_CRUISE = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_CRUISE;
FLIGHT_PHASE_HOLD_IN_ENROUTE_PATTERN = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_HOLD_IN_ENROUTE_PATTERN;
FLIGHT_PHASE_TOP_OF_DESCENT = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_TOP_OF_DESCENT;
FLIGHT_PHASE_INITIAL_DESCENT = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_INITIAL_DESCENT;
FLIGHT_PHASE_HOLD_IN_ARRIVAL_PATTERN = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_HOLD_IN_ARRIVAL_PATTERN
FLIGHT_PHASE_APPROACH = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_APPROACH;
FLIGHT_PHASE_FINAL_APPROACH = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_FINAL_APPROACH;
FLIGHT_PHASE_GO_AROUND = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_GO_AROUND;
FLIGHT_PHASE_TOUCHDOWN = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_TOUCHDOWN;
FLIGHT_PHASE_LAND = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_LAND;
FLIGHT_PHASE_EXIT_RUNWAY = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_EXIT_RUNWAY;
FLIGHT_PHASE_TAXI_ARRIVING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_TAXI_ARRIVING;
FLIGHT_PHASE_RUNWAY_CROSSING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_RUNWAY_CROSSING;
FLIGHT_PHASE_RAMP_ARRIVING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_RAMP_ARRIVING;
FLIGHT_PHASE_DESTINATION_GATE = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_DESTINATION_GATE;
FLIGHT_PHASE_LANDED = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_LANDED;
FLIGHT_PHASE_HOLDING = JPackage('com').osi.util.FlightPhase.FLIGHT_PHASE_HOLDING;

AIRCRAFT_CLEARANCE_PUSHBACK = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_PUSHBACK
AIRCRAFT_CLEARANCE_TAXI_DEPARTING = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_TAXI_DEPARTING
AIRCRAFT_CLEARANCE_TAKEOFF = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_TAKEOFF
AIRCRAFT_CLEARANCE_ENTER_ARTC = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_ENTER_ARTC
AIRCRAFT_CLEARANCE_DESCENT_FROM_CRUISE = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_DESCENT_FROM_CRUISE
AIRCRAFT_CLEARANCE_ENTER_TRACON = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_ENTER_TRACON
AIRCRAFT_CLEARANCE_APPROACH = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_APPROACH
AIRCRAFT_CLEARANCE_TOUCHDOWN = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_TOUCHDOWN
AIRCRAFT_CLEARANCE_TAXI_LANDING = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_TAXI_LANDING
AIRCRAFT_CLEARANCE_RAMP_LANDING = JPackage('com').osi.util.AircraftClearance.AIRCRAFT_CLEARANCE_RAMP_LANDING

# NATS simulation status definition
# You can get simulation status and know what it refers to
GNATS_SIMULATION_STATUS_READY = JPackage('com').osi.util.Constants.GNATS_SIMULATION_STATUS_READY
GNATS_SIMULATION_STATUS_START = JPackage('com').osi.util.Constants.GNATS_SIMULATION_STATUS_START
GNATS_SIMULATION_STATUS_PAUSE = JPackage('com').osi.util.Constants.GNATS_SIMULATION_STATUS_PAUSE
GNATS_SIMULATION_STATUS_RESUME = JPackage('com').osi.util.Constants.GNATS_SIMULATION_STATUS_RESUME
GNATS_SIMULATION_STATUS_STOP = JPackage('com').osi.util.Constants.GNATS_SIMULATION_STATUS_STOP
GNATS_SIMULATION_STATUS_ENDED = JPackage('com').osi.util.Constants.GNATS_SIMULATION_STATUS_ENDED

clsGNATSStandalone = JClass('GNATSStandalone')
# Start GNATS Standalone environment
gnatsStandalone = clsGNATSStandalone.start()

if (gnatsStandalone is None) :
	print("Can't start GNATS Standalone")
	quit()

simulationInterface = gnatsStandalone.getSimulationInterface()

entityInterface = gnatsStandalone.getEntityInterface()
controllerInterface = entityInterface.getControllerInterface()
pilotInterface = entityInterface.getPilotInterface()

environmentInterface = gnatsStandalone.getEnvironmentInterface()
airportInterface = environmentInterface.getAirportInterface()
weatherInterface = environmentInterface.getWeatherInterface()
terminalAreaInterface = environmentInterface.getTerminalAreaInterface()
terrainInterface = environmentInterface.getTerrainInterface()

equipmentInterface = gnatsStandalone.getEquipmentInterface()
aircraftInterface = equipmentInterface.getAircraftInterface()
cnsInterface = equipmentInterface.getCNSInterface()

riskMeasuresInterface = gnatsStandalone.getRiskMeasuresInterface()

if simulationInterface is None:
	print("Can't get SimulationInterface")
	exit()
