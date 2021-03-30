import time

from paraatm.io.nats import NatsSimulationWrapper, NatsEnvironment

class GateToGate(NatsSimulationWrapper):
    def __init__(self):
        NatsEnvironment.start_jvm()
        natsStandalone = NatsEnvironment.get_nats_standalone()

        self.simulationInterface = natsStandalone.getSimulationInterface()

        self.entityInterface = natsStandalone.getEntityInterface()
        self.controllerInterface = self.entityInterface.getControllerInterface()
        self.pilotInterface = self.entityInterface.getPilotInterface()

        self.environmentInterface = natsStandalone.getEnvironmentInterface()
        self.airportInterface = self.environmentInterface.getAirportInterface()
        self.weatherInterface = self.environmentInterface.getWeatherInterface()
        self.terminalAreaInterface = self.environmentInterface.getTerminalAreaInterface()
        self.terrainInterface = self.environmentInterface.getTerrainInterface()

        self.equipmentInterface = natsStandalone.getEquipmentInterface()
        self.aircraftInterface = self.equipmentInterface.getAircraftInterface()

        self.DIR_share = NatsEnvironment.share_dir    
    def simulation(self, trx_name,mfl_name):

        GNATS_SIMULATION_STATUS_PAUSE = NatsEnvironment.get_nats_constant('GNATS_SIMULATION_STATUS_PAUSE')
        GNATS_SIMULATION_STATUS_ENDED = NatsEnvironment.get_nats_constant('GNATS_SIMULATION_STATUS_ENDED')

        DIR_share = NatsEnvironment.share_dir

        self.simulationInterface.clear_trajectory()

        self.environmentInterface.load_rap(self.DIR_share + "/tg/rap")

        self.aircraftInterface.load_aircraft(trx_name, mfl_name)

        #     # Controller to set human error: delay time
        #     # Users can try the following setting and see the difference in trajectory
        #self.controllerInterface.setDelayPeriod("SWA1897", AIRCRAFT_CLEARANCE_PUSHBACK, 7)
        #controllerInterface.setDelayPeriod("SWA1897", AIRCRAFT_CLEARANCE_TAKEOFF, 20)

        self.simulationInterface.setupSimulation(22000, 30,1,10) # SFO - PHX

        self.simulationInterface.start()

        # Use a while loop to constantly check simulation status.  When the simulation finishes, continue to output the trajectory data
        while True:
            runtime_sim_status = self.simulationInterface.get_runtime_sim_status()
            if (runtime_sim_status == GNATS_SIMULATION_STATUS_PAUSE) :
                break
            else:
                time.sleep(1)

        # Pilot to set error scenarios
        # Users can try the following setting and see the difference in trajectory
        #pilotInterface.skipFlightPhase('SWA1897', 'FLIGHT_PHASE_CLIMB_TO_CRUISE_ALTITUDE')
        #pilotInterface.setActionRepeat('SWA1897', "VERTICAL_SPEED")
        #pilotInterface.setWrongAction('SWA1897', "AIRSPEED", "FLIGHT_LEVEL")
        #pilotInterface.setActionReversal('SWA1897', 'VERTICAL_SPEED')
        #pilotInterface.setPartialAction('SWA1897', 'COURSE', 200, 50)
        #pilotInterface.skipChangeAction('SWA1897', 'COURSE')
        #pilotInterface.setActionLag('SWA1897', 'COURSE', 10, 0.05, 60)

        self.simulationInterface.resume()

        while True:
            runtime_sim_status = self.simulationInterface.get_runtime_sim_status()
            if (runtime_sim_status == GNATS_SIMULATION_STATUS_ENDED) :
                break
            else:
                time.sleep(1)

    def write_output(self, filename):
        self.simulationInterface.write_trajectories(filename)

    def cleanup(self):
        self.aircraftInterface.release_aircraft()
        self.environmentInterface.release_rap()
