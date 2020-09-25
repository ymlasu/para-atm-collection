import time

from paraatm.io.gnats import GnatsSimulationWrapper, GnatsEnvironment

class GateToGate(GnatsSimulationWrapper):
    def simulation(self, pushback_clearance_delay=7):

        GNATS_SIMULATION_STATUS_PAUSE = GnatsEnvironment.get_gnats_constant('GNATS_SIMULATION_STATUS_PAUSE')
        GNATS_SIMULATION_STATUS_ENDED = GnatsEnvironment.get_gnats_constant('GNATS_SIMULATION_STATUS_ENDED')
        AIRCRAFT_CLEARANCE_PUSHBACK = GnatsEnvironment.get_gnats_constant('AIRCRAFT_CLEARANCE_PUSHBACK', 'AircraftClearance')

        DIR_share = GnatsEnvironment.share_dir

        simulationInterface = GnatsEnvironment.simulationInterface
        environmentInterface = GnatsEnvironment.environmentInterface
        aircraftInterface = GnatsEnvironment.aircraftInterface
        controllerInterface = GnatsEnvironment.controllerInterface

        simulationInterface.clear_trajectory()

        environmentInterface.load_rap(DIR_share + "/tg/rap")

        aircraftInterface.load_aircraft(DIR_share + "/tg/trx/TRX_DEMO_SFO_PHX_GateToGate_geo.trx", DIR_share + "/tg/trx/TRX_DEMO_SFO_PHX_mfl.trx")

        #     # Controller to set human error: delay time
        #     # Users can try the following setting and see the difference in trajectory
        controllerInterface.setDelayPeriod("SWA1897", AIRCRAFT_CLEARANCE_PUSHBACK, 7)
        #controllerInterface.setDelayPeriod("SWA1897", AIRCRAFT_CLEARANCE_TAKEOFF, 20)

        simulationInterface.setupSimulation(22000, 30) # SFO - PHX

        simulationInterface.start(660)

        # Use a while loop to constantly check simulation status.  When the simulation finishes, continue to output the trajectory data
        while True:
            runtime_sim_status = simulationInterface.get_runtime_sim_status()
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

        simulationInterface.resume()

        while True:
            runtime_sim_status = simulationInterface.get_runtime_sim_status()
            if (runtime_sim_status == GNATS_SIMULATION_STATUS_ENDED) :
                break
            else:
                time.sleep(1)

    def write_output(self, filename):
        GnatsEnvironment.simulationInterface.write_trajectories(filename)

    def cleanup(self):
        GnatsEnvironment.aircraftInterface.release_aircraft()
        GnatsEnvironment.environmentInterface.release_rap()
