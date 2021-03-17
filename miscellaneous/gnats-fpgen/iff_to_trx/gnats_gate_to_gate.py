import time

from paraatm.io.gnats import GnatsSimulationWrapper, GnatsEnvironment

class GateToGate(GnatsSimulationWrapper):
    def __init__(self):
        GnatsEnvironment.start_jvm()

        self.gnatsStandalone = GnatsEnvironment.get_gnats_standalone()

        self.simulationInterface = self.gnatsStandalone.getSimulationInterface()

        if self.simulationInterface is None:
            self.gnatsStandalone.stop()
            raise RuntimeError("Can't get SimulationInterface")

        self.entityInterface = self.gnatsStandalone.getEntityInterface()
        self.controllerInterface = self.entityInterface.getControllerInterface()
        self.pilotInterface = self.entityInterface.getPilotInterface()

        self.environmentInterface = self.gnatsStandalone.getEnvironmentInterface()

        self.equipmentInterface = self.gnatsStandalone.getEquipmentInterface()
        self.aircraftInterface = self.equipmentInterface.getAircraftInterface()
        self.airportInterface = self.environmentInterface.getAirportInterface()

    def setupAircraft(self,**kwargs):
        trx_file = kwargs["trx_file"]
        mfl_file= kwargs["mfl_file"]
        self.aircraftInterface.load_aircraft(trx_file,mfl_file)

    def simulation(self, *args, **kwargs):

        GNATS_SIMULATION_STATUS_PAUSE = GnatsEnvironment.get_gnats_constant('GNATS_SIMULATION_STATUS_PAUSE')
        GNATS_SIMULATION_STATUS_ENDED = GnatsEnvironment.get_gnats_constant('GNATS_SIMULATION_STATUS_ENDED')
        
        DIR_share = GnatsEnvironment.share_dir

        # simulationInterface = GnatsEnvironment.simulationInterface
        # environmentInterface = GnatsEnvironment.environmentInterface
        # aircraftInterface = GnatsEnvironment.aircraftInterface

        self.simulationInterface.clear_trajectory()

        self.environmentInterface.load_rap(DIR_share + "/tg/rap")


            # Controller to set human error: delay time
            # Users can try the following setting and see the difference in trajectory
        # self.controllerInterface.setDelayPeriod("SWA1897", AIRCRAFT_CLEARANCE_PUSHBACK, 7)
        # self.controllerInterface.setDelayPeriod("SWA1897", AIRCRAFT_CLEARANCE_TAKEOFF, 20)

        self.simulationInterface.setupSimulation(7200,1) # SFO - PHX

        self.simulationInterface.start()

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
