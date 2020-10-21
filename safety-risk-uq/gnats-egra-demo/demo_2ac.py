"""Implementation of simple 2-aircraft NATS simulation

The paraatm Python package is used to facilitate interfacing with
NATS, but using the NatsSimulationWrapper base class.  The simulation
method is written to accept options that are used to control the
simulation: the latitude and longitude coordinates for the specified
waypoint of the first aircraft.  These serve as input variables to
control the simulation.
"""

import time

from paraatm.io.nats import NatsSimulationWrapper, NatsEnvironment

class TwoAcSim(NatsSimulationWrapper):
    def simulation(self, waypoint_index=4, latitude=34.362, longitude=-85.764):
        """
        Run simulation

        Parameters
        ----------
        waypoint_index : int
            Waypoint index (appears to be 0-based) that the response
            function input values will control
        latitude : float
            Waypoint latitude value to set
        longitude : float
            Waypoint longitude value to set
        """

        NATS_SIMULATION_STATUS_PAUSE = NatsEnvironment.get_nats_constant('NATS_SIMULATION_STATUS_PAUSE')
        NATS_SIMULATION_STATUS_ENDED = NatsEnvironment.get_nats_constant('NATS_SIMULATION_STATUS_ENDED')

        natsStandalone = NatsEnvironment.get_nats_standalone()

        simulationInterface = natsStandalone.getSimulationInterface()

        entityInterface = natsStandalone.getEntityInterface()
        controllerInterface = entityInterface.getControllerInterface()
        pilotInterface = entityInterface.getPilotInterface()

        environmentInterface = natsStandalone.getEnvironmentInterface()

        equipmentInterface = natsStandalone.getEquipmentInterface()
        aircraftInterface = equipmentInterface.getAircraftInterface()

        if simulationInterface is None:
            natsStandalone.stop()
            raise RuntimeError("Can't get SimulationInterface")

        simulationInterface.clear_trajectory()

        environmentInterface.load_rap("share/tg/rap")

        aircraftInterface.load_aircraft(NatsEnvironment.build_path('demo_2ac.trx'), NatsEnvironment.build_path('demo_2ac_mfl.trx'))

        aclist = aircraftInterface.getAllAircraftId()
        ac = aircraftInterface.select_aircraft(aclist[0])

        ac.setFlight_plan_latitude_deg(waypoint_index, latitude)
        ac.setFlight_plan_longitude_deg(waypoint_index, longitude)
        
        # Useful print statements to either check the initial waypoint
        # coordinates or to verify that the setFlight_plan function is
        # working as expected
        # print('lat:', ac.getFlight_plan_latitude_array())
        # print('long:', ac.getFlight_plan_longitude_array())


        simulationInterface.setupSimulation(30000, 30)

        simulationInterface.start()

        while True:
            runtime_sim_status = simulationInterface.get_runtime_sim_status()
            if (runtime_sim_status == NATS_SIMULATION_STATUS_ENDED) :
                break
            else:
                time.sleep(1)

        # Store attribute for access by write_output and cleanup:
        self.simulationInterface = simulationInterface
        self.environmentInterface = environmentInterface
        self.aircraftInterface = aircraftInterface

    def write_output(self, filename):
        self.simulationInterface.write_trajectories(filename)

    def cleanup(self):
        self.aircraftInterface.release_aircraft()
        self.environmentInterface.release_rap()        
