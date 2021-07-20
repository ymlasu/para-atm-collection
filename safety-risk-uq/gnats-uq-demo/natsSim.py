"""Implementation of simple 2-aircraft NATS simulation

The paraatm Python package is used to facilitate interfacing with
NATS, but using the NatsSimulationWrapper base class.  The simulation
method is written to accept options that are used to control the
simulation: the latitude and longitude coordinates for the specified
waypoint of the first aircraft.  These serve as input variables to
control the simulation.
"""

import time
import os

from paraatm.io.nats import NatsSimulationWrapper, NatsEnvironment

class NatsSim(NatsSimulationWrapper):
    TRX_NAME = 'iff_to_gnats_geo_SFO5.trx'
    MFL_NAME = 'iff_to_gnats_geo_SFO5_mfl.trx'
    def simulation(self,cont_resp=10.):
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

        NATS_SIMULATION_STATUS_PAUSE = NatsEnvironment.get_nats_constant('GNATS_SIMULATION_STATUS_PAUSE')
        NATS_SIMULATION_STATUS_ENDED = NatsEnvironment.get_nats_constant('GNATS_SIMULATION_STATUS_ENDED')

        AIRCRAFT_CLEARANCE_PUSHBACK = NatsEnvironment.get_nats_clearance("AIRCRAFT_CLEARANCE_PUSHBACK")
        AIRCRAFT_CLEARANCE_TAXI_DEPARTING = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_TAXI_DEPARTING")
        AIRCRAFT_CLEARANCE_TAKEOFF = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_TAKEOFF")
        AIRCRAFT_CLEARANCE_ENTER_ARTC = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_ENTER_ARTC")
        AIRCRAFT_CLEARANCE_DESCENT_FROM_CRUISE = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_DESCENT_FROM_CRUISE")
        AIRCRAFT_CLEARANCE_ENTER_TRACON = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_ENTER_TRACON")
        AIRCRAFT_CLEARANCE_APPROACH = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_APPROACH")
        AIRCRAFT_CLEARANCE_TOUCHDOWN = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_TOUCHDOWN")
        AIRCRAFT_CLEARANCE_TAXI_LANDING = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_TAXI_LANDING")
        AIRCRAFT_CLEARANCE_RAMP_LANDING = NatsEnvironment.get_nats_clearance( "AIRCRAFT_CLEARANCE_RAMP_LANDING")

        acClearances = [AIRCRAFT_CLEARANCE_PUSHBACK, AIRCRAFT_CLEARANCE_TAXI_DEPARTING, AIRCRAFT_CLEARANCE_TAKEOFF, 
                    AIRCRAFT_CLEARANCE_ENTER_ARTC, AIRCRAFT_CLEARANCE_DESCENT_FROM_CRUISE, AIRCRAFT_CLEARANCE_ENTER_TRACON, 
                    AIRCRAFT_CLEARANCE_APPROACH, AIRCRAFT_CLEARANCE_TOUCHDOWN, AIRCRAFT_CLEARANCE_TAXI_LANDING, 
                    AIRCRAFT_CLEARANCE_RAMP_LANDING]
       
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
        DIR_share = NatsEnvironment.share_dir

        environmentInterface.load_rap(DIR_share+"/tg/rap")
        aircraftInterface.load_aircraft(NatsEnvironment.build_path(self.TRX_NAME), NatsEnvironment.build_path(self.MFL_NAME))
        aclist = list(aircraftInterface.getAllAircraftId())

        simulationInterface.setupSimulation(7200, 1)
    
        for ac in aclist:
            for acClearance in acClearances:
                controllerInterface.setDelayPeriod(ac,acClearance,cont_resp)

        simulationInterface.start()
        print('Running Simulation...')

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
