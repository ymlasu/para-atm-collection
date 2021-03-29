from nats_functions import (get_closest_node_at_airport,
                            get_list_of_adjacent_nodes,
                            get_adjacent_node_closer_to_runway,
                            get_closest_airport,
                            get_landing_rwy_entry_and_end_point)

from paraatm.io.gnats import GnatsEnvironment

def get_departure_airport_from_iff(iff_data,callsign,gnatsSim,arrivalAirport=None,flmap=None):
    import random
    origin_opts = []

    asdex_airport = iff_data[3][iff_data[3].callsign==callsign].Source.unique()[0][:3]
    
    # Get all unique origin options from the iff_data set
    for key in iff_data.keys():
        df = iff_data[key][iff_data[key].callsign==callsign].copy()    
        if 'Orig' in df.columns:
            df.dropna(axis=0,subset=['Orig'],inplace=True)
            origin_opts.extend([orig for orig in list(df.Orig.unique()) if not orig=='nan'])
        if 'estOrig' in df.columns:
            df.dropna(axis=0,subset=['estOrig'],inplace=True)
            origin_opts.extend([orig for orig in list(df.estOrig.unique()) if not orig=='nan'])

    # In the case of multiple origin options names, take the first one
    if len(origin_opts)>0:
        origin = list(set((origin_opts)))[0]
    else:
        origin = []

    # Add K to the front of an airport code to make it compatible with NATS
    if len(origin)==3:
        origin = 'K'+origin
        
    elif len(origin) == 4 and origin[0]=='K':
        origin = origin
    elif (arrivalAirport is not None and flmap is not None):
        fplist=[key for key in flmap if (key.endswith(arrivalAirport) or key.endswith(arrivalAirport[1:]))]
        depOpts = [dep.split('-')[0] for dep in fplist]
        #departOpts = [dep for dep in departOpts if len(dep)==3]
        origin = random.choice(depOpts)
        
    return origin
        

def get_arrival_airport_from_iff(iff_data,callsign,gnatsSim,departureAirport,flmap):
    import random

    dest_opts = []

    asdex_airport = iff_data[3][iff_data[3].callsign==callsign].Source.unique()[0][:3]
    
    allAirports = [apt[-3] for apt in list(gnatsSim.airportInterface.getAllAirportCodesInGNATS())]

    # Get all unique origin options from the iff_data set
    for key in iff_data.keys():
        df = iff_data[key][iff_data[key].callsign==callsign].copy()   
        if 'Dest' in df.columns:
            df.dropna(axis=0,subset=['Dest'],inplace=True)
            dest_opts.extend([orig for orig in list(df.Dest.unique()) if not orig=='nan'])
        if 'estDest' in df.columns:
            df.dropna(axis=0,subset=['estDest'],inplace=True)
            dest_opts.extend([orig for orig in list(df.estDest.unique()) if not orig=='nan'])
    # In the case of multiple origin options names, take the first one

    if len(dest_opts)>0:
        dest = list(set(dest_opts))[0]
    else:
        dest = []
        
    # Add K to the front of an airport code to make it compatible with NATS
    if len(dest)==3:
        dest = 'K'+dest
        
    elif len(dest) == 4 and dest[0]=='K':
        dest = dest

    else:
        print("No viable destination airport found for {}. Returning random from FlightPlanSelector options.".format(callsign,'K'+asdex_airport))
        
        fplist=[key for key in flmap if (key.startswith(departureAirport) or key.startswith(departureAirport[1:]))]
        departOpts = [dep.split('-')[1] for dep in fplist]
        allAirports = [apt[-3:] for apt in list(gnatsSim.airportInterface.getAllAirportCodesInGNATS())]
        departOpts = [dep for dep in departOpts if dep[-3:] in allAirports]
        dest = random.choice(departOpts)
    return dest

def check_if_flight_has_departed(iff_data,callsign,natsSim,departureAirport):
    import numpy as np
    departureAirportElevation = natsSim.airportInterface.select_airport(departureAirport).getElevation()
    departureAirportLat, departureAirportLon = natsSim.airportInterface.getLocation(departureAirport)

    initial_lat = iff_data[3].loc[iff_data[3].callsign==callsign,'latitude'].iloc[0]
    initial_lon = iff_data[3].loc[iff_data[3].callsign==callsign,'longitude'].iloc[0]
    initial_alt = iff_data[3].loc[iff_data[3].callsign==callsign,'altitude'].iloc[0]

    dist_from_airport = np.sqrt((departureAirportLat-initial_lat)**2+(departureAirportLon-initial_lon)**2)
    
    if ((initial_alt < departureAirportElevation+50.) & (dist_from_airport < 0.1)):
        flightTakenOff = False
    else:
        flightTakenOff = True
    return flightTakenOff

def check_if_flight_landing_in_dataset(iff_data,callsign,natsSim,arrivalAirport):
    import numpy as np
    arrivalAirportElevation = natsSim.airportInterface.select_airport(arrivalAirport).getElevation()
    arrivalAirportLat, arrivalAirportLon = natsSim.airportInterface.getLocation(arrivalAirport)

    lat = iff_data[3].loc[iff_data[3].callsign==callsign,'latitude'].iloc[-1]
    lon = iff_data[3].loc[iff_data[3].callsign==callsign,'longitude'].iloc[-1]
    alt = iff_data[3].loc[iff_data[3].callsign==callsign,'altitude'].iloc[-1]

    dist_from_airport = np.sqrt((arrivalAirportLat-lat)**2+(arrivalAirportLon-lon)**2)
    
    if ((alt < arrivalAirportElevation+50.) & (dist_from_airport < 0.02)):
        flightHasLanded = True
    else:
        flightHasLanded = False
    return flightHasLanded


def create_gate_to_runway_from_iff(trackData,natsSim,departureAirport):

    # Assign airport nodes from NATS to all track data points
    trackData.loc[:,'airportNodes']= [get_closest_node_at_airport(lat,lon,departureAirport) for lat,lon in zip(trackData.latitude,trackData.longitude)]

    # Determine the takeoff runway from track data
    takeoff_rwy = get_takeoff_runway_from_track_data(trackData)
    print(takeoff_rwy)

    # Get the list of unique nodes identified in the track data
    unique_nodes = trackData.airportNodes.unique()
    # Get the first node that starts with a gate
    # TODO: Return an error if trackList is empty
    gateList = [node for node in unique_nodes if 'Gate' in node]
    i=0
    trackList=[]
    while not trackList:
        if gateList:
            trackList=[gateList[0]]
        else:
            adjacent_nodes=get_list_of_adjacent_nodes(unique_nodes[i],departureAirport)
            adjacent_gates=[node for node in adjacent_nodes if 'Gate' in node]
            if adjacent_gates:
                trackList=[adjacent_gates[0]]
        i += 1
        
    # Get from gate to taxiways while not allowing going back to the gate
    # TODO: How to handle if it does go back to the gate?
    while 'Txy' not in trackList[-1]:

        # Get list of adjacent nodes to the last node in the trackList.
        # Upon first iteration this will be the gate
        adjacent_nodes = get_list_of_adjacent_nodes(trackList[-1],departureAirport)

        # Get list of adjacent nodes that have already been identified
        # in the set of unique nodes
        adjacent_nodes_in_unique_nodes = [adj for adj in adjacent_nodes
                                          if (adj in unique_nodes and
                                              adj not in trackList and
                                              'Gate' not in adj)]

        # If only one of the adjacent nodes is in the set of unique nodes
        # Use it as the next node
        if len(adjacent_nodes_in_unique_nodes) == 1:
            trackList.append(adjacent_nodes_in_unique_nodes[0])

        # If more than one of the adjacent nodes is in the set of unique
        # nodes, identify the one that is closest to the runway
        # and use it as the next node
        elif len(adjacent_nodes_in_unique_nodes) > 1:
            # Use adjacent node that gets closer to the runway
            trackList.append(get_adjacent_node_closer_to_runway(
                adjacent_nodes_in_unique_nodes,takeoff_rwy,departureAirport))

        # If none of the adjacent nodes have been identified as being in
        # the set of unique nodes, use the node that is closest to the
        # runway.
        # TODO: Could also use the one that is closest to the next node in
        # the unique node set
        else:
            adjacent_nodes_not_in_trackList = [adj for adj in adjacent_nodes
                                               if adj not in trackList and 'Gate'
                                               not in adj]

            trackList.append(get_adjacent_node_closer_to_runway(adjacent_nodes_not_in_trackList,takeoff_rwy,departureAirport))
    removed_nodes=[]
    while takeoff_rwy not in trackList[-1]:
        print('TrackList:',trackList)
        print('Removed Nodes:',removed_nodes)
        adjacent_nodes = get_list_of_adjacent_nodes(trackList[-1],departureAirport)
        adjacent_nodes_in_unique_nodes = [adj for adj in adjacent_nodes if (adj in unique_nodes and adj not in trackList and adj not in removed_nodes and 'Gate' not in adj)]

        print('Adjacent nodes in unique nodes:',adjacent_nodes_in_unique_nodes)
        if len(adjacent_nodes_in_unique_nodes) == 1:
            print("There is one unique node in adjacent nodes.")
            trackList.append(adjacent_nodes_in_unique_nodes[0])
        elif len(adjacent_nodes_in_unique_nodes) > 1:
            print('There is more than one unique node in adjacent nodes.')
            trackList.append(get_adjacent_node_closer_to_runway(adjacent_nodes_in_unique_nodes,takeoff_rwy,departureAirport,removed_nodes=removed_nodes))
        else:
            print('Using an adjacent node getting closer to the runway')
            adjacent_nodes_not_in_trackList = [adj for adj in adjacent_nodes if adj not in trackList and 'Gate' not in adj]
            print('Adjacent nodes not in trackList:',adjacent_nodes_not_in_trackList)
            if not adjacent_nodes_not_in_trackList:
                print('All adjacent nodes already in trackList.')
                removed_nodes.append(trackList[-1])
                trackList=trackList[:-1]
            else:
                trackList.append(get_adjacent_node_closer_to_runway(adjacent_nodes_not_in_trackList,takeoff_rwy,departureAirport,removed_nodes=removed_nodes))
                        
    return trackList

     
def get_runway_from_track_data(trackData,minRwySpeed=30.):
    import numpy as np
    
    trackData = trackData[trackData.tas >= minRwySpeed].copy()
    runway_segments = [node for node in trackData.airportNodes if 'Rwy' in node]
    runway_numbers = [seg.split('_')[1] for seg in runway_segments]
    rwys_opts,counts = np.unique(runway_numbers,return_counts=True)
    idx = np.argmax(counts)
    takeoff_runway_no = rwys_opts[idx]
    #Get first Rwy segment in trackData that has takeoff_runway_no in it
    takeoff_runway_node = [rwy for rwy in runway_segments if takeoff_runway_no in rwy][0]
    # TODO:Check if takeoff_runway_node is an entry point in NATS
    # If not then find the closest runway entry point in NATS

    return takeoff_runway_node

def get_arrival_gate_and_rwy_from_iff(iff_data,callsign,gnatsSim,arrivalAirport,minRwySpeed=30.):
    import numpy as np
    import random

    trackData=iff_data[3].loc[iff_data[3].callsign==callsign]
    trackData = trackData[trackData.tas >= minRwySpeed].copy()
    trackData.loc[:,'airportNodes']= [get_closest_node_at_airport(lat,lon,arrivalAirport) for lat,lon in zip(trackData.latitude,trackData.longitude)]

    runway_segments = [node for node in trackData.airportNodes if 'Rwy' in node]
    runway_numbers = [seg.split('_')[1] for seg in runway_segments]
    rwys_opts,counts = np.unique(runway_numbers,return_counts=True)
    idx = np.argmax(counts)
    arrival_runway_no = rwys_opts[idx]
    #Get first Rwy segment in trackData that has takeoff_runway_no in it
    arrival_runway_node = [rwy for rwy in runway_segments if arrival_runway_no in rwy][0]
    # TODO:Check if takeoff_runway_node is an entry point in NATS
    # If not then find the closest runway entry point in NATS

        # Get the list of unique nodes identified in the track data
    unique_nodes = trackData.airportNodes.unique()
    # Get the first node that starts with a gate
    # TODO: Return an error if trackList is empty
    gateList = [node for node in unique_nodes if 'Gate' in node]
    if gateList:
        arrival_gate = gateList[-1]
    else:
        gateOpts = gnatsSim.airportInterface.getAllGates(arrivalAirport)
        gateOpts = [opt for opt in gateOpts if opt.lower().startswith('gate')]
        arrival_gate = random.choice(gateOpts)

    rwy_entry,rwy_end=get_landing_rwy_entry_and_end_point(arrival_runway_node,arrivalAirport,domain=['Rwy'])

    return rwy_end,arrival_gate

def get_departure_rwy_from_iff(iff_data,callsign,gnatsSim,departureAirport,minRwySpeed=30.):
    import numpy as np

    trackData=iff_data[3].loc[iff_data[3].callsign==callsign]
    trackData = trackData[trackData.tas >= minRwySpeed].copy()
    trackData.loc[:,'airportNodes']= [get_closest_node_at_airport(lat,lon,departureAirport) for lat,lon in zip(trackData.latitude,trackData.longitude)]

    runway_segments = [node for node in trackData.airportNodes if 'Rwy' in node]
    runway_numbers = [seg.split('_')[1] for seg in runway_segments]
    rwys_opts,counts = np.unique(runway_numbers,return_counts=True)
    idx = np.argmax(counts)
    dep_runway_no = rwys_opts[idx]
    #Get first Rwy segment in trackData that has takeoff_runway_no in it
    dep_runway_node = [rwy for rwy in runway_segments if dep_runway_no in rwy][0]
    # TODO:Check if takeoff_runway_node is an exit point in NATS
    # If not then find the closest runway exit point in NATS

    return dep_runway_node
    
def get_departure_gate_and_rwy_from_iff(iff_data,callsign,gnatsSim,departureAirport,minRwySpeed=30.):
    import numpy as np
    import random

    trackData=iff_data[3].loc[iff_data[3].callsign==callsign]
    trackData = trackData[trackData.tas >= minRwySpeed].copy()
    trackData.loc[:,'airportNodes']= [get_closest_node_at_airport(lat,lon,departureAirport) for lat,lon in zip(trackData.latitude,trackData.longitude)]

    runway_segments = [node for node in trackData.airportNodes if 'Rwy' in node]
    runway_numbers = [seg.split('_')[1] for seg in runway_segments]
    rwys_opts,counts = np.unique(runway_numbers,return_counts=True)
    idx = np.argmax(counts)
    dep_runway_no = rwys_opts[idx]
    #Get first Rwy segment in trackData that has takeoff_runway_no in it
    dep_runway_node = [rwy for rwy in runway_segments if dep_runway_no in rwy][-1]
    # TODO:Check if takeoff_runway_node is an entry point in NATS
    # If not then find the closest runway entry point in NATS

        # Get the list of unique nodes identified in the track data
    unique_nodes = trackData.airportNodes.unique()
    # Get the first node that starts with a gate
    # TODO: Return an error if trackList is empty
    gateList = [node for node in unique_nodes if 'Gate' in node]
    if gateList:
        dep_gate = gateList[0]
    else:
        gateOpts = gnatsSim.airportInterface.getAllGates(departureAirport)
        gateOpts = [opt for opt in gateOpts if opt.lower().startswith('gate')]
        dep_gate = random.choice(gateOpts)

    rwy_entry,rwy_end=get_landing_rwy_entry_and_end_point(dep_runway_node,departureAirport,domain=['Rwy'])

    return rwy_entry,dep_gate


def random_airport_gate_and_rwy(gnatsSim,airport,arrival=True):

    import random
    if len(airport)==3: airport = 'K'+airport

    gateOpts =list(gnatsSim.airportInterface.getAllGates(airport))
    gateOpts = [opt for opt in gateOpts if opt.lower().startswith('gate')]
    gate = random.choice(gateOpts)

    rwyOpts = list(gnatsSim.airportInterface.getAllRunways(airport))
    rwyOpts = [list(r) for r in rwyOpts]
    rwyOpts = [ent[1] for ent in rwyOpts]
    runway_node = random.choice(rwyOpts)
    
    rwy_entry,rwy_end=get_landing_rwy_entry_and_end_point(runway_node,airport,domain=['Rwy'])

    if arrival: rwy = rwy_end
    if not arrival: rwy = rwy_entry

    return rwy,gate