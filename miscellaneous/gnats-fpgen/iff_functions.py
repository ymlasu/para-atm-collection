from nats_functions import (get_closest_node_at_airport,
                            get_list_of_adjacent_nodes,
                            get_adjacent_node_closer_to_runway,
                            get_closest_airport,
                            get_landing_rwy_entry_and_end_point,
                            get_usable_apts_and_rwys)

from paraatm.io.gnats import GnatsEnvironment

def get_departure_airport_from_iff(iff_data,callsign,gnatsSim,arrivalAirport=None,flmap=None):
    import random
    origin_opts = []

    asdex_airport = iff_data[3][iff_data[3].callsign==callsign].Source.unique()[0][:3]
    usable_apts_and_rwys = get_usable_apts_and_rwys(gnatsSim)
    allAirports =list(usable_apts_and_rwys.keys())
    
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
    origin_opts = [origin.rjust(4,'K') for origin in origin_opts]
    origin_opts = [origin for origin in origin_opts if origin in allAirports]
    if len(origin_opts)==1:
        origin=origin_opts
    if len(origin_opts)>0:
        origin = list(set(origin_opts))[0]
    if not origin_opts:
        print("No viable origin airport found for {}. Returning random from FlightPlanSelector options.".format(callsign,'K'+asdex_airport))
        fplist=[key for key in flmap if (key.endswith(arrivalAirport) or key.endswith(arrivalAirport[1:]))]
        departOpts = [dep.split('-')[0] for dep in fplist]
        allAirports = [apt[-3:] for apt in allAirports if apt not in [arrivalAirport]]
        origin = random.choice(departOpts)
        origin = origin.rjust(4,'K')
        
    return origin
        

def get_arrival_airport_from_iff(iff_data,callsign,gnatsSim,departureAirport,flmap):
    import random

    dest_opts = []

    asdex_airport = iff_data[3][iff_data[3].callsign==callsign].Source.unique()[0][:3]
    
    usable_apts_and_rwys = get_usable_apts_and_rwys(gnatsSim)
    allAirports =list(usable_apts_and_rwys.keys())

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
    dest_opts = [dest.rjust(4,'K') for dest in dest_opts]
    dest_opts = [dest for dest in dest_opts if dest in allAirports]
    if len(dest_opts)==1:
        dest=dest_opts
    if len(dest_opts)>0:
        dest = list(set(dest_opts))[0]
    if not dest_opts:
        print("No viable destination airport found for {}. Returning random from FlightPlanSelector options.".format(callsign,'K'+asdex_airport))
        fplist=[key for key in flmap if (key.startswith(departureAirport) or key.startswith(departureAirport[1:]))]
        departOpts = [dep.split('-')[1] for dep in fplist]
        allAirports = [apt[-3:] for apt in allAirports if apt not in [departureAirport]]
        departOpts = [dep for dep in departOpts if dep[-3:] in allAirports]
        dest = random.choice(departOpts)
        dest = dest.rjust(4,'K')
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

    return arrival_gate,rwy_end

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

    usable_apts_and_rwys = get_usable_apts_and_rwys(gnatsSim)
    usable_rwys = usable_apts_and_rwys[departureAirport]
    usable_rwy_entries = [list(gnatsSim.airportInterface.getRunwayEnds(departureAirport,rwy))[1] for rwy in usable_rwys]

    runway_segments = [node for node in trackData.airportNodes if 'Rwy' in node]
    #Get first Rwy segment in trackData that has takeoff_runway_no in it
    dep_runway_nodes = [rwy for rwy in runway_segments if rwy in usable_rwy_entries]
    if dep_runway_nodes:
        rwy_idx = usable_rwy_entries.index(dep_runway_nodes[0])
        rwy_name = usable_rwys[rwy_idx]
    else:
        rwy_name = random.choice(usable_rwys)

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

    return dep_gate,rwy_name


def random_airport_gate_and_rwy(gnatsSim,airport,arrival=True):

    import random
    if len(airport)==3: airport = 'K'+airport

    gateOpts =list(gnatsSim.airportInterface.getAllGates(airport))
    gateOpts = [opt for opt in gateOpts if opt.lower().startswith('gate')]
    gate = random.choice(gateOpts)

    usable_apts_and_rwys = get_usable_apts_and_rwys(gnatsSim)
    usable_rwys = usable_apts_and_rwys[airport]
    rwy_name = random.choice(usable_rwys)

    [rwy_entry,rwy_end]=list(gnatsSim.airportInterface.getRunwayEnds(airport,rwy_name))

    #if arrival: rwy = rwy_end
    #if not arrival: rwy = rwy_entry

    return gate,rwy_name