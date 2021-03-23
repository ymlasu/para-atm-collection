import numpy as np
import pandas as pd

from iff_functions import (check_if_flight_has_departed,check_if_flight_landing_in_dataset)
from nats_functions import get_closest_node_at_airport

def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60.
    m = int(md)
    sd = (md - m) * 60.
    s = int(sd)
    dec = (sd-s)
    dms_str = str(d)+str(m).zfill(2)+str(s).zfill(2)+str(np.round(dec,3)).strip('0')
    return dms_str

def get_closest_runway_id(all_airport_runways,rwy_seg_id):
    options = [[rwy,rwy_id] for rwy,rwy_id in all_airport_runways if rwy_id.split('_')[1]==rwy_seg_id.split('_')[1]]
    int_of_seg = int(rwy_seg_id.split('_')[-1])
    int_of_opts = np.array([int(seg_id.split('_')[-1]) for rwy,seg_id in options])
    idx = np.argmin(abs(int_of_opts-int_of_seg))
    return options[idx][1]
    
def get_runway_ends_from_track_data(trackData,airport,runways_at_airport,gnatsSim,minRwySpeed=30.):
    trackData = trackData[trackData.tas >= minRwySpeed].copy()
    trackData.loc[:,'airportNodes']=[get_closest_node_at_airport(lat,lon,airport) for lat,lon in zip(trackData.latitude,trackData.longitude)]
    
    runway_segments = [node for node in trackData.airportNodes if 'Rwy' in node]
    runway_numbers = [seg.split('_')[1] for seg in runway_segments]
    rwys_opts,counts = np.unique(runway_numbers,return_counts=True)
    idx = np.argmax(counts)
    runway_no = rwys_opts[idx]
    runway_segment_id = [rwy_id for rwy_id in runway_segments if runway_no in rwy_id.split('_')[1]][0]
    
    #Get first Rwy segment in trackData that has takeoff_runway_no in it
    runway_segment_id = get_closest_runway_id(runways_at_airport,runway_segment_id)
    runway = [[rwy,rwy_id] for rwy,rwy_id in runways_at_airport if runway_no in rwy_id and runway_segment_id==rwy_id][0]
    print('Runway for airport {} is {}'.format(airport,runway))

    runway_ends = list(gnatsSim.airportInterface.getRunwayEnds(airport,runway[0]))
    return runway_ends

def get_lat_lon_from_id(node_id,node_data,node_map):
    node_no = [node for node_name,node in node_map if node_name==node_id][0]
    
    lat,lon = [[lat,lon] for node,lat,lon,rwy_entry,n1, rwy_exit, n2 in node_data if node==node_no][0]

    return lat,lon

def get_fp_runway(trackData,airport,gnatsSim):

    node_data = [list(entry) for entry in list(gnatsSim.airportInterface.getLayout_node_data(airport))]
    node_map =  [list(entry) for entry in list(gnatsSim.airportInterface.getLayout_node_map(airport))]
    all_airport_runways = gnatsSim.airportInterface.getAllRunways(airport)

    airport_alt = get_airport_altitude_from_gnats(gnatsSim,airport)
    trackData = trackData[trackData.altitude < airport_alt+50.].copy()
    
    rwy_entry_id, rwy_exit_id = get_runway_ends_from_track_data(trackData,airport,all_airport_runways,gnatsSim)
    lat_entry,lon_entry = get_lat_lon_from_id(rwy_entry_id,node_data,node_map)
    lat_exit,lon_exit = get_lat_lon_from_id(rwy_exit_id,node_data,node_map)
    lat_entry = dd2dms(lat_entry)
    lon_entry = dd2dms(lon_entry)
    lat_exit = dd2dms(lat_exit)
    lon_exit = dd2dms(lon_exit)
    
    fp_runway = '{{"lat": "{0}", "lon": "{1}"}}, {{"lat": "{2}", "lon": "{3}"}}'.format(lat_entry,lon_entry,lat_exit,lon_exit)

    print('Checkpoint 1')
    
    return fp_runway,rwy_entry_id,rwy_exit_id

def get_airport_altitude_from_gnats(gnatsSim,airport):
    airportElevation=gnatsSim.airportInterface.select_airport(airport).getElevation()
    return airportElevation


def make_fp_origin(gnatsSim,origin):
    origin_lat_lon = list(gnatsSim.airportInterface.getLocation(origin))
    origin_lat_dms = dd2dms(origin_lat_lon[0])
    origin_lon_dms = dd2dms(origin_lat_lon[1])
    origin_alt = get_airport_altitude_from_gnats(gnatsSim,origin)
    full_name = gnatsSim.airportInterface.getFullName(origin).strip(' ')

    fp_origin ='{{"ap_code": "{0}", "ap_name": "{1}", "lat": "{2}", "lon":"{3}", "alt": {4}}}'.format(origin,full_name,origin_lat_dms, origin_lon_dms, np.round(origin_alt,1))
    return fp_origin

def make_fp_dest(gnatsSim,dest):
    dest_lat_lon = list(gnatsSim.airportInterface.getLocation(dest))
    dest_lat_dms = dd2dms(dest_lat_lon[0])
    dest_lon_dms = dd2dms(dest_lat_lon[1])
    dest_alt = get_airport_altitude_from_gnats(gnatsSim,dest)
    full_name = gnatsSim.airportInterface.getFullName(dest).strip(' ')

    fp_dest = '{{"ap_code": "{0}", "ap_name": "{1}", "lat": "{2}", "lon": "{3}", "alt": {4}}}'.format(dest,gnatsSim.airportInterface.getFullName(dest).strip(' '),dest_lat_dms, dest_lon_dms, np.round(dest_alt,1))

    return fp_dest

def get_gate_to_runway_entry(trackData,origin,runway,gnatsSim):
    airport_alt = get_airport_altitude_from_gnats(gnatsSim,origin)
    trackData = trackData[trackData.altitude.between(airport_alt-10.,airport_alt+10.)].copy()
    trackData.loc[:,'airportNodes']= [get_closest_node_at_airport(lat,lon,origin) for lat,lon in zip(trackData.latitude,trackData.longitude)]
    types = {'Rwy':'runway','Txy':'taxiway','Gate':'gate','Ramp':'ramp','Park':'Txy'}

    lat_lons = ['{{"type": "{0}", "lat": "{1}", "lon": "{2}", "alt": {3}}}'.format(types[aptNode.split('_')[0]],lat,lon,np.round(airport_alt,1)) for aptNode,lat,lon,alt in trackData.loc[:,['airportNodes','latitude_dms','longitude_dms','altitude']].values]

    return ', '.join(lat_lons)

def get_runway_to_gate_entry(trackData,dest,runway,gnatsSim):
    airport_alt = get_airport_altitude_from_gnats(gnatsSim,dest)
    trackData = trackData[trackData.altitude.between(airport_alt-10.,airport_alt+10.)].copy()
    #trackData = trackData.iloc[::10,:].copy()
    trackData.loc[:,'airportNodes']= [get_closest_node_at_airport(lat,lon,dest) for lat,lon in zip(trackData.latitude,trackData.longitude)]
    types = {'Rwy':'runway','Txy':'taxiway','Gate':'gate','Ramp':'ramp','Park':'taxiway'}

    lat_lons = ['{{"type": "{0}", "lat": "{1}", "lon": "{2}", "alt": {3}}}'.format(types[aptNode.split('_')[0]],lat,lon,np.round(airport_alt,1)) for aptNode,lat,lon,alt in trackData.loc[:,['airportNodes','latitude_dms','longitude_dms','altitude']].values]

    return ', '.join(lat_lons)

def get_flight_phase(alt,airport_alt,takeoff=True):
    if takeoff:
        if alt > airport_alt and alt < 800.: phase='TAKEOFF'
        elif alt >=800. and alt < 10000.: phase = 'CLIMBOUT'
        elif alt >= 10000. and alt < 30000: phase = 'CLIMB_TO_CRUISE_ALTITUDE'
        elif alt >= 30000.: phase = 'CRUISE'

    if not takeoff:
        if alt > airport_alt and alt < 800.: phase='FINAL_APPROACH'
        elif alt >=800. and alt < 10000.: phase = 'APPROACH'
        elif alt >= 20000. and alt < 30000: phase = 'INITIAL_DESCENT'
        elif alt >= 30000.: phase = 'CRUISE'

    return phase

def add_points_to_destination(trackData,origin_alt,dest_alt,dest_lat_lon,takeoff=True):
    curr_phase = get_flight_phase(trackData.altitude.values[-1],origin_alt,takeoff=takeoff)

    while curr_phase != 'CRUISE':
        if curr_phase == 'TAKEOFF':
            rate_of_climb = 2000
            angle_of_attack = 10
            curr_phase = 'CLIMBOUT'
            print('Checkpoint 5a')
            trackData = trackData.append(trackData.loc[len(trackData)-1],ignore_index=True).copy()
            print('Checkpoint 5b')
            trackData.loc[len(trackData)-1,['altitude','phase']]=[800.,curr_phase]         
            
        if curr_phase == 'CLIMBOUT':
            rate_of_climb = 1500
            angle_of_attack = 8
            curr_phase = 'CLIMB_TO_CRUISE_ALTITUDE'
            print('Checkpoint 5c')
            s = trackData.loc[len(trackData)-1,:].copy()
            trackData.append(s,ignore_index=True)
            print('Checkpoint 5d')
            trackData.loc[len(trackData)-1,['altitude','phase']]=[10000.,curr_phase] 
            
        if curr_phase == 'CLIMB_TO_CRUISE_ALTITUDE':
            rate_of_climb = 1000
            angle_of_attack = 5
            curr_phase = 'CRUISE'
            print('Checkpoint 5e')
            trackData.append(trackData.loc[len(trackData)-1,:],ignore_index=True)
            print('Checkpoint 5f')
            trackData.loc[len(trackData)-1,['altitude','phase']]=[30000.,curr_phase] 

    if curr_phase == 'CRUISE':
        print('Checkpoint 5g')
        trackData.append(trackData.loc[len(trackData)-1,:],ignore_index=True)
        trackData.loc[len(trackData)-1,'latitude']=dest_lat_lon[0]
        trackData.loc[len(trackData)-1,'longitude']=dest_lat_lon[1]
        trackData.loc[len(trackData)-1,'latitude_dms']=dd2dms(dest_lat_lon[0])
        trackata.loc[len(trackData)-1,'longitude_dms']=dd2dms(dest_lat_lon[1])
              
    return trackData         
        
def get_waypoints(trackData,origin,dest,gnatsSim,takeoff=True):

    if takeoff:
        origin_alt = get_airport_altitude_from_gnats(gnatsSim,origin)
        trackData = trackData[trackData.altitude>origin_alt+10].copy()
        dest_alt = get_airport_altitude_from_gnats(gnatsSim,dest)
        dest_lat_lon = list(gnatsSim.airportInterface.getLocation(dest))
        print('Checkpoint 5')
        trackData = add_points_to_destination(trackData,origin_alt,dest_alt,dest_lat_lon,takeoff=takeoff)
        print('Checkpoint 6')
        trackData.loc[:,'phase']= [get_flight_phase(alt,origin_alt,takeoff=takeoff) for alt in trackData.altitude.values]
        
        
    else:
        dest_alt = get_airport_altitude_from_gnats(gnatsSim,dest)
        trackData = trackData[trackData.altitude>dest_alt+10].copy()
        trackData.loc[:,'phase']= [get_flight_phase(alt,dest_alt,takeoff=takeoff) for alt in trackData.altitude.values]
    
    waypoints = ['{{"wp_name": "Waypoint {0}", "lat": "{1}", "lon": "{2}", "alt": {3}, "phase": "{4}"}}'.format(idx+1,lat,lon,np.round(alt,1),phase) for idx,[lat,lon,alt,phase] in zip(range(len(trackData)),trackData.loc[:,['latitude_dms','longitude_dms','altitude','phase']].values)]

    return ', '.join(waypoints)

def get_fp_route_from_iff_geo(gnatsSim,trackData,origin,dest,flightTakenOff,flightLandingInDataset):

    fp_origin = make_fp_origin(gnatsSim,origin)
    fp_dest = make_fp_dest(gnatsSim,dest)

    if not flightTakenOff:
        fp_runway,rwy_entry_id,rwy_exit_id = get_fp_runway(trackData,origin,gnatsSim)
        print('Checkpoint 3')                         
        fp_gate_to_runway = get_gate_to_runway_entry(trackData,origin,rwy_entry_id,gnatsSim)
        print('Checkpoint 4')
        fp_runway_to_waypoints = get_waypoints(trackData,origin,dest,gnatsSim,takeoff=True)
        print('Checkpoint 5')
        fp_route = fp_origin+'.<'+fp_gate_to_runway+'>.RW<'+fp_runway+'>.<'+fp_runway_to_waypoints+'>.RW<>.<>.'+fp_dest
        print('Checkpoint 6')
    elif flightTakenOff and flightLandingInDataset:
        fp_runway,rwy_entry_id,rwy_exit_id =get_fp_runway(trackData,dest,gnatsSim)
        print('Checkpoint 5')
        fp_runway_to_gate = get_runway_to_gate_entry(trackData,dest,rwy_exit_id,gnatsSim)
        fp_waypoints_to_runway = get_waypoints(trackData,origin,dest,gnatsSim,takeoff=False)
        
        fp_route = fp_origin+'<>.RW<>.<'+fp_waypoints_to_runway+'>.RW<'+fp_runway+'>.<'+fp_runway_to_gate+'>.<'+fp_dest
        print('Checkpoint 6')
    else:
        fp_waypoints = get_waypoints(trackData,origin,dest,gnatsSim,takeoff=False)
        fp_route = fp_origin+'<>.RW<>.<'+fp_waypoints+'>.RW<>.<>.'+fp_dest   
    
    return fp_route

def write_trx_geo(iff_data,callsign,departureAirport,arrivalAirport,gnatsSim,trx_fname,mfl_fname):
    flightTakenOff = check_if_flight_has_departed(iff_data,callsign,gnatsSim,departureAirport)
    flightLandingInDataset = check_if_flight_landing_in_dataset(iff_data,callsign,gnatsSim,arrivalAirport)

    trackData = iff_data[3][iff_data[3].callsign==callsign].copy()
    trackData.loc[:,'latitude_dms']=[dd2dms(lat) for lat in trackData.latitude]
    trackData.loc[:,'longitude_dms']=[dd2dms(lon) for lon in trackData.longitude]
    init = trackData.iloc[0,:]
    actype = iff_data[2][iff_data[2].callsign==callsign].iloc[0,:].acType
    if actype == 'NaN'or actype == None:
        actype = 'B733'
    ts = init.time
    cs = init.callsign
    origin = departureAirport
    dest = arrivalAirport
    lat_dms = init.latitude_dms
    lon_dms = init.longitude_dms
    if flightTakenOff:
        alt = init.altitude/100.
    else:
        alt = np.round(get_airport_altitude_from_gnats(gnatsSim,origin),1)/100.
    spd = init.tas
    hdg = init.heading

    fp_route = get_fp_route_from_iff_geo(gnatsSim,trackData,origin,dest,flightTakenOff,flightLandingInDataset)
    
    f = open('{}'.format(trx_fname),"a+")
    f.write('TRACK_TIME {}\n'.format(ts.asm8.astype(np.int64)//10**9))
    f.write('TRACK {0} {1} {2} {3} {4} {5} {6}\n'.format(cs,actype,lat_dms,lon_dms,np.round(spd,0),np.round(alt,2),np.round(hdg,0)))
    f.write('    FP_ROUTE {}\n\n'.format(fp_route))
    f.close()
    
    f = open('{}'.format(mfl_fname),"a+")
    f.write('{} 330\n'.format(cs))
    f.close()
    
    return fp_route


