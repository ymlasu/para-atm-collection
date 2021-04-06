import os

GNATS_HOME = os.environ.get('NATS_HOME')

def get_usable_apts_and_rwys(natsSim):
    apts=list(natsSim.airportInterface.getAllAirportCodesInGNATS())
    usable_apts_and_rwys = {}
    dumbAirports = ['KMEM','KSNA']
    for apt in apts:
        if apt[0]=='K' and apt not in dumbAirports:
            rwys = list(natsSim.airportInterface.getAllRunways(apt))
            rwys= [list(rwy) for rwy in rwys]
            rwy_nodes = [rwy[1] for rwy in rwys]
            rwys = [rwy[0] for rwy in rwys]
            approachProcedures = natsSim.terminalAreaInterface.getAllApproaches(apt)
            usableAPs = [ap[1:] for ap in approachProcedures if ap.startswith("I")]

            usableRws = []
            for rwy_node in rwy_nodes:
                rwy_entry,rwy_end=get_landing_rwy_entry_and_end_point(rwy_node,apt,domain=['Rwy'])
                if rwy_entry[2:] in usableAPs:
                    usableRws.append(rwy_entry)
            if usableRws:
                usable_apts_and_rwys.update({apt : usableRws})
    return usable_apts_and_rwys

def get_landing_rwy_entry_and_end_point(landing_rwy_node,airport,domain=['Rwy']):
    import pandas as pd
    import numpy as np
    import random
    
    df = pd.read_csv(GNATS_HOME+'/../GNATS_Server/share/libairport_layout/Airport_Rwy/{}_Nodes_Def.csv'.format(airport))
    df = df.loc[df.domain.isin(domain)]
    df_subs = df.dropna(axis=0,how='any')
    df_subs_id = df_subs.id.values.tolist()

    if np.any(df_subs.id==landing_rwy_node):
        df_subs_landing = df_subs[df_subs.id==landing_rwy_node]
        entry_point = df_subs_landing.refName1.values[0]
        end_point = df_subs_landing.refName2.values[0]
    elif np.any([i.startswith(landing_rwy_node[:6]) for i in df_subs_id]):
        df_subs_landing = df_subs.loc[[i.startswith(landing_rwy_node[:6]) for i in df_subs_id],:]
        entry_point = df_subs_landing.refName1.values[0]
        end_point = df_subs_landing.refName2.values[0]
    else:
        landing_rwy_node = random.choice(df_subs.id.values)
        df_subs_landing = df_subs[df_subs.id==landing_rwy_node]
        entry_point = df_subs_landing.refName1.values[0]
        end_point = df_subs_landing.refName2.values[0]

    return entry_point,end_point

def get_closest_node_at_airport(lat,lon,airport,domain=['Rwy','Gate','Txy','Ramp','Parking']):
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(GNATS_HOME+'/../GNATS_Server/share/libairport_layout/Airport_Rwy/{}_Nodes_Def.csv'.format(airport))
    df = df.loc[df.domain.isin(domain)]
    df['dists']=np.sqrt((df.lat-lat)**2+(df.lon-lon)**2)
    closest_node = df.loc[df.dists.idxmin()]['id']
    return closest_node

def get_list_of_adjacent_nodes(node,airport):
    import pandas as pd
    df = pd.read_csv(GNATS_HOME+'/../GNATS_Server/share/libairport_layout/Airport_Rwy/{}_Nodes_Links.csv'.format(airport))
    df = df.loc[(df['n1.id']==node) | (df['n2.id']==node)]
    adjacent_nodes = [nid for nid in df['n1.id'] if nid != node]+[nid for nid in df['n2.id'] if nid != node]
    return list(set(adjacent_nodes))

def get_adjacent_node_closer_to_runway(nodeList,runwayNode,airport,removed_nodes=[]):
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(GNATS_HOME+'/../GNATS_Server/share/libairport_layout/Airport_Rwy/{}_Nodes_Def.csv'.format(airport))
    rwy_lat = df.loc[df['id']==runwayNode]['lat'].values[0]
    rwy_lon = df.loc[df['id']==runwayNode]['lon'].values[0]

    df = df.loc[df['id'].isin([node for node in nodeList if node not in removed_nodes])].copy()
    print(df.head())
    df['dists']=np.sqrt((df.lat-rwy_lat)**2+(df.lon-rwy_lon)**2)
    closest_node = df.loc[df.dists.idxmin()]['id']
    print('closest node is:',closest_node)
    return closest_node

def get_closest_airport(gnatsSim,lat,lon,asdex_apt):
    import pandas as pd
    import numpy as np

    candApts = list(gnatsSim.airportInterface.getAirportsWithinMiles(lat,lon,100))
    candApts = [apt for apt in candApts if apt.startswith('K') and asdex_apt not in apt]
    lats_lons = [list(gnatsSim.airportInterface.getLocation(apt)) for apt in candApts]
    dists = [np.sqrt((entry[0]-lat)**2+(entry[1]-lon)**2) for entry in lats_lons]
    closest_apt = candApts[np.argmin(dists)]

    print('closest airport is:',closest_apt)
    return closest_apt
    
