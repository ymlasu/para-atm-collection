import os

GNATS_HOME = os.environ.get('GNATS_HOME')

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
    df = pd.read_csv(GNATS_HOME+'../GNATS_Server/share/libairport_layout/Airport_Rwy/{}_Nodes_Links.csv'.format(airport))
    df = df.loc[(df['n1.id']==node) | (df['n2.id']==node)]
    adjacent_nodes = [nid for nid in df['n1.id'] if nid != node]+[nid for nid in df['n2.id'] if nid != node]
    return list(set(adjacent_nodes))

def get_adjacent_node_closer_to_runway(nodeList,runwayNode,airport,removed_nodes=[]):
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(GNATS_HOME+'../GNATS_Server/share/libairport_layout/Airport_Rwy/{}_Nodes_Def.csv'.format(airport))
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
    
