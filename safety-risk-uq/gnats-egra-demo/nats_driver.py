"""Driver script to run NATS "demo_2ac" simulation for UQpy

Usage: python nats_driver.py <input_file> <output_file>

The script is intended to be run from the command line, passing two
arguments: the names of the input and output files.  The input file is
a plain text file containing the values of the two variables (waypoint
latitude and longitude coordinates).  The script will then execute
NATS using those inputs as parameters, post-process the results to
compute separation distance, and write the separation distance scalar
result to output file.
"""

import numpy as np
import sys

from paraatm.io.nats import NatsEnvironment

from demo_2ac import TwoAcSim

def calc_sep_distance_vs_time(df):
    """Compute separation distance vs time for two aircraft
    
    Note: this function is for demonstration only.  For simplicity, it
    computes separation distance directly in terms of a latitude and
    longitude.  A better approach may be to first convert latitude and
    longitude into a projection or other more meaningful coordinate
    system for use in the separation distance calculation.

    Parameters
    ----------
    df : DataFrame
        DataFrame obtained from reading NATS simulation results.
        Simulation results should contain data for two aircraft
    
    Returns
    -------
    DataFrame
        DataFrame with time as index and 'sep' column for separation
        distance
    """
    callsigns = df['callsign'].unique()
    def extract_single_ac_data(callsign):
        return df.loc[df['callsign'] == callsign, ['time','latitude','longitude']].set_index('time').resample('1T').mean()
    df1 = extract_single_ac_data(callsigns[0])
    df2 = extract_single_ac_data(callsigns[1])

    new_df = df1.merge(df2, left_index=True, right_index=True)
    # Todo: use a better separation distance calculation.  This is just an L2 distance of the lat/long, but may be better to first compute a projection of the coordinates
    new_df['sep'] = new_df.apply(lambda row: np.sqrt((row['latitude_x'] - row['latitude_y'])**2 + (row['longitude_x'] - row['longitude_y'])**2), axis=1)

    return new_df

if __name__ == '__main__':
    input_file, output_file = sys.argv[1:]

    with open(input_file,'r') as f:
        input_data = [float(x) for x in f.readline().split()]

    print('Running NATS simulation with input:', input_data)
    sim = TwoAcSim()
    # Execute the simulation.  Note that we also explicitly pass a
    # value for output_file, so that we get a record of the output
    # file.  Conveniently, UQpy coordinates each run in a separate
    # directory, so we will have a record of the NATS trajectory
    # output for each individual simulation that is performed.
    df = sim(output_file='nats_output.csv', latitude=input_data[0], longitude=input_data[1])['trajectory']
    sep = calc_sep_distance_vs_time(df)
    result = sep['sep'].iloc[-1]
    print('Separation distance:', result)

    # Running NATS requires a directory change.  By stopping the JVM
    # here, PARA-ATM will automatically return us to the original
    # working directory.  This is necessary so that the output file
    # goes to the right place. (PARA-ATM will automatically stop the
    # JVM when the Python process exits, but we want to do it here
    # prior to writing the output file.)

    NatsEnvironment.stop_jvm()

    with open(output_file, 'w') as f:
        f.write('{}\n'.format(result))
