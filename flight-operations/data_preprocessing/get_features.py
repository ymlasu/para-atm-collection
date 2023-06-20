import os
import sys
from tqdm import tqdm
import argparse
import glob
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import Window
from pkg_resources.extern import packaging


def parse_version(v):
    try:
        return packaging.version.Version(v)
    except packaging.version.InvalidVersion:
        return packaging.version.LegacyVersion(v)


def read_iff_file(filename, record_types=3, callsigns=None, chunksize=50000, encoding='latin-1'):
    """
    Read IFF file and return data frames for requested record types

    From IFF 2.15 specification, record types include:
    2. header
    3. track point
    4. flight plan
    5. data source program
    6. sectorization
    7. minimum safe altitude
    8. flight progress
    9. aircraft state
    Parameters
    ----------
    filename : str
        File to read
    record_types : int, sequence of ints, or 'all'
        Record types to return
    callsigns : None, string, or list of strings
        If None, return records for all aircraft callsigns.
        Otherwise, only return records that match the given callsign
        (in the case of a single string) or match one of the specified
        callsigns (in the case of a list of strings).
    chunksize: int
        Number of rows that are read at a time by pd.read_csv.  This
        limits memory usage when working with large files, as we can
        extract out the desired rows from each chunk, intead of
        reading everything into one large DataFrame and then taking a
        subset.
    encoding: str
        Encoding argument passed on to open and pd.read_csv.  Using
        'latin-1' instead of the default will suppress errors that
        might otherwise occur with minor data corruption.  See
        http://python-notes.curiousefficiency.org/en/latest/python3/text_file_processing.html

    Returns
    -------
    DataFrame or dict of DataFrames
       If record_types is a scalar, return a DataFrame containing the
       data for that record type only.  Otherwise, return a dictionary
       mapping each requested record type to a corresponding DataFrame.
    """
    # Note default record_type of 3 (track point) is used for
    # consistency with the behavior of other functions that expect
    # flight tracking data

    # Determine file format version.  This is in record type 1, which
    # for now we assume to occur on the first line.
    with open(filename, 'r') as f:
        version = parse_version(f.readline().split(',')[2])

    # Columns for each record type, from version 2.6 specification.
    cols = {0: ['recType', 'comment'],
            1: ['recType', 'fileType', 'fileFormatVersion'],
            2: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'acType', 'Orig', 'Dest', 'opsType', 'estOrig', 'estDest'],
            3: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'coord1', 'coord2', 'alt', 'significance', 'coord1Accur', 'coord2Accur', 'altAccur', 'groundSpeed', 'course', 'rateOfClimb', 'altQualifier', 'altIndicator', 'trackPtStatus', 'leaderDir', 'scratchPad', 'msawInhibitInd', 'assignedAltString', 'controllingFac', 'controllingSeg', 'receivingFac', 'receivingSec', 'activeContr', 'primaryContr', 'kybrdSubset', 'kybrdSymbol', 'adsCode', 'opsType', 'airportCode'],
            4: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'acType', 'Orig', 'Dest', 'altcode', 'alt', 'maxAlt', 'assignedAltString', 'requestedAltString', 'route', 'estTime', 'fltCat', 'perfCat', 'opsType', 'equipList', 'coordinationTime', 'coordinationTimeType', 'leaderDir', 'scratchPad1', 'scratchPad2', 'fixPairScratchPad', 'prefDepArrRoute', 'prefDepRoute', 'prefArrRoute'],
            5: ['recType', 'dataSource', 'programName', 'programVersion'],
            6: ['recType', 'recTime', 'Source', 'msgType', 'rectypeCat', 'sectorizationString'],
            7: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'coord1', 'coord2', 'alt', 'significance', 'coord1Accur', 'coord2Accur', 'altAccur', 'msawtype', 'msawTimeCat', 'msawLocCat', 'msawMinSafeAlt', 'msawIndex1', 'msawIndex2', 'msawVolID'],
            8: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'acType', 'Orig', 'Dest', 'depTime', 'depTimeType', 'arrTime', 'arrTimeType'],
            9: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'coord1', 'coord2', 'alt', 'pitchAngle', 'trueHeading', 'rollAngle', 'trueAirSpeed', 'fltPhaseIndicator'],
            10: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'configType', 'configSpec']}

    # For newer versions, additional columns are supported.  However,
    # this code could be commented out, and it should still be
    # compatible with newer versions, but just ignoring the additional
    # columns.
    if version >= parse_version('2.13'):
        cols[2] += ['modeSCode']
        cols[3] += ['trackNumber', 'tptReturnType', 'modeSCode']
        cols[4] += ['coordinationPoint',
                    'coordinationPointType', 'trackNumber', 'modeSCode']
    if version >= parse_version('2.15'):
        cols[3] += ['sensorTrackNumberList', 'spi', 'dvs', 'dupM3a', 'tid']

    # Determine the record type of each row
    with open(filename, 'r', encoding=encoding) as f:
        # An alternative, using less memory, would be to directly
        # create skiprows indices for a particular record type, using
        # a comprehension on enumerate(f); however, that would not
        # allow handling multiple record types.
        line_record_types = [int(line.split(',')[0]) for line in f]

    # Determine which record types to retrieve, and whether the result
    # should be a scalar or dict:
    if record_types == 'all':
        record_types = np.unique(line_record_types)
        scalar_result = False
    elif hasattr(record_types, '__getitem__'):
        scalar_result = False
    else:
        record_types = [record_types]
        scalar_result = True

    if callsigns is not None:
        callsigns = list(np.atleast_1d(callsigns))

    data_frames = dict()
    for record_type in record_types:
        # Construct list of rows to skip:
        skiprows = [i for i, lr in enumerate(
            line_record_types) if lr != record_type]

        # Passing usecols is necessary because for some records, the
        # actual data has extraneous empty columns at the end, in which
        # case the data does not seem to get read correctly without
        # usecols
        if callsigns is None:
            df = pd.concat((chunk for chunk in pd.read_csv(filename, header=None, skiprows=skiprows,
                           names=cols[record_type], usecols=cols[record_type], na_values='?', encoding=encoding, chunksize=chunksize, low_memory=False)), ignore_index=True)
        else:
            df = pd.concat((chunk[chunk['AcId'].isin(callsigns)] for chunk in pd.read_csv(filename, header=None, skiprows=skiprows, names=cols[record_type],
                           usecols=cols[record_type], na_values='?', encoding=encoding, chunksize=chunksize, low_memory=False)), ignore_index=True)

        # For consistency with other PARA-ATM data:
        df.rename(columns={'recTime': 'time',
                           'AcId': 'callsign',
                           'coord1': 'latitude',
                           'coord2': 'longitude',
                           'alt': 'altitude',
                           'rateOfClimb': 'rocd',
                           'groundSpeed': 'tas',
                           'course': 'heading'},
                  inplace=True)

        # if 'time' in df:
        #df['time'] = pd.to_datetime(df['time'], unit='s')
        if 'altitude' in df:
            df['altitude'] *= 100  # Convert 100s ft to ft

        # Store to dict of data frames
        data_frames[record_type] = df

    if scalar_result:
        result = data_frames[record_types[0]]
    else:
        result = data_frames

    return result


def haversine_vectorize(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    m = 6367 * dist * 1000  # 6367 for distance in KM for miles use 3958
    return m


def num_ac_ahead(current_time, dest, data, interval):  # ahead
    tmp_data = data.loc[data['recTime'].between(
        current_time-interval, current_time)]
    tmp_data.loc[:, 'distance'] = haversine_vectorize(
        tmp_data['lon'], tmp_data['lat'], airportsCoord[dest][1], airportsCoord[dest][0])
    return len(tmp_data.loc[tmp_data['distance'].between(in_circl, out_circl)]['acId'].unique())


def num_ac_behind(current_time, dest, data, interval):  # behind
    tmp_data = data.loc[data['recTime'].between(
        current_time, current_time+interval)]
    tmp_data.loc[:, 'distance'] = haversine_vectorize(
        tmp_data['lon'], tmp_data['lat'], airportsCoord[dest][1], airportsCoord[dest][0])
    return len(tmp_data.loc[tmp_data['distance'].between(in_circl, out_circl)]['acId'].unique())


def num_ev_ahead(current_time, dest, data, interval):  # event check range 40-200nm
    tmp_data = data.loc[data['tEv'].between(
        current_time-interval, current_time)]
    tmp_data.loc[:, 'distance'] = haversine_vectorize(
        tmp_data['Lon'], tmp_data['Lat'], airportsCoord[dest][1], airportsCoord[dest][0])
    tmp_data = tmp_data.loc[tmp_data['distance'].between(
        in_circl, 2*out_circl)]
    n_rrt = tmp_data[tmp_data['EvType'] == 'EV_RRT'].shape[0]
    n_loop = tmp_data[tmp_data['EvType'] == 'EV_LOOP'].shape[0]
    n_goa = tmp_data[tmp_data['EvType'] == 'EV_GOA'].shape[0]
    return n_rrt, n_loop, n_goa


if __name__ == '__main__':
    spark = SparkSession.builder\
        .appName("Terminal_Area_Flight_Data_Query")\
        .config("spark.driver.memory", "50g")\
        .config("spark.driver.maxResultSize", "50g")\
        .getOrCreate()

    # # recType3 Data
    # ## Custom schema of the data
    # ### References to IFF_2.15_Specs_Sherlock.doc

    myschema = StructType([
        # 1  //track point record type number
        StructField("recType", ShortType(), True),
        # 2  //seconds since midnigght 1/1/70 UTC
        StructField("recTime", StringType(), True),
        StructField("fltKey", LongType(), True),  # 3  //flight key
        # 4  //digit range from 0 to 7
        StructField("bcnCode", IntegerType(), True),
        StructField("cid", IntegerType(), True),  # 5  //computer flight id
        StructField("Source", StringType(), True),  # 6  //source of the record
        StructField("msgType", StringType(), True),  # 7
        StructField("acId", StringType(), True),  # 8  //call sign
        StructField("recTypeCat", StringType(), True),  # 9
        StructField("lat", DoubleType(), True),  # 10
        StructField("lon", DoubleType(), True),  # 11
        StructField("alt", DoubleType(), True),  # 12  //in 100s of feet
        # 13 //digit range from 1 to 10
        StructField("significance", ShortType(), True),
        StructField("latAcc", DoubleType(), True),  # 14
        StructField("lonAcc", DoubleType(), True),  # 15
        StructField("altAcc", DoubleType(), True),  # 16
        StructField("groundSpeed", IntegerType(), True),  # 17 //in knots
        # 18  //in degrees from true north
        StructField("course", DoubleType(), True),
        # 19  //in feet per minute
        StructField("rateOfClimb", DoubleType(), True),
        # 20  //Altitude qualifier (the “B4 character”)
        StructField("altQualifier", StringType(), True),
        # 21  //Altitude indicator (the “C4 character”)
        StructField("altIndicator", StringType(), True),
        # 22  //Track point status (e.g., ‘C’ for coast)
        StructField("trackPtStatus", StringType(), True),
        # 23  //int 0-8 representing the direction of the leader line
        StructField("leaderDir", IntegerType(), True),
        StructField("scratchPad", StringType(), True),  # 24
        # 25 // MSAW Inhibit Indicator (0=not inhibited, 1=inhibited)
        StructField("msawInhibitInd", ShortType(), True),
        StructField("assignedAltString", StringType(), True),  # 26
        StructField("controllingFac", StringType(), True),  # 27
        StructField("controllingSec", StringType(), True),  # 28
        StructField("receivingFac", StringType(), True),  # 29
        StructField("receivingSec", StringType(), True),  # 30
        # 31  // the active control number
        StructField("activeContr", IntegerType(), True),
        # 32  //The primary(previous, controlling, or possible next)controller number
        StructField("primaryContr", IntegerType(), True),
        # 33  //identifies a subset of controller keyboards
        StructField("kybrdSubset", StringType(), True),
        # 34  //identifies a keyboard within the keyboard subsets
        StructField("kybrdSymbol", StringType(), True),
        # 35  //arrival departure status code
        StructField("adsCode", IntegerType(), True),
        # 36  //Operations type (O/E/A/D/I/U)from ARTS and ARTS 3A data
        StructField("opsType", StringType(), True),
        StructField("airportCode", StringType(), True),  # 37
        StructField("trackNumber", IntegerType(), True),  # 38
        StructField("tptReturnType", StringType(), True),  # 39
        StructField("modeSCode", StringType(), True),  # 40
        # 41 //a list of sensor/track number combinations
        StructField("sensorTrackNumberList", StringType(), True),
        # 42 // representing the Ident feature
        StructField("spi", StringType(), True),
        # 43 // indicate the aircraft is within a suppresion volumn area
        StructField("dvs", StringType(), True),
        # 44 // indicate 2 aircraft have the same mode 3a code
        StructField("dupM3a", StringType(), True),
        # 45 //Aircraft Ident entered by pilot
        StructField("tid", StringType(), True),
    ])

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--date", required=True, type=int,
                    help="IFFdate")
    ap.add_argument("-s", "--sector", required=True, type=str,
                    help="sector")
    args = vars(ap.parse_args())

    date = args['date']
    sector = args['sector']
    # date = '20190801'
    # sector = 'ZTL'

    # load event data file
    ev_file_path = f"/media/ypang6/paralab/Research/data/EV_{sector}/"
    ev_file_names = f'EV_{sector}_{date}*.csv'
    ev_file_dir = glob.glob(ev_file_path+ev_file_names)[0]
    df_ev = pd.read_csv(ev_file_dir)
    cols = ['tMidnightSecs', 'AcId', 'tEv', 'EvType', 'Lat', 'Lon']
    df_ev = df_ev[cols]
    df_ev = df_ev[df_ev["EvType"].isin(["EV_GOA", "EV_LOOP", "EV_RRT"])]
    df_ev["tEv"] = df_ev["tEv"] + df_ev["tMidnightSecs"]
    df_ev["tEv"] = df_ev[["tEv"]].astype(int)
    df_ev = df_ev.drop(['tMidnightSecs'], axis=1)

    # load IFF data file
    file_path = f"/media/ypang6/paralab/Research/data/{sector}/"
    file_names = f'IFF_{sector}_{date}*.csv'
    file_dir = glob.glob(file_path+file_names)[0]

    df = spark.read.csv(file_dir, header=False, sep=",", schema=myschema)
    cols = ['recType', 'recTime', 'acId', 'lat', 'lon', 'alt', 'groundSpeed']
    df_rec3 = df.select(*cols).filter(df['recType'] == 3).withColumn(
        "recTime", df['recTime'].cast(IntegerType()))

    pdf_rec3 = df_rec3.toPandas()

    pd_df_rec2 = pd.DataFrame()
    for file_name in [file_names]:
        pd_df_rec2_tmp = pd.DataFrame()
        pd_df_rec2_tmp = read_iff_file(
            glob.glob(file_path+file_names)[0], record_types=2, chunksize=1e6)
        pd_df_rec2 = pd.concat([pd_df_rec2, pd_df_rec2_tmp])
    #pd_df_rec2 = read_iff_file(file_path, record_types=2, chunksize = 1e6)

    cols_rec2 = ['recType', 'time', 'callsign', 'acType', 'Orig', 'Dest']
    pd_df_2 = pd_df_rec2[['recType', 'time',
                          'callsign', 'acType', 'Orig', 'Dest']]

    #pd.options.display.max_rows = 1000

    # pandas
    pdf_rec3['next_acId'] = pdf_rec3['acId'].shift(-1)
    pdf_rec3['previous_acId'] = pdf_rec3['acId'].shift(1)
    pdf_rec3['init_ac'] = pdf_rec3['acId'] != pdf_rec3['previous_acId']
    pdf_rec3['end_ac'] = pdf_rec3['acId'] != pdf_rec3['next_acId']
    pdf_rec3.drop(columns=['previous_acId', 'next_acId'])

    # pyspark
    Windowspec = Window.orderBy("recType")
    df_rec3 = df_rec3.withColumn(
        'prev_acId', F.lag(df_rec3['acId']).over(Windowspec))
    df_rec3 = df_rec3.withColumn(
        'next_acId', F.lead(df_rec3['acId']).over(Windowspec))
    df_rec3 = df_rec3.withColumn('init_ac', F.when(
        df_rec3.acId == df_rec3.prev_acId, False).otherwise(True))
    df_rec3 = df_rec3.withColumn('end_ac', F.when(
        df_rec3.acId == df_rec3.next_acId, False).otherwise(True))
    df_rec3 = df_rec3.withColumn('index', F.row_number().over(Windowspec))
    df_rec3.drop('prev_acId')
    cols = ['index', 'recType', 'recTime', 'acId', 'lat',
            'lon', 'alt', 'groundSpeed', 'init_ac', 'end_ac']

    # ZTL_airports = ['KATL', 'KCHA', 'KGSO', 'KGSP', 'KHKY', 'KHSV']
    # airportsCoord = {'KATL':(33.6407, -84.4277),
    #                  'KCHA':(35.0374, -85.1970),
    #                  'KSGO':(36.1044, -79.9352),
    #                  'KGSP':(34.8959, -82.2172),
    #                  'KHKY':(35.7422, -81.3893),
    #                  'KHSV':(34.6403, -86.7757)}
    ZTL_airports = ['KATL']
    airportsCoord = {'KATL': (33.6407, -84.4277),
                     }

    schema = StructType([
        StructField('time', StringType(), True),
        StructField('hour', StringType(), True),
        StructField('acId', StringType(), True),
        StructField('lat', DoubleType(), True),
        StructField('lon', DoubleType(), True),
        StructField('alt', DoubleType(), True),
        StructField('Dest', StringType(), True),
        StructField('acType', StringType(), True),
        StructField('arr_time', StringType(), True)
        #StructField('time_100', StringType(), True),
        #StructField('time_40', StringType(), True)
    ])

    circl200 = 370400  # 200nm
    out_circl = 370400/2
    in_circl = 74080  # 40nm

    pd.options.mode.chained_assignment = None  # default='warn'

    columns = ['recTime', 'time', 'round-60', 'round-30', 'round-15', 'acId', 'lat', 'lon', 'alt', 'Dest', 'distance', 'acType', 'groundSpeed', '#AC_10mins_ahead', '#AC_10mins_behind', '#AC_30mins_ahead', '#AC_30mins_behind', '#AC_60mins_ahead', '#AC_60mins_behind',
               'EV_RRT_600', 'EV_RRT_1800', 'EV_RRT_3600', 'EV_LOOP_600', 'EV_LOOP_1800', 'EV_LOOP_3600', 'EV_GOA_600', 'EV_GOA_1800', 'EV_GOA_3600',
               'arr_time', 'time200', 'time100', 'time40']
    pdf3 = pd.DataFrame([], columns=columns)
    for index, row in tqdm(pdf_rec3.iterrows(), total=pdf_rec3.shape[0]):
        if row['init_ac']:
            init_index = index
            timestamp = row.recTime
        if row.end_ac:
            if row.alt <= 25:
                end_index = index
                tmp_pdf = pdf_rec3.iloc[init_index:end_index+1, :]

                tmp_dest = pd_df_2[pd_df_2['time'] == timestamp]['Dest'].to_string(
                    index=False).strip()
                acType = pd_df_2[pd_df_2['time'] == timestamp]['acType'].to_string(
                    index=False).strip()
                if tmp_dest in airportsCoord.keys():
                    tmp_pdf.loc[:, 'Dest'] = tmp_dest
                    tmp_pdf.loc[:, 'acType'] = acType
                    tmp_pdf.loc[:, 'dest_lat'] = airportsCoord[tmp_dest][0]
                    tmp_pdf.loc[:, 'dest_lon'] = airportsCoord[tmp_dest][1]
                    tmp_pdf.loc[:, 'distance'] = haversine_vectorize(
                        tmp_pdf['lon'], tmp_pdf['lat'], tmp_pdf['dest_lon'], tmp_pdf['dest_lat'])

                    #tmp_pdf = tmp_pdf[tmp_pdf['distance'] <= 185200]
                    tmp_pdf = tmp_pdf.loc[tmp_pdf['distance'] <= circl200]
                    if tmp_pdf.shape[0] == 0:
                        pass
                    else:
                        tmp_pdf.loc[:, 'time200'] = tmp_pdf.head(
                            1)['recTime'].values[0]

                    tmp_pdf_200 = tmp_pdf.loc[tmp_pdf['distance'].between(
                        circl200 - 20000, circl200)]  # 200nm-20km --> 200nm
                    # print(tmp_pdf.head(1)['recTime'].values)

                    tmp_pdf = tmp_pdf.loc[tmp_pdf['distance'] <= out_circl]
                    # 40nm <=  Points  <=100nm
                    tmp_pdf = tmp_pdf.loc[tmp_pdf['distance'] >= in_circl]
                    tmp_pdf_out = tmp_pdf.loc[tmp_pdf['distance'].between(
                        out_circl - 20000, out_circl)]  # 100nm-20km --> 100nm
                    tmp_pdf_in = tmp_pdf.loc[tmp_pdf['distance'].between(
                        in_circl, in_circl + 20000)]  # 40nm --> 40nm+20km
                    if not tmp_pdf_in.empty and not tmp_pdf_out.empty and not tmp_pdf_200.empty:
                        tmp_time = pd.to_datetime(tmp_pdf.head(
                            1)['recTime'].values[0]-14400, unit='s')  # UTC-4H time zone
                        # import pdb; pdb.set_trace()
                        tmp_pdf.loc[:, 'time'] = tmp_time
                        #tmp_pdf.loc[:,'time'] = tmp_pdf.head(1)['recTime']
                        tmp_pdf.loc[:, 'round-60'] = tmp_time.hour
                        # round to the 30 minutes interval
                        tmp_pdf.loc[:, 'round-30'] = tmp_time.hour * \
                            2+tmp_time.minute//30
                        # round to the 15 minutes interval
                        tmp_pdf.loc[:, 'round-15'] = tmp_time.hour * \
                            4+tmp_time.minute//15

                        in_time = tmp_pdf.head(1)['recTime'].values[0]
                        out_time = tmp_pdf.tail(1)['recTime'].values[0]
                        #center_time = tmp_pdf.loc[tmp_pdf['distance'].between(out_circl/2-100000, out_circl/2)].head(1)['recTime'].values[0]
                        tmp_pdf.loc[:, 'arr_time'] = out_time - \
                            in_time  # out_time - center_time #in_time
                        tmp_pdf.loc[:, 'time40'] = out_time
                        tmp_pdf.loc[:, 'time100'] = in_time
                        first_time = tmp_pdf.head(1)['recTime'].values[0]
                        number_ahead_600 = num_ac_ahead(
                            in_time, tmp_dest, pdf_rec3, 600)
                        number_behind_600 = num_ac_behind(
                            in_time, tmp_dest, pdf_rec3, 600)
                        number_ahead_1800 = num_ac_ahead(
                            in_time, tmp_dest, pdf_rec3, 600*3)
                        number_behind_1800 = num_ac_behind(
                            in_time, tmp_dest, pdf_rec3, 600*3)
                        number_ahead_3600 = num_ac_ahead(
                            in_time, tmp_dest, pdf_rec3, 600*6)
                        number_behind_3600 = num_ac_behind(
                            in_time, tmp_dest, pdf_rec3, 600*6)
                        tmp_pdf.loc[:, '#AC_10mins_ahead'] = number_ahead_600
                        tmp_pdf.loc[:, '#AC_10mins_behind'] = number_behind_600
                        tmp_pdf.loc[:, '#AC_30mins_ahead'] = number_ahead_1800
                        tmp_pdf.loc[:,
                                    '#AC_30mins_behind'] = number_behind_1800
                        tmp_pdf.loc[:, '#AC_60mins_ahead'] = number_ahead_3600
                        tmp_pdf.loc[:,
                                    '#AC_60mins_behind'] = number_behind_3600

                        rrt_600, loop_600, goa_600 = num_ev_ahead(
                            in_time, tmp_dest, df_ev, 600)
                        rrt_1800, loop_1800, goa_1800 = num_ev_ahead(
                            in_time, tmp_dest, df_ev, 1800)
                        rrt_3600, loop_3600, goa_3600 = num_ev_ahead(
                            in_time, tmp_dest, df_ev, 3600)
                        tmp_pdf.loc[:, 'EV_RRT_600'], tmp_pdf.loc[:, 'EV_RRT_1800'], tmp_pdf.loc[:,
                                                                                                 'EV_RRT_3600'] = rrt_600, rrt_1800, rrt_3600
                        tmp_pdf.loc[:, 'EV_LOOP_600'], tmp_pdf.loc[:, 'EV_LOOP_1800'], tmp_pdf.loc[:,
                                                                                                   'EV_LOOP_3600'] = loop_600, loop_1800, loop_3600
                        tmp_pdf.loc[:, 'EV_GOA_600'], tmp_pdf.loc[:, 'EV_GOA_1800'], tmp_pdf.loc[:,
                                                                                                 'EV_GOA_3600'] = goa_600, goa_1800, goa_3600

                        pdf3 = pd.concat(
                            [pdf3, tmp_pdf[tmp_pdf['recTime'] == first_time][columns]])

                        #print("processed aircraft: {}".format(pdf3.shape[0]),end='\r')
                        sys.stdout.flush()

    print("processed aircraft: {}".format(pdf3.shape[0]), end='\r')
    pdf3.to_csv(os.getcwd()+f'/processed_features/{sector}_{date}.csv')
