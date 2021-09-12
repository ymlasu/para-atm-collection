# Set spark environments
import os
os.environ["SPARK_HOME"] = '/home/ypang6/spark-3.1.1-bin-hadoop3.2'
os.environ["PYTHONPATH"] = '/home/ypang6/anaconda3/bin/python3.8'
os.environ['PYSPARK_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.8'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.8'

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
import glob, argparse
from pyspark.ml.feature import StringIndexer


def load_schema():
    myschema = StructType([
        StructField("recType", ShortType(), True),  # 1  //track point record type number
        StructField("recTime", StringType(), True),  # 2  //seconds since midnigght 1/1/70 UTC
        StructField("fltKey", LongType(), True),  # 3  //flight key
        StructField("bcnCode", IntegerType(), True),  # 4  //digit range from 0 to 7
        StructField("cid", IntegerType(), True),  # 5  //computer flight id
        StructField("Source", StringType(), True),  # 6  //source of the record
        StructField("msgType", StringType(), True),  # 7
        StructField("acId", StringType(), True),  # 8  //call sign
        StructField("recTypeCat", IntegerType(), True),  # 9
        StructField("lat", DoubleType(), True),  # 10
        StructField("lon", DoubleType(), True),  # 11
        StructField("alt", DoubleType(), True),  # 12  //in 100s of feet
        StructField("significance", ShortType(), True),  # 13 //digit range from 1 to 10
        StructField("latAcc", DoubleType(), True),  # 14
        StructField("lonAcc", DoubleType(), True),  # 15
        StructField("altAcc", DoubleType(), True),  # 16
        StructField("groundSpeed", IntegerType(), True),  # 17 //in knots
        StructField("course", DoubleType(), True),  # 18  //in degrees from true north
        StructField("rateOfClimb", DoubleType(), True),  # 19  //in feet per minute
        StructField("altQualifier", StringType(), True),  # 20  //Altitude qualifier (the “B4 character”)
        StructField("altIndicator", StringType(), True),  # 21  //Altitude indicator (the “C4 character”)
        StructField("trackPtStatus", StringType(), True),  # 22  //Track point status (e.g., ‘C’ for coast)
        StructField("leaderDir", IntegerType(), True),  # 23  //int 0-8 representing the direction of the leader line
        StructField("scratchPad", StringType(), True),  # 24
        StructField("msawInhibitInd", ShortType(), True),  # 25 // MSAW Inhibit Indicator (0=not inhibited, 1=inhibited)
        StructField("assignedAltString", StringType(), True),  # 26
        StructField("controllingFac", StringType(), True),  # 27
        StructField("controllingSec", StringType(), True),  # 28
        StructField("receivingFac", StringType(), True),  # 29
        StructField("receivingSec", StringType(), True),  # 30
        StructField("activeContr", IntegerType(), True),  # 31  // the active control number
        StructField("primaryContr", IntegerType(), True),
        # 32  //The primary(previous, controlling, or possible next)controller number
        StructField("kybrdSubset", StringType(), True),  # 33  //identifies a subset of controller keyboards
        StructField("kybrdSymbol", StringType(), True),  # 34  //identifies a keyboard within the keyboard subsets
        StructField("adsCode", IntegerType(), True),  # 35  //arrival departure status code
        StructField("opsType", StringType(), True),  # 36  //Operations type (O/E/A/D/I/U)from ARTS and ARTS 3A data
        StructField("airportCode", StringType(), True),  # 37
        StructField("trackNumber", IntegerType(), True),  # 38
        StructField("tptReturnType", StringType(), True),  # 39
        StructField("modeSCode", StringType(), True)  # 40
    ])
    return myschema


def load_data(date):
    file_path = glob.glob("/media/ypang6/paralab/Research/data/ATL/IFF_ATL+ASDEX_{}*.csv".format(date))[0]
    df = spark.read.csv(file_path, header=False, sep=",", schema=iff_schema)
    return df


if __name__ == '__main__':

    spark = SparkSession.\
        builder.\
        master("local[*]").\
        appName("Sector_IFF_Parser").\
        config("spark.serializer", KryoSerializer.getName).\
        config("spark.kryo.registrator", SedonaKryoRegistrator.getName) .\
        config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.0.0-incubating,org.datasyslab:geotools-wrapper:geotools-24.0") .\
        getOrCreate()

    SedonaRegistrator.registerAll(spark)
    sc = spark.sparkContext
    iff_schema = load_schema()

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--date", required=True, type=int,
                    help="IFFdate")
    ap.add_argument("-a", "--duration", required=True, type=int,
                    help="time range in hours")
    ap.add_argument("-r", "--range", required=True, type=float,
                    help="1/2 length of square")
    args = vars(ap.parse_args())

    date = args['date']
    df = load_data(date)

    # select columns
    cols = ['recType', 'recTime', 'acId', 'lat', 'lon', 'alt']
    df = df.select(*cols).filter(df['recType'] == 3).withColumn("recTime", df['recTime'].cast(IntegerType()))

    # time window to query
    duration = args['duration']  # hours
    t_start = 1564668000 + (date - 20190801) * 24 * 3600  # start from June 1st, 2pm, 2019, UTC
    t_end = t_start + 3600 * duration

    # register pyspark df in SQL
    df.registerTempTable("pointtable")

    # create shape column in geospark
    spatialdf = spark.sql(
      """
      SELECT ST_Point(CAST(lat AS Decimal(24, 20)), CAST(lon AS Decimal(24, 20))) AS geom, recTime, acId, alt
      FROM pointtable
      WHERE recTime>={} AND recTime<={}
      """.format(t_start, t_end))

    spatialdf.createOrReplaceTempView("spatialdf")

    katl = [33.6366996, -84.4278640, 11]  #https://www.airnav.com/airport/katl
    r = args['range']  # rectangular query

    # ST_PolygonFromEnvelope (MinX:decimal, MinY:decimal, MaxX:decimal, MaxY:decimal, UUID1, UUID2, ...)
    df_result = spark.sql(
        """
          SELECT *
          FROM spatialdf
          WHERE ST_Contains(ST_PolygonFromEnvelope({}, {}, {}, {}), geom) AND alt>{}
        """.format(katl[0] - r, katl[1] - r, katl[0] + r, katl[1] + r, katl[2]))

    # create relevant timestamp column
    df_result.createOrReplaceTempView("spatialdf")
    df = spark.sql(
        """
            SELECT acId, recTime-{} AS t, geom, alt
            FROM spatialdf
        """.format(t_start)
    )

    indexer = StringIndexer(inputCol="acId", outputCol="FacId")
    df_new = indexer.fit(df).transform(df).drop('acId')
    df_new.createOrReplaceTempView("spatialdf")

    df = spark.sql(
        """
            SELECT t, CAST(FacId AS Integer), ST_X(geom) as lat, ST_Y(geom) as lon, alt
            FROM spatialdf
        """
    )
    
    # RESAMPLE TIMESERIES WITH INTERVAL dt
    dt = 5
    t_interval = list(range(0, 3600 * duration, dt))
    df = df[df.t.isin(t_interval)]
    
    # save data
    outdir1 = './processed_data/radius_{}/duration_{}/3d/{}'.format(r, duration, date)
    outdir2 = './processed_data/radius_{}/duration_{}/2d/{}'.format(r, duration, date)
    if not os.path.exists(outdir1):
        try:
            os.makedirs(outdir1)
        except:
            raise OSError("Can't create destination directory (%s)!" % (outdir1))
    if not os.path.exists(outdir2):
        try:
            os.makedirs(outdir2)
        except:
            raise OSError("Can't create destination directory (%s)!" % (outdir2))

    # save data with altitude dimension
    csv_name = os.path.join(outdir1, 'true_pos_.csv')
    df.toPandas().T.to_csv(csv_name, sep=',', index=False, header=False)

    # save data without altitude dimension
    csv_name = os.path.join(outdir2, 'true_pos_.csv')
    df.drop('alt').toPandas().T.to_csv(csv_name, sep=',', index=False, header=False)
