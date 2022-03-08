from paraatm.io.flightaware import extract_flightaware_data

FA_URL = 'https://flightaware.com/live/flight/AAL302/history/20210806/2240Z/KJFK/KLAX/tracklog'

#FA_URL = 'https://flightaware.com/live/flight/UAL1895/history/20200224/1225Z/KTPA/KIAH/tracklog'
#FA_URL = 'https://flightaware.com/live/flight/UAL1895/history/20220212/1620Z/KIAH/KPHX/tracklog'
#FA_URL = 'https://flightaware.com/live/flight/AAL179/history/20220209/1409Z/KJFK/KSFO/tracklog'
#FA_URL='https://flightaware.com/live/flight/DAL2815/history/20220215/2050Z/KATL/KSAT/tracklog'

#traj_mat, Time_s, Latitude, Longitude, Course_deg, TAS_kts, TAS_mph, Altitude_ft, ROCD = extract_flightaware_data(FA_URL)

traj_mat = extract_flightaware_data(FA_URL)

Time_s = traj_mat['time']
Latitude = traj_mat['latitude']
Longitude = traj_mat['longitude'] 
Course_deg = traj_mat['heading']
TAS_kts = traj_mat['tas']
TAS_mph = traj_mat['tas_mph']
Altitude_ft = traj_mat['altitude']
ROCD = traj_mat['rocd']

#print(Time_s, Latitude, Longitude, Course_deg, TAS_kts, TAS_mph, Altitude_ft, ROCD)