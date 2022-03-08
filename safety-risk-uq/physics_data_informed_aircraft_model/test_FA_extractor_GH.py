from paraatm.io.flightaware import extract_flightaware_data

FA_URL = 'https://flightaware.com/live/flight/AAL302/history/20210806/2240Z/KJFK/KLAX/tracklog'

#FA_URL = 'https://flightaware.com/live/flight/UAL1895/history/20200224/1225Z/KTPA/KIAH/tracklog'
#FA_URL = 'https://flightaware.com/live/flight/UAL1895/history/20220212/1620Z/KIAH/KPHX/tracklog'
#FA_URL = 'https://flightaware.com/live/flight/AAL179/history/20220209/1409Z/KJFK/KSFO/tracklog'
#FA_URL='https://flightaware.com/live/flight/DAL2815/history/20220215/2050Z/KATL/KSAT/tracklog'

#traj_mat, Time_s, Latitude, Longitude, Course_deg, TAS_kts, TAS_mph, Altitude_ft, ROCD = extract_flightaware_data(FA_URL)

traj_mat = extract_flightaware_data(FA_URL)

Time_s = traj_mat['Time_s']
Latitude = traj_mat['Latitude']
Longitude = traj_mat['Longitude'] 
Course_deg = traj_mat['Course_deg']
TAS_kts = traj_mat['TAS_kts']
TAS_mph = traj_mat['TAS_mph']
Altitude_ft = traj_mat['Altitude_ft']
ROCD = traj_mat['ROCD']

#print(Time_s, Latitude, Longitude, Course_deg, TAS_kts, TAS_mph, Altitude_ft, ROCD)