# GNATS Class
#
# Optimal Synthesis Inc.
#
# Oliver Chen
# 11.06.2020
#
# Class FlightPlanSelector
#
# This class is mainly for generating FP_ROUTE string for further usage in a TRX file.
#
# Main function
# generate()
#     This function generates FP_ROUTE string and initial latitude and longitude
#     The program takes different flight plan types and executes corresponding calculation.

import math
from jpype import JPackage

planned_dirname = ""
output_filename = ""

import os,re,sys

def getAlphaCharOnly(entry):
	no_digits = ""
	
	# Iterate through the string, adding non-numbers to the no_digits list
	for i in entry:
		if not i.isdigit():
			no_digits += i
	
	return no_digits
		   

class FlightPlanSelector:
	
	def __init__(self,gnatsSim,fname = "../../GNATS_Server/share/tg/trx/TRX_DEMO_100rec_beta1.5.trx"):
		self.fname = fname
		self.readFile()
		self.FLIGHT_PLAN_TYPE_GATE_TO_GATE = 1
		self.FLIGHT_PLAN_TYPE_RUNWAY_TO_RUNWAY = 2
		self.FLIGHT_PLAN_TYPE_CRUISE = 3
		self.FLIGHT_PLAN_TYPE_CRUISE_TO_GATE = 4
		self.str_readFlightPlan = ""
		self.enroute_fp = ""
		self.selected_sid = ""
		self.selected_star = ""
		self.selected_approach = ""
		self.starting_latitude = ""
		self.starting_longitude = ""

		self.gnatsSim = gnatsSim
	
	def readFile(self):
		if not os.path.exists(self.fname): 
			print('File ',self.fname,' not found.')
			raise IOError

		fid = open(self.fname,'r')
		Lines = fid.readlines()
		self.flmap = {}
		for l in Lines:
			if 'FP_ROUTE' in l:
				l = re.sub('<[^>]+>','', l)								
				l = l.split(' ')
				while '' in l:
					l.remove('')			
				fplan = l[1]	  
				fplan = fplan.replace('<>','')
				fplan = fplan.replace('\n','')
				if fplan[-5] == '/':
					fplan = fplan[:-5]
				oidx = 4
				if fplan[3] == '.':
					oidx = 3						  
				origin = fplan[:oidx]
				didx = 4
				if fplan[-4] == '.':
					didx = 3				
				destination = fplan[-didx:]
				self.flmap[origin+'-'+destination] = fplan
	
	def readFlightPlan(self, origin, destination):
		return_string = ""
		
		if origin == '' or destination == '':
			raise('Please provide origin and destination')
				
		if origin+'-' + destination in self.flmap.keys():
			key = origin+'-' + destination
			return_string = self.flmap[key]
		elif origin[1:]+'-' + destination[1:] in self.flmap.keys():
			key = origin[1:]+'-' + destination[1:]
			return_string = self.flmap[key]
		elif origin+'-' + destination[1:] in self.flmap.keys():
			key = origin+'-' + destination[1:]
			return_string = self.flmap[key]
		elif origin[1:]+'-' + destination in self.flmap.keys():
			key = origin[1:]+'-' + destination
			return_string = self.flmap[key]	
		else:
			print('Origin destination pair not found.\n') 

		# Filter out SID if it is included in the returned string
		tmp_Sids = self.gnatsSim.terminalAreaInterface.getAllSids(origin)
		if not(tmp_Sids is None) :
			for tmp_sid in tmp_Sids:
				tmp_sid_charOnly = getAlphaCharOnly(tmp_sid)
				
				if -1 < return_string.find(tmp_sid_charOnly) :
					tmp_string_part_2 = return_string[return_string.index(".", return_string.find(tmp_sid_charOnly))+1 :]

					return_string = origin + "." + tmp_string_part_2
					break
		
		# Filter out STAR if it is included in the returned string
		tmp_arrivalStars = self.gnatsSim.terminalAreaInterface.getAllStars(destination)
		if not(tmp_arrivalStars is None) :
			for tmp_star in tmp_arrivalStars:
				tmp_star_charOnly = getAlphaCharOnly(tmp_star)
				
				if -1 < return_string.find(tmp_star_charOnly) :
					tmp_string_part_1 = return_string[: return_string.index(tmp_star_charOnly)-1]
					
					return_string = tmp_string_part_1 + "." + destination
					break
		
		return return_string
	
	def distance(self, procedurePoint, enroutePoint):
		lat1, lon1 = procedurePoint
		lat2, lon2 = enroutePoint
		radius = 6371

		dlat = math.radians(lat2-lat1)
		dlon = math.radians(lon2-lon1)
		a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
			* math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
		c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
		d = radius * c

		return d * 0.621371


	def getTerminalProcedures(self, origin, destination, departureRunway, arrivalRunway):
		if len(origin) == 3:
			if origin in ["ANC", "HNL"]:
				origin = "P" + origin
			else:
				origin = "K" + origin

		if len(destination) == 3:
			if destination in ["ANC", "HNL"]:
				destination = "P" + destination
			else:
				destination = "K" + destination

		departureSids = self.gnatsSim.terminalAreaInterface.getAllSids(origin)

		arrivalStars = self.gnatsSim.terminalAreaInterface.getAllStars(destination)
		
		firstEnroutePoint = ""
		
		flightPlanComponents = self.str_readFlightPlan.split(".")[1:]
		
		for component in flightPlanComponents:
			if component.isalpha():
				firstEnroutePoint = component
				break

		# SID
		enrouteWp = self.gnatsSim.terminalAreaInterface.getWaypoint_Latitude_Longitude_deg(firstEnroutePoint)
		distanceToEnroute = 9999999
		optimalSid = ""
		for sid in departureSids:
			legs = self.gnatsSim.terminalAreaInterface.getProcedure_leg_names('SID', sid, origin)
			for leg in legs:
				legWps = self.gnatsSim.terminalAreaInterface.getWaypoints_in_procedure_leg("SID", sid, origin, leg)
				if legWps:
					latLonVal = self.gnatsSim.terminalAreaInterface.getWaypoint_Latitude_Longitude_deg(list(legWps)[-1])
					if not (latLonVal is None) :
						if self.distance(latLonVal, enrouteWp) < distanceToEnroute:
							optimalSid = sid

		# APPROACH
		optimalApproach = ""
		firstApproachWaypoint = ""
		approachProcedures = self.gnatsSim.terminalAreaInterface.getAllApproaches(destination)
		if "I" + arrivalRunway[2:] in approachProcedures:
			optimalApproach = "I" + arrivalRunway[2:]
		elif "I" + arrivalRunway[2:-1] in approachProcedures:
			optimalApproach = "I" + arrivalRunway[2:-1]
		else:
			optimalApproach = "I" + arrivalRunway[2:4]

		# TAKEOFF
		optimalTakeoff = ""
		firstTakeoffWaypoint = ""
		takeoffProcedures = self.gnatsSim.terminalAreaInterface.getAllApproaches(origin)
		if "I" + departureRunway[2:] in takeoffProcedures:
			optimalTakeoff = "I" + departureRunway[2:]
		elif "I" + departureRunway[2:-1] in takeoffProcedures:
			optimalTakeoff = "I" + departureRunway[2:-1]
		
		result_Procedure_leg_names = self.gnatsSim.terminalAreaInterface.getProcedure_leg_names('APPROACH', optimalApproach, destination)
		if not (result_Procedure_leg_names is None) :
			approachLeg = result_Procedure_leg_names[0]
			firstApproachWaypoint = self.gnatsSim.terminalAreaInterface.getWaypoints_in_procedure_leg("APPROACH", optimalApproach, destination, approachLeg)[0]

		# STAR
		distanceToApproach = 9999999
		approachWp = self.gnatsSim.terminalAreaInterface.getWaypoint_Latitude_Longitude_deg(firstApproachWaypoint)
		optimalStar = ""
		for star in arrivalStars:
			legs = self.gnatsSim.terminalAreaInterface.getProcedure_leg_names('STAR', star, destination)
			for leg in legs:
				legWps = self.gnatsSim.terminalAreaInterface.getWaypoints_in_procedure_leg("STAR", star, destination, leg)
				latLonVal = self.gnatsSim.terminalAreaInterface.getWaypoint_Latitude_Longitude_deg(list(legWps)[-1])
				if not (latLonVal is None) :
					if self.distance(latLonVal, approachWp) < distanceToApproach:
						optimalStar = star

		return (optimalSid, optimalStar, optimalApproach, optimalTakeoff)

	def generateDepartureTaxiPlan(self, origin_airport, departure_runway, origin_gate) :
		tmp_surface_plan_string = ""
		
		tmp_node_seq = -1 # Reset
		
		tmp_array_node_data = self.gnatsSim.airportInterface.getLayout_node_data(origin_airport)
		if not(tmp_array_node_data is None) :
			for i in range(0, len(tmp_array_node_data)) :
				if (tmp_array_node_data[i][3] == departure_runway) and (tmp_array_node_data[i][4] == "Entry") :
					tmp_node_seq = tmp_array_node_data[i][0]
					break
		
		if (tmp_node_seq == -1) :
			print("Can't find the departing runway entry point")
		else :
			tmp_array_node_map = self.gnatsSim.airportInterface.getLayout_node_map(origin_airport)
			if not(tmp_array_node_map is None) :
				for i in range(0, len(tmp_array_node_map)) :
					if (tmp_node_seq == tmp_array_node_map[i][1]) :
						departing_runway_entry = tmp_array_node_map[i][0]
						break
				
				# Find the waypoint name of the first point
				for i in range(0, len(tmp_array_node_map)) :
					if (origin_gate == tmp_array_node_map[i][0]) :
						tmp_node_seq_first_point = tmp_array_node_map[i][1]
						break
			
			tmp_design_taxi_plan = self.gnatsSim.airportInterface.get_taxi_route_from_A_To_B("", origin_airport, origin_gate, departing_runway_entry)
			if not(tmp_design_taxi_plan is None) :
				for i in range(0, len(tmp_design_taxi_plan)) :
					if not(i == 0) :
						tmp_surface_plan_string = tmp_surface_plan_string + ", "
					tmp_surface_plan_string = tmp_surface_plan_string + '{"id": "' + tmp_design_taxi_plan[i] + '"}'
			
			if not(tmp_node_seq_first_point == -1) :
				for i in range(0, len(tmp_array_node_data)) :
					if (tmp_node_seq_first_point == tmp_array_node_data[i][0]) :
						self.starting_latitude = str(tmp_array_node_data[i][1])
						self.starting_longitude = str(tmp_array_node_data[i][2])
						break

		return tmp_surface_plan_string
	
	def generateArrivalTaxiPlan(self, destination_airport, arrival_runway, destination_gate) :
		tmp_surface_plan_string = ""
		
		tmp_node_seq = -1 # Reset
		
		tmp_array_node_data = self.gnatsSim.airportInterface.getLayout_node_data(destination_airport)
		if not(tmp_array_node_data is None) :
			for i in range(0, len(tmp_array_node_data)) :
					if not(tmp_array_node_data[i][5] is None) and (-1 < tmp_array_node_data[i][5].find(arrival_runway)) and not(tmp_array_node_data[i][6] is None) and (-1 < tmp_array_node_data[i][6].find("End")) :
						tmp_node_seq = tmp_array_node_data[i][0]
						break
		
		if (tmp_node_seq == -1) :
			print("Can't find the landing runway end point")
		else :
			tmp_array_node_map = self.gnatsSim.airportInterface.getLayout_node_map(destination_airport)
			if not(tmp_array_node_map is None) :
				for i in range(0, len(tmp_array_node_map)) :
					if (tmp_node_seq == tmp_array_node_map[i][1]) :
						landing_runway_end = tmp_array_node_map[i][0]
						break
			
			tmp_design_taxi_plan = self.gnatsSim.airportInterface.get_taxi_route_from_A_To_B("", destination_airport, landing_runway_end, destination_gate)
			if not(tmp_design_taxi_plan is None) :
				for i in range(0, len(tmp_design_taxi_plan)) :
					if not(i == 0) :
						tmp_surface_plan_string = tmp_surface_plan_string + ", "
					tmp_surface_plan_string = tmp_surface_plan_string + '{"id": "' + tmp_design_taxi_plan[i] + '"}'
		
		return tmp_surface_plan_string
	
	def generate(self,
				flight_plan_type,
				origin_airport,
				destination_airport,
				origin_gate,
				destination_gate,
				departure_runway,
				arrival_runway) :
		fp_generated = "" # Reset
				
		self.starting_latitude = "" # Reset
		self.starting_longitude = "" # Reset
		
		tmp_departing_surface_plan_string = "" # Reset
		tmp_landing_surface_plan_string = "" # Reset
		
		departing_runway_entry = "" # Reset
		landing_runway_end = "" # Reset
		
		if flight_plan_type == self.FLIGHT_PLAN_TYPE_GATE_TO_GATE :
			if origin_airport is None or origin_airport == "" or destination_airport is None or destination_airport == "":
				print("Please input origin and destination airports")
				quit()
			
			if departure_runway is None or departure_runway == "" or arrival_runway is None or arrival_runway == "":
				print("Please input departure and arrival runways")
				quit()
			
			if origin_gate is None or origin_gate == "" or destination_gate is None or destination_gate == "":
				print("Please input origin and destination gates")
				quit()
		elif flight_plan_type == self.FLIGHT_PLAN_TYPE_RUNWAY_TO_RUNWAY :
			if origin_airport is None or origin_airport == "" or destination_airport is None or destination_airport == "":
				print("Please input origin and destination airports")
				quit()
			
			if departure_runway is None or departure_runway == "" or arrival_runway is None or arrival_runway == "":
				print("Please input departure and arrival runways")
				quit()
		elif flight_plan_type == self.FLIGHT_PLAN_TYPE_CRUISE :
			if origin_airport is None or origin_airport == "" or destination_airport is None or destination_airport == "":
				print("Please input origin and destination airports")
				quit()
		elif flight_plan_type == self.FLIGHT_PLAN_TYPE_CRUISE_TO_GATE :
			if origin_airport is None or origin_airport == "" or destination_airport is None or destination_airport == "":
				print("Please input origin and destination airports")
				quit()
			
			if arrival_runway is None or arrival_runway == "":
				print("Please input arrival runway")
				quit()
			
			if destination_gate is None or destination_gate == "":
				print("Please input destination gate")
		else :
			print("Please input valid flight plan type")
			quit()
		
		# =================================================
		
		self.str_readFlightPlan = self.readFlightPlan(origin_airport, destination_airport)

		if (-1 < self.str_readFlightPlan.index(".")) and (self.str_readFlightPlan.index(".") < self.str_readFlightPlan.rindex(".")) :
			self.enroute_fp = self.str_readFlightPlan[self.str_readFlightPlan.index(".")+1 : self.str_readFlightPlan.rindex(".")]
		self.selected_takeoff=''
		if flight_plan_type == self.FLIGHT_PLAN_TYPE_GATE_TO_GATE or flight_plan_type == self.FLIGHT_PLAN_TYPE_RUNWAY_TO_RUNWAY :
			self.result_terminalProcedure = self.getTerminalProcedures(origin_airport, destination_airport, departure_runway, arrival_runway)
			if len(self.result_terminalProcedure) == 4 :
				self.selected_sid = self.result_terminalProcedure[0]
				self.selected_star = self.result_terminalProcedure[1]
				self.selected_approach = self.result_terminalProcedure[2]
				self.selected_takeoff = self.result_terminalProcedure[3]
			else:
				self.selected_sid = self.result_terminalProcedure[0]
				self.selected_star = self.result_terminalProcedure[1]
				self.selected_approach = self.result_terminalProcedure[2]
				self.selected_takeoff = ''
			
		if (self.enroute_fp.find("/.") == 0) :
			self.enroute_fp = self.enroute_fp[2:]

		if flight_plan_type == self.FLIGHT_PLAN_TYPE_GATE_TO_GATE :
			tmp_node_seq = -1 # Reset
			tmp_node_seq_first_point = -1 # Reset
			
			# Obtain departing surface plan ---------------
			tmp_departing_surface_plan_string = self.generateDepartureTaxiPlan(origin_airport, departure_runway, origin_gate)
			
			# Obtain landing surface plan -----------------
			tmp_landing_surface_plan_string = self.generateArrivalTaxiPlan(destination_airport, arrival_runway, destination_gate)

		elif flight_plan_type == self.FLIGHT_PLAN_TYPE_RUNWAY_TO_RUNWAY :
			tmp_array_node_data = self.gnatsSim.airportInterface.getLayout_node_data(origin_airport)
			if not(tmp_array_node_data is None) :
				for i in range(0, len(tmp_array_node_data)) :
					if (tmp_array_node_data[i][3] == departure_runway) and (tmp_array_node_data[i][4] == "Entry") :
						self.starting_latitude = str(tmp_array_node_data[i][1])
						self.starting_longitude = str(tmp_array_node_data[i][2])
						break
		
		elif (flight_plan_type == self.FLIGHT_PLAN_TYPE_CRUISE) :
			tmp_first_waypoint = self.enroute_fp[: self.enroute_fp.find(".")]
			tmp_lat_lon = self.gnatsSim.terminalAreaInterface.getWaypoint_Latitude_Longitude_deg(tmp_first_waypoint)
			if not(tmp_lat_lon is None) :
				self.starting_latitude = str(tmp_lat_lon[0])
				self.starting_longitude = str(tmp_lat_lon[1])
		elif (flight_plan_type == self.FLIGHT_PLAN_TYPE_CRUISE_TO_GATE) :
			# Obtain landing surface plan -----------------
			tmp_landing_surface_plan_string = self.generateArrivalTaxiPlan(destination_airport, arrival_runway, destination_gate)
			
			tmp_first_waypoint = self.enroute_fp[: self.enroute_fp.find(".")]
			tmp_lat_lon = self.gnatsSim.terminalAreaInterface.getWaypoint_Latitude_Longitude_deg(tmp_first_waypoint)
			if not(tmp_lat_lon is None) :
				self.starting_latitude = str(tmp_lat_lon[0])
				self.starting_longitude = str(tmp_lat_lon[1])
		
		# =================================================
		# Combine the final returning value
		fp_generated = origin_airport
		
		fp_generated = fp_generated + ".<"
		if not(tmp_departing_surface_plan_string == "") :
			fp_generated = fp_generated + tmp_departing_surface_plan_string
		fp_generated = fp_generated + ">"
		
		if not(departure_runway == "") :
			fp_generated = fp_generated + "." + departure_runway

		if not(self.selected_takeoff==""):
			fp_generated = fp_generated + "." + self.selected_takeoff

		if (flight_plan_type == self.FLIGHT_PLAN_TYPE_GATE_TO_GATE) or (flight_plan_type == self.FLIGHT_PLAN_TYPE_RUNWAY_TO_RUNWAY) :
			if not(self.selected_sid == "") :
				fp_generated = fp_generated + "." + self.selected_sid
		
		fp_generated = fp_generated + "." + self.enroute_fp

		if (flight_plan_type == self.FLIGHT_PLAN_TYPE_GATE_TO_GATE) or (flight_plan_type == self.FLIGHT_PLAN_TYPE_RUNWAY_TO_RUNWAY) or (flight_plan_type == self.FLIGHT_PLAN_TYPE_CRUISE_TO_GATE) :
			if not(self.selected_star == "") :
				fp_generated = fp_generated + "." + self.selected_star
			
			if not(self.selected_approach == "") :
				fp_generated = fp_generated + "." + self.selected_approach
		
		if not(arrival_runway == "") :
			fp_generated = fp_generated + "." + arrival_runway
		
		fp_generated = fp_generated + ".<"
		if not(tmp_landing_surface_plan_string == "") :
			fp_generated = fp_generated + tmp_landing_surface_plan_string
		fp_generated = fp_generated + ">"
		
		fp_generated = fp_generated + "." + destination_airport
		
		clsGeometry = JPackage('com').osi.util.Geometry
		
		if not(self.starting_latitude.strip() == "") and not(self.starting_longitude.strip() == "") :
			# Convert latitude/longitude degree string to degree-minute-second format
			self.starting_latitude = clsGeometry.convertLatLonDeg_to_degMinSecString(self.starting_latitude)
			self.starting_longitude = clsGeometry.convertLatLonDeg_to_degMinSecString(self.starting_longitude)
		
		return (fp_generated, self.starting_latitude, self.starting_longitude)
