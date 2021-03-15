#!/usr/bin/env python
"""
Plots generated flight plan using Google Maps API.

Author: Hari Iyer
Date: 01/12/2019
"""

import math
import os
import csv
from glob import glob

from distutils.spawn import find_executable

# Please enter your Google map API key
GOOGLE_MAP_API_KEY = "AIzaSyA0uZLor9GF8qdQS1EHBtE0xN0UZCtJJB4"

def plotOnGoogleMap(FLIGHT_CALLSIGN, csvFile):
    '''
    Plotting flight plan on Google Map.
    '''

    #Get CSV with Geo-cordinates, parsing them into Google Maps format
    gnatsStandaloneDirName = glob("../*GNATS_Standalone*/")[0]

    latLonFile = list(csv.reader(open(gnatsStandaloneDirName + "/" + csvFile), delimiter=','))
        
    latLonData = []
    
    startIndex = 0

    for flight in FLIGHT_CALLSIGN:
        latLonSubData = []
        for row in latLonFile:
            if (len(row) > 3):
                time, latitude, longitude= row[0], row[1], row[2]
                if (row[0] == "AC" and row[2] == flight):
                    startIndex = latLonFile.index(row) + 1
                    
        for index in range(startIndex, len(latLonFile)):
            if (latLonFile[index] == []):
                break
            time, latitude, longitude= latLonFile[index][0], latLonFile[index][1], latLonFile[index][2]
            if [float(latitude), float(longitude)] != [0.0,0.0]:
                latLonSubData.append({'lat': float(latitude), 'lng': float(longitude), 'time': str(time)})
        
        latLonData.append(latLonSubData)


    #HTML content for flight plan visualization
    mapHTML = """
    <!DOCTYPE html>
    <html>
       <head>
          <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
          <meta charset="utf-8">
          <title>Flight Plan</title>
          <style>
         html, body, #gMap {
         height: 100%;
         }
          </style>
       </head>
       <body>
          <div id="gMap"></div>
          <script>
         function setPlanToMap() {
         var gMap = new google.maps.Map(document.getElementById('gMap'), {
         zoom: 5,
         center: {lat: 38.04, lng: -99.17},
         mapTypeId: 'satellite'
                 });

                 var flightPlanPoints = """ + repr(latLonData) + """;
                 for(flight = 0; flight < flightPlanPoints.length; flight++) {
                     for(waypoint = 0; waypoint < flight.length; waypoint++) {
                     alert("d");
                     latitude = parseFloat(flightPlanPoints[flight][waypoint]['lat']);
                     longitude = parseFloat(flightPlanPoints[flight][waypoint]['lng']);
                     var marker = new google.maps.Marker({position:new google.maps.LatLng(latitude, longitude)});
                     marker.setMap(gMap);
                     label = flightPlanPoints[flight][waypoint]['time'].toString()
                     var infowindow = new google.maps.InfoWindow({
               content: label
             });
                      infowindow.open(gMap, marker);
                     }
        
         var flightRoute = new google.maps.Polyline({
         path: flightPlanPoints[flight],
         geodesic: true,
         strokeColor: "#"+((1<<24)*Math.random()|0).toString(16),
         strokeWeight: 3
         });
         flightRoute.setMap(gMap);
         }
         }
          </script>
          <script async defer
         src="https://maps.googleapis.com/maps/api/js?key=""" + GOOGLE_MAP_API_KEY + """&callback=setPlanToMap"></script>
       </body>
    </html>
        """

    #Write HTML to file, open using system command on browser
    f = open("map.html","w")
    f.write(mapHTML)
    f.close()
    
    if (find_executable("google-chrome") is not None) :
        os.system("google-chrome map.html")
    elif (find_executable("firefox") is not None) :
        os.system("firefox map.html")
