'''
          GENARALIZED NATIONAL AIRSPACE TRAJECTORY-PREDICTION SYSTEM (GNATS)
          Copyright 2018 by Optimal Synthesis Inc. All rights reserved
          
Author: Parikshit Dutta
Date: 2018-04-02

Update: 2019.06.18
'''


import math
import numpy as np
import os

try:
    import xml.etree.ElementTree as ET
except ImportError:
    print('Please install XML element tree')

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Please install matplotlib')   

import matplotlib 
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

def is_float(val):
    try:
        np.float(val)
        return True;
    except:
        return False;
    
def linear_interpolation(xlow,xhigh,flow,fhigh,xeval):
    
    feval = flow + (xeval-xlow)/(xhigh-xlow)*(fhigh-flow)
    return feval;
    

'''This class read the output files at the given file path and plots histograms. 
    '''
class PostProcessor:


    def __init__(self,file_path = '../GNATS_Standalone', \
                 ac_name = 'ULI13-2553603', \
                 file_type = 'csv', plot_failure = False, \
                 failure_file = "conflict_manager.txt", \
                 filenames_file = "file_list.txt"):

        '''MODIFY THIS TO THE LOCATION WHERE MC FILES ARE SITTING'''
        self.path_to_csv = file_path;
        if not os.path.exists(self.path_to_csv):
            print(self.path_to_csv,' no such path exists.')
            if not self.searchAndChangePath():
                print('No GNATS HOME folder found. Exiting.')
                quit();
        
        '''THIS IS THE FILE TYPE IN WHICH THE OUTPUTS ARE SAVED'''
        self.file_type = file_type;
        
        if not plot_failure:
            '''THIS VARIABLE IS USED FOR PLOTTING LAT LON ALT HISTOGRAMS'''
            self.LatLonAltSamp = []
        
            self.get_all_files();
            if not (type(ac_name) is list):
                '''THIS IS THE AC WHOSE LAT LON WERE MANIPULATED'''
                self.ac_mod = ac_name;
                self.createCSVSamples();            
            elif len(ac_name) == 1:
                '''THIS IS THE AC WHOSE LAT LON WERE MANIPULATED'''
                self.ac_mod = ac_name[0];
                self.createCSVSamples();
            else:
                self.readCSVSamples()
        else:
            self.failure_file_path = os.path.join(file_path,failure_file); 
            self.filenames_file = os.path.join(file_path,filenames_file)
    
    '''Change Path if wrong path given for GNATS HOME'''
    def searchAndChangePath(self):
        parentidx = self.path_to_csv.rfind('/')
        parent = self.path_to_csv[:parentidx]
        parentfolders = os.listdir(parent);
        
        foldname = ''
        for fname in parentfolders:
            if 'GNATS_Standalone' in fname:
                foldname = fname
        
        if foldname == '':
            return False;
        else:
            self.path_to_csv = parent +'/'+ foldname;
            print('Changing directory to',self.path_to_csv);
            return True;
            
                
                 
    
    '''Gets the list of all files with the given extension.
    Although only csv has been implemented. Can be extended to xml too.  
    '''    
    def get_all_files(self):
        
        self.filepath = [];
        for fl in os.listdir(self.path_to_csv):
            if fl.endswith("."+self.file_type):                
                self.filepath.append(self.path_to_csv+'/'+ fl)
        if self.filepath == []:
            print('No such files found')
            quit();
    
    
    '''Creates a database of Samples by reading the CSV files from the 
    output folder'''
    def createCSVSamples(self):
        
        self.all_sample_dict = {}
        cnt = 0
        for fp in self.filepath:
            print('Reading ',cnt+1,' file: ', fp)
            cnt = cnt+1
            fid = open(fp,'r');
            lines = fid.readlines()
            fid.close()
            sim_start_time = -100;
            self.ac_info_map = {}
            self.ac_traj_map = {}
    
            currac = ''
            prevac = ''
            tpoints = [];
            for k in range(len(lines)):
                lines[k] = lines[k].split(',')
        
                if len(lines[k]) == 1:
                    if is_float(lines[k][0]) and sim_start_time <0:
                        sim_start_time = np.float(lines[k][0])
                    else:
                        continue;
        
                if '*' in lines[k] or len(lines[k]) == 0 or lines[k] == '\n':
                    continue;
                
                if 'AC' in lines[k]:            
                    currac = lines[k][2];
                    actype = lines[k][3];
                    origin = lines[k][4];
                    dest = lines[k][5];
                    ac_sim_start = np.float(lines[k][6])
                    sim_int_gnd = np.float(lines[k][7])
                    sim_int_air = np.float(lines[k][8])
                    cruise_alt_ft = np.float(lines[k][9])
                    cruise_tas_kts = np.float(lines[k][10])
                    orig_elev_ft = np.float(lines[k][11])
                    dest_elev_ft = np.float(lines[k][12])
                    num_traj_rec = np.int(lines[k][13])
                    
                    self.ac_info_map[currac] = [actype,origin,dest,ac_sim_start,sim_int_gnd,
                                           sim_int_air,cruise_alt_ft, cruise_tas_kts,orig_elev_ft,
                                           dest_elev_ft,num_traj_rec];
                    if tpoints != []: 
                        self.ac_traj_map[prevac] = tpoints;
                    tpoints = [];
                elif currac != '':        

                    timestamp = np.float(lines[k][0])            
                    lat_deg = np.float(lines[k][1])
                    lon_deg = np.float(lines[k][2])
                    alt_ft = np.float(lines[k][3])
                    rocd_fps = np.float(lines[k][4])
                    tas_kts = np.float(lines[k][5])
                    tas_kts_gnd = np.float(lines[k][6]) 
                    heading_rad = np.float(lines[k][7])
                    fpa_rad = np.float(lines[k][8])
                    #sector_idx = np.int(lines[k][9])
                    #sector_names = lines[k][10]
                    flight_phase = lines[k][9]            
                    traj_prop = [timestamp,
                                 lat_deg,
                                 lon_deg,
                                 alt_ft,
                                 rocd_fps,
                                 tas_kts,
                                 tas_kts_gnd,
                                 heading_rad * 180 / math.pi,
                                 fpa_rad * 180 / math.pi,
                                 #sector_idx,
                                 #sector_names,
                                 flight_phase];
            
                    tpoints.append(traj_prop)
                else:
                    pass                      
        
                prevac = currac;   
    
            if tpoints != []:
                self.ac_traj_map[prevac] = tpoints;
    
            value = self.ac_traj_map[self.ac_mod]
    
            lat_lon_alt_arr = []
            for s in value:
                lat_lon_alt_arr.append([s[0],s[1],s[2],s[3]])
    
            self.LatLonAltSamp.append(lat_lon_alt_arr);
            
            self.all_sample_dict[fp] = [self.ac_traj_map,self.ac_info_map]
            
    '''Creates a database of Samples by reading the XML files from the output folder''' 
    def createXMLSamples(self):

        root_array = []
        for fl in self.filepath:
            tree = ET.parse(fl)
            root_array.append(tree.getroot())
            self.ac_info_map = {}
            self.ac_traj_map = {}
        
        self.LatLonAltSamp = []
        for root in root_array:
            Trajs = [];
            for s in root:
                Traj = [];
                acid =  s.get('callsign')        
                if acid != self.ac_mod:
                    continue;
                #         print acid
                for p in s:
                    Trajpt = []
                    for r in p:                
                        try: 
                            Trajpt.append(np.double(r.text));
                        except:                
                            pass
                    Traj.append(Trajpt)
                Trajs.append(Traj)
            self.LatLonAltSamp.append(Trajs)
            
    '''Plotting routine. Plots the trajectory and histograms for times given in times_to_plot
    variable. By default it plots histograms at beginning of the simulation. Midpoint of
    the simulation and at the end of the simulation.'''
            
    def plotRoutine(self,ac_list = [],times_to_plot = []):
        
        if ac_list != []:
            self.plotMultipleTrajs(ac_list)
            return 0;
        
        idx = []
        if times_to_plot != []:
            if len(times_to_plot) == 3:
                
                for t in times_to_plot:                                        
                    for k in len(self.ac_traj_map[self.ac_mod]):
                        if self.ac_traj_map[self.ac_mod][k][0] == t:
                            idx.append( k);

        
        Ini_lat_lon_alt = []
        Mid_lat_lon_alt = []
        End_lat_lon_alt = []
#         Fig0 = plt.figure()
#         ax = Fig0.add_subplot(111, projection='3d')   
        for lat_lon_alt_vec in self.LatLonAltSamp:
            lat_lon_alt_vec = np.array(lat_lon_alt_vec, dtype = np.double)
            vec_len = len(lat_lon_alt_vec);
            if idx == []:
                idx = [0,vec_len/2,vec_len-1]
            times = [lat_lon_alt_vec[0,0]/3600.0,lat_lon_alt_vec[int(vec_len/2),0]/3600.0,lat_lon_alt_vec[-1,0]/3600.0]
                
            Ini_lat_lon_alt.append([lat_lon_alt_vec[0,1],lat_lon_alt_vec[0,2],lat_lon_alt_vec[0,3]])
            Mid_lat_lon_alt.append([lat_lon_alt_vec[int(vec_len/2),1],lat_lon_alt_vec[int(vec_len/2),2],lat_lon_alt_vec[int(vec_len/2),3]])
            End_lat_lon_alt.append([lat_lon_alt_vec[-1,1],lat_lon_alt_vec[-1,2],lat_lon_alt_vec[-1,3]])
#             ax.plot(lat_lon_alt_vec[:,2],lat_lon_alt_vec[:,1],zs = lat_lon_alt_vec[:,3])
#             ax.set_xlabel('Longitude (deg)')
#             ax.set_ylabel('Latitude (deg)')
#             ax.set_zlabel('Altitude (ft)')
#             ax.set_title('Plotting for callsign '+ self.ac_mod,loc = 'left');
        
                
        Ini_lat_lon_alt = np.array(Ini_lat_lon_alt,dtype = np.float);
        Mid_lat_lon_alt = np.array(Mid_lat_lon_alt,dtype = np.float);
        End_lat_lon_alt = np.array(End_lat_lon_alt,dtype = np.float);
        
        
        Fig1,axarr = plt.subplots(nrows=3,ncols=3)
        plt.suptitle('Plotting for callsign '+ self.ac_mod)
        
        _,histvals,_ = axarr[0,0].hist(Ini_lat_lon_alt[:,0],30,normed=False)
        axarr[0,0].set_title('time = '+ str(times[0])+' hr' )
        axarr[0,0].set_xlabel('Lat (deg)')
        axarr[0,0].set_xticks([histvals[0],histvals[-1]]);
        
        _,histvals,_ =axarr[0,1].hist(Mid_lat_lon_alt[:,0],30,normed=False)
        axarr[0,1].set_title('time = '+ str(times[1])+' hr')
        axarr[0,1].set_xlabel('Lat (deg)')
        axarr[0,1].set_xticks([histvals[0],histvals[-1]]);
        
        _,histvals,_ =axarr[0,2].hist(End_lat_lon_alt[:,0],30,normed=False)
        axarr[0,2].set_title('time = '+ str(times[2])+' hr')
        axarr[0,2].set_xlabel('Lat (deg)')
        axarr[0,2].set_xticks([histvals[0],histvals[-1]]);

        _,histvals,_ =axarr[1,0].hist(Ini_lat_lon_alt[:,1],30,normed=False)
        axarr[1,0].set_xlabel('Lon (deg)')
        axarr[1,0].set_xticks([histvals[0],histvals[-1]]);
                               
        _,histvals,_ =axarr[1,1].hist(Mid_lat_lon_alt[:,1],30,normed=False)
        axarr[1,1].set_xlabel('Lon (deg)')
        axarr[1,1].set_xticks([histvals[0],histvals[-1]]);        
        
        _,histvals,_ =axarr[1,2].hist(End_lat_lon_alt[:,1],30,normed=False)
        axarr[1,2].set_xlabel('Lon (deg)')
        axarr[1,2].set_xticks([histvals[0],histvals[-1]]);

        _,histvals,_ =axarr[2,0].hist(Ini_lat_lon_alt[:,2],30,normed=False)
        axarr[2,0].set_xlabel('Alt (ft)')
        axarr[2,0].set_xticks([histvals[0],histvals[-1]]);
                               
        _,histvals,_ =axarr[2,1].hist(Mid_lat_lon_alt[:,2],30,normed=False)
        axarr[2,1].set_xlabel('Alt (ft)')
        axarr[2,1].set_xticks([histvals[0],histvals[-1]]);        
        
        _,histvals,_ =axarr[2,2].hist(End_lat_lon_alt[:,2],30,normed=False)
        axarr[2,2].set_xlabel('Alt (ft)')
        axarr[2,2].set_xticks([histvals[0],histvals[-1]]);   
        Fig1.tight_layout()   
        Fig1.subplots_adjust(top=0.9)
        
        Fig2,axarr = plt.subplots(nrows=1,ncols=3)
        plt.suptitle('Plotting for callsign '+ self.ac_mod)
        
        _,xhistvals,yhistvals,_ = axarr[0].hist2d(Ini_lat_lon_alt[:,0],Ini_lat_lon_alt[:,1],30,normed=False)
        axarr[0].set_title('time = '+ str(times[0])+' hr' )
        axarr[0].set_xlabel('Lat (deg)')
        axarr[0].set_ylabel('Lon (deg)')
        axarr[0].set_xticks([xhistvals[0],xhistvals[-1]]);
        
        
        _,xhistvals,yhistvals,_ = axarr[1].hist2d(Mid_lat_lon_alt[:,0],Mid_lat_lon_alt[:,1],30,normed=False)
        axarr[1].set_title('time = '+ str(times[1])+' hr')
        axarr[1].set_xlabel('Lat (deg)')
        axarr[1].set_ylabel('Lon (deg)')
        axarr[1].set_xticks([xhistvals[0],xhistvals[-1]]);
        
         
        _,xhistvals,yhistvals,_ = axarr[2].hist2d(End_lat_lon_alt[:,0],End_lat_lon_alt[:,1],30,normed=False)
        axarr[2].set_title('time = '+ str(times[2])+' hr')
        axarr[2].set_xlabel('Lat (deg)')
        axarr[2].set_ylabel('Lon (deg)')
        axarr[2].set_xticks([xhistvals[0],xhistvals[-1]]);
                
        plt.show()
    
    
    
    
    
    def readCSVSamples(self):
        
        self.LatLonAltSamp = []
        self.all_sample_dict = {}
        cnt = 0;
        for fp in self.filepath:            
            print('Reading ',cnt+1,' file: ', fp)
            cnt = cnt+1            
            fid = open(fp,'r');
            lines = fid.readlines()
            fid.close()
            sim_start_time = -100;
            self.ac_info_map = {}
            self.ac_traj_map = {}
    
            currac = ''
            prevac = ''
            tpoints = [];
            for k in range(len(lines)):
                lines[k] = lines[k].split(',')
        
                if len(lines[k]) == 1:
                    if is_float(lines[k][0]) and sim_start_time <0:
                        sim_start_time = np.float(lines[k][0])
                    else:
                        continue;
        
                if '*' in lines[k] or len(lines[k]) == 0 or lines[k] == '\n':
                    continue;
        
                
                if 'AC' in lines[k]:            
                    currac = lines[k][2];
                    actype = lines[k][3];
                    origin = lines[k][4];
                    dest = lines[k][5];
                    ac_sim_start = np.float(lines[k][6])
                    sim_int_gnd = np.float(lines[k][7])
                    sim_int_air = np.float(lines[k][8])
                    cruise_alt_ft = np.float(lines[k][9])
                    cruise_tas_kts = np.float(lines[k][10])
                    orig_elev_ft = np.float(lines[k][11])
                    dest_elev_ft = np.float(lines[k][12])
                    num_traj_rec = np.int(lines[k][13])
                    
                    self.ac_info_map[currac] = [actype,origin,dest,ac_sim_start,sim_int_gnd,
                                           sim_int_air,cruise_alt_ft, cruise_tas_kts,orig_elev_ft,
                                           dest_elev_ft,num_traj_rec];
                    if tpoints != []: 
                        self.ac_traj_map[prevac] = tpoints;
                    tpoints = [];
                elif currac != '':        

                    timestamp = np.float(lines[k][0])            
                    lat_deg = np.float(lines[k][1])
                    lon_deg = np.float(lines[k][2])
                    alt_ft = np.float(lines[k][3])
                    rocd_fps = np.float(lines[k][4])
                    tas_kts = np.float(lines[k][5])
                    tas_kts_gnd = np.float(lines[k][6]) 
                    heading_rad = np.float(lines[k][7])
                    fpa_rad = np.float(lines[k][8])
                    #sector_idx = np.int(lines[k][9])
                    #sector_names = lines[k][10]
                    flight_phase = lines[k][9]            
                    traj_prop = [timestamp,
                                 lat_deg,
                                 lon_deg,
                                 alt_ft,
                                 rocd_fps,
                                 tas_kts,
                                 tas_kts_gnd,
                                 heading_rad * 180 / math.pi,
                                 fpa_rad * 180 / math.pi,
                                 #sector_idx,
                                 #sector_names,
                                 flight_phase];
            
                    tpoints.append(traj_prop)
                else:
                    pass                      
        
                prevac = currac;   
            if tpoints != []:
                self.ac_traj_map[currac] = tpoints;
            
            
            self.all_sample_dict[fp] = [self.ac_traj_map,self.ac_info_map];
    
        return self.ac_traj_map,self.ac_info_map;
    
    
    def plotMultipleTrajs(self,ac_list):
        
        for ac in ac_list:
            self.ac_mod = ac;
            self.LatLonAltSamp = []
            for fp in self.filepath:
                [ac_traj_map,ac_info_map] = self.all_sample_dict[fp]
                value = ac_traj_map[self.ac_mod];        
    
                lat_lon_alt_arr = []
                for s in value:
                    lat_lon_alt_arr.append([s[0],s[1],s[2],s[3]])
    
                self.LatLonAltSamp.append(lat_lon_alt_arr);
            Ini_lat_lon_alt = []
            Mid_lat_lon_alt = []
            End_lat_lon_alt = []
   
            for lat_lon_alt_vec in self.LatLonAltSamp:
                lat_lon_alt_vec = np.array(lat_lon_alt_vec, dtype = np.double)
                vec_len = len(lat_lon_alt_vec);
                
                times = [lat_lon_alt_vec[0,0]/3600.0,lat_lon_alt_vec[int(vec_len/2),0]/3600.0,lat_lon_alt_vec[-1,0]/3600.0]
                
                Ini_lat_lon_alt.append([lat_lon_alt_vec[0,1],lat_lon_alt_vec[0,2],lat_lon_alt_vec[0,3]])
                Mid_lat_lon_alt.append([lat_lon_alt_vec[int(vec_len/2),1],lat_lon_alt_vec[int(vec_len/2),2],lat_lon_alt_vec[int(vec_len/2),3]])
                End_lat_lon_alt.append([lat_lon_alt_vec[-1,1],lat_lon_alt_vec[-1,2],lat_lon_alt_vec[-1,3]])

        
            Ini_lat_lon_alt = np.array(Ini_lat_lon_alt,dtype = np.float);
            Mid_lat_lon_alt = np.array(Mid_lat_lon_alt,dtype = np.float);
            End_lat_lon_alt = np.array(End_lat_lon_alt,dtype = np.float);

        
            Fig1,axarr = plt.subplots(nrows=3,ncols=3)
            plt.suptitle('Plotting for callsign '+ self.ac_mod)
        
            _,histvals,_ = axarr[0,0].hist(Ini_lat_lon_alt[:,0],30,normed=False)
#             axarr[0,0].set_title('time = '+ str(times[0])+' hr' )
            axarr[0,0].set_title('time = {:.2f} hr'.format(times[0]) )
            axarr[0,0].set_xlabel('Lat (deg)')
            axarr[0,0].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[0,0].set_xticks([histvals[0],histvals[-1]]);
        
            _,histvals,_ =axarr[0,1].hist(Mid_lat_lon_alt[:,0],30,normed=False)
            axarr[0,1].set_title('time = {:.2f} hr'.format(times[1]))
            axarr[0,1].set_xlabel('Lat (deg)')
            axarr[0,1].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[0,1].set_xticks([histvals[0],histvals[-1]]);
        
            _,histvals,_ =axarr[0,2].hist(End_lat_lon_alt[:,0],30,normed=False)
            axarr[0,2].set_title('time = {:.2f} hr'.format(times[2]))
            axarr[0,2].set_xlabel('Lat (deg)')
            axarr[0,2].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[0,2].set_xticks([histvals[0],histvals[-1]]);

            _,histvals,_ =axarr[1,0].hist(Ini_lat_lon_alt[:,1],30,normed=False)
            axarr[1,0].set_xlabel('Lon (deg)')
            axarr[1,0].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[1,0].set_xticks([histvals[0],histvals[-1]]);
                               
            _,histvals,_ =axarr[1,1].hist(Mid_lat_lon_alt[:,1],30,normed=False)
            axarr[1,1].set_xlabel('Lon (deg)')
            axarr[1,1].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[1,1].set_xticks([histvals[0],histvals[-1]]);        
        
            _,histvals,_ =axarr[1,2].hist(End_lat_lon_alt[:,1],30,normed=False)
            axarr[1,2].set_xlabel('Lon (deg)')
            axarr[1,2].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[1,2].set_xticks([histvals[0],histvals[-1]]);

            _,histvals,_ =axarr[2,0].hist(Ini_lat_lon_alt[:,2],30,normed=False)
            axarr[2,0].set_xlabel('Alt (ft)')
            axarr[2,0].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[2,0].set_xticks([histvals[0],histvals[-1]]);
                               
            _,histvals,_ =axarr[2,1].hist(Mid_lat_lon_alt[:,2],30,normed=False)
            axarr[2,1].set_xlabel('Alt (ft)')
            axarr[2,1].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[2,1].set_xticks([histvals[0],histvals[-1]]);        
        
            _,histvals,_ =axarr[2,2].hist(End_lat_lon_alt[:,2],30,normed=False)
            axarr[2,2].set_xlabel('Alt (ft)')
            axarr[2,2].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[2,2].set_xticks([histvals[0],histvals[-1]]);   
            Fig1.tight_layout()   
            Fig1.subplots_adjust(top=0.9)
        
            Fig2,axarr = plt.subplots(nrows=1,ncols=3)
            plt.suptitle('Plotting for callsign '+ self.ac_mod)
        
            _,xhistvals,yhistvals,_ = axarr[0].hist2d(Ini_lat_lon_alt[:,0],Ini_lat_lon_alt[:,1],30,normed=False)
            axarr[0].set_title('time = {:.2f} hr'.format(times[0]) )
            axarr[0].set_xlabel('Lat (deg)')
            axarr[0].set_ylabel('Lon (deg)')
            axarr[0].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[0].set_xticks([xhistvals[0],xhistvals[-1]]);
        
        
            _,xhistvals,yhistvals,_ = axarr[1].hist2d(Mid_lat_lon_alt[:,0],Mid_lat_lon_alt[:,1],30,normed=False)
            axarr[1].set_title('time = {:.2f} hr'.format(times[1]) )
            axarr[1].set_xlabel('Lat (deg)')
            axarr[1].set_ylabel('Lon (deg)')
            axarr[1].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[1].set_xticks([xhistvals[0],xhistvals[-1]]);
        
         
            _,xhistvals,yhistvals,_ = axarr[2].hist2d(End_lat_lon_alt[:,0],End_lat_lon_alt[:,1],30,normed=False)
            axarr[2].set_title('time = {:.2f} hr'.format(times[2]) )
            axarr[2].set_xlabel('Lat (deg)')
            axarr[2].set_ylabel('Lon (deg)')
            axarr[2].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            axarr[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[2].set_xticks([xhistvals[0],xhistvals[-1]]);
    
            
            Fig2.tight_layout()   
            Fig2.subplots_adjust(top=0.9)
                
            plt.show()
            
        
        
        
        
    
    '''timestamp,latitude_deg,longitude_deg,altitude_ft,rocd_fps,
    tas_knots,heading_deg,fpa_deg,flight_mode'''
    def plotVariable(self,varname = 'altitude'):

        var = [];

        fig1,axs = plt.subplots(nrows = 1, ncols = 3);
        plt.suptitle(self.ac_mod)
        for fp in self.filepath:
            [ac_traj_map,ac_info_map] = self.all_sample_dict[fp]
            value = ac_traj_map[self.ac_mod];
            idxf = len(value)
            if varname == 'initial_altitude':
                idx = 3;
            elif varname == 'departure_delay':
                idx = 0;
            elif varname == 'initial_latitude':
                idx = 1;
            
            var.append( [value[0][idx],value[int(idxf/2)][idx],value[-1][idx]] )
            
            tm= [];lat = []; lon = []; alt = [];
            for val in value:
                tm.append(val[0])
                lat.append([val[1]])
                lon.append(val[2])
                alt.append(val[3])
            
            axs[0].plot(tm,lat);axs[0].set_xlabel('time (s)');axs[0].set_ylabel('lat (deg)')
            
            axs[1].plot(tm,lon);axs[1].set_xlabel('time (s)');axs[1].set_ylabel('lon (deg)')
            
            axs[2].plot(tm,alt);axs[2].set_xlabel('time (s)');axs[2].set_ylabel('alt (ft)')
            
            
        
        
        
        var = np.array(var,dtype = np.float)
        fig2,axs = plt.subplots(nrows=2,ncols=1)
        plt.suptitle(self.ac_mod)
        axs[0].hist((var[:,0])/3600.0,bins = 30,normed=False,label = 'departure time')
        axs[0].set_xlabel('departure time (hr)')
        axs[1].hist((var[:,2])/3600.0,bins = 30,normed=False,label = 'arrival time')
        axs[1].set_xlabel('arrival time (hr)')
        fig2.tight_layout()      
        fig2.subplots_adjust(top=0.9)
        plt.show();
    
    '''timestamp,latitude_deg,longitude_deg,altitude_ft,rocd_fps,
    tas_knots,heading_deg,fpa_deg,flight_mode'''
    def plot_var_at_time(self, t_to_plt,inp_array,ac,varname = 'latitude'):
        
        Var = []
        for fp in self.filepath:
            [ac_traj_map,ac_info_map] = self.all_sample_dict[fp]
            ac_states = ac_traj_map[ac];
            
            idxf = len(ac_states)
            if varname == 'latitude':
                idx = 1;
            elif varname == 'longitude':
                idx = 2;
            elif varname == 'altitude':
                idx = 3;
            elif varname == 'course':
                idx = 6;
            elif varname == 'fpa':
                idx = 7;
            else:
                idx = -1;
            
            
            if idx <0:
                return 0;
            T = []
            tidxb = -1; tidxa=-1; 
            for k in range(idxf-1):
                if ac_states[k][0] == t_to_plt:
                    tidxb = k; 
                    break;
                elif ac_states[k][0] < t_to_plt and ac_states[k+1][0] > t_to_plt:
                    tidxb = k;
                    tidxa = k+1;
                    break;
            
            if tidxb < 0 and tidxa < 0:
                return 0;
            elif  tidxb > 0 and tidxa < 0:                
                var_val = ac_states[tidxb][idx]
            else :
                var_val = linear_interpolation(ac_states[tidxb][0], \
                                               ac_states[tidxa][0], \
                                               ac_states[tidxb][1], \
                                               ac_states[tidxa][1], \
                                               t_to_plt);
            
            Var.append(var_val)
        
        fig = plt.figure()
        fig.suptitle(varname)
        plt.plot(inp_array,Var)
        plt.show();
        
        
    def plotFailureHistograms(self,process_filename = False,nominal_failure = 7,start_idx = 0):
        
        
        fid = open(self.failure_file_path,'r')
        Failure_cnt = fid.readlines()
        fid.close()
    
        for i in range(len(Failure_cnt)):
            Failure_cnt[i] = np.float(Failure_cnt[i])
            
        Failure_cnt = Failure_cnt[start_idx:]
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

        
        matplotlib.rc('font', **font)
        
        Fig1 = plt.figure()
        ax = Fig1.gca();
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        n,_,_ = plt.hist(Failure_cnt,bins = 5,normed=True)
        plt.xlabel('Number of conflicts',fontsize = 20,fontstyle = 'normal',fontweight = 'bold'); 
        plt.ylabel('Normed Frequency',fontsize = 20,fontstyle = 'normal',fontweight = 'bold')
        plt.axvline(nominal_failure,c='r');
        avgval = ( min(n)+max(n) )/ 2;
        plt.annotate('Nominal case',xy=(nominal_failure+0.1, avgval), rotation = 90,fontsize = 15)
        
        
        
        
        if not process_filename:
            plt.show()
            return ;
        
        fid = open(self.filenames_file,'r')
        Filenames = fid.readlines()
        fid.close()
        
        
        
        
        delayval = []
        for k in range(len(Filenames)):
            Filenames[k] = Filenames[k].split(" ")
            fname = Filenames[k][-1]
            delay = ((fname.split('/'))[-1].split('_') )[2]
            delayval.append(np.float(delay));
            
        
        
        F = [Failure_cnt,delayval]
        F = np.transpose( np.array(F, dtype = np.double) )
        

        Fig2 = plt.figure()
        plt.plot(F[:,1],F[:,0],'xr')
        plt.ylabel('Number of conflicts'); plt.xlabel('Departure delay(s)')
        
        plt.show()
        
    
    def plotSingleAircraftTrajectory(self):
        
        fig,axs = plt.subplots(nrows = 2, ncols = 2)
        axs = axs.ravel()
        
        for lat_lon_alt_vec in self.LatLonAltSamp:
            lat_lon_alt_vec = np.array(lat_lon_alt_vec, dtype = np.double)
            
            axs[0].plot(lat_lon_alt_vec[:,0],lat_lon_alt_vec[:,1]);
            axs[0].set_xlabel('time (s)'); axs[0].set_ylabel('latitude (deg)')
            
            axs[1].plot(lat_lon_alt_vec[:,0],lat_lon_alt_vec[:,2]);
            axs[1].set_xlabel('time (s)'); axs[1].set_ylabel('longitude (deg)')
            
            
            axs[2].plot(lat_lon_alt_vec[:,0],lat_lon_alt_vec[:,3]);
            axs[2].set_xlabel('time (s)'); axs[2].set_ylabel('altitude (ft)')
            
            
        plt.show()
    
        
        
    