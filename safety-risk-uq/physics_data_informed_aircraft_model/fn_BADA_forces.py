# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:39:57 2021

@author: Abhinav
"""


import pickle
import numpy as np

def pyfn_BADA_drag_thrust(h, V_TAS, flight_phase, phi, AC_model, GL_Origin):   
    # Load BADA data for aircraft models
#    filename1 = 'data_dictBADA_1'
    filename2 = 'data_dictBADA_2'
#    with open(filename1, 'rb') as f:
#        dictBADA_1 = pickle.load(f)
    with open(filename2, 'rb') as f:
        dictBADA_2 = pickle.load(f)

    # load variables from dictionary    
    m = dictBADA_2[AC_model]['reference_mass']
    S = dictBADA_2[AC_model]['ref_wing_surf_area']
    V_stall_cruise = dictBADA_2[AC_model]['Vstall_CR']
    V_stall_approach = dictBADA_2[AC_model]['Vstall_AP']
    C_D0_TO = dictBADA_2[AC_model]['CD0_TO']
    C_D0_IC = dictBADA_2[AC_model]['CD0_IC']
    C_D0_CR = dictBADA_2[AC_model]['CD0_CR']
    C_D0_AP = dictBADA_2[AC_model]['CD0_AP']
    C_D0_LDG = dictBADA_2[AC_model]['CD0_LD']
    C_D2_TO = dictBADA_2[AC_model]['CD2_TO']
    C_D2_IC = dictBADA_2[AC_model]['CD2_IC']
    C_D2_CR = dictBADA_2[AC_model]['CD2_CR']
    C_D2_AP = dictBADA_2[AC_model]['CD2_AP']
    C_D2_LDG = dictBADA_2[AC_model]['CD2_LD']
    C_TC1 = dictBADA_2[AC_model]['Max_climb_thrust_coefficients_1']
    C_TC2 = dictBADA_2[AC_model]['Max_climb_thrust_coefficients_2']
    C_TC3 = dictBADA_2[AC_model]['Max_climb_thrust_coefficients_3']
    C_TC4 = dictBADA_2[AC_model]['Max_climb_thrust_coefficients_4']
    C_TC5 = dictBADA_2[AC_model]['Max_climb_thrust_coefficients_5']
    C_Tdes_high = dictBADA_2[AC_model]['Desc_high']
    C_Tdes_low = dictBADA_2[AC_model]['Desc_low']
    C_Tdes_app = dictBADA_2[AC_model]['Desc_app']
    C_Tdes_ld = dictBADA_2[AC_model]['Desc_ld']
    h_des = dictBADA_2[AC_model]['Desc_level']
    
    # common quantities
    # Inputs needed: h = ?, V_TAS = ?, phi = ?   
    g = 9.81 # Units: m/s^2
    R = 287.04 # Units: m^2/Ks^2
    
    T_0_ISA = 288.15 # Units: K
    DT_ISA = 0 # Units: K, assuming standard atmospheric conditions
    T_0 = T_0_ISA + DT_ISA
    
    h_trop = 11000 + 1000*DT_ISA/6.5
    T_trop = 216.65 # Units: K
    if h>=h_trop:
        T = T_trop
    else:
        T = T_0 - 6.5*h/1000
    
    rho_0_ISA = 1.225 # Units: kg/m^3
    rho_0 = rho_0_ISA*T_0_ISA/T_0
    if h<h_trop:
        rho = rho_0*np.power(T/T_0, 4.25864)
    else:
        rho_trop = rho_0*np.power(T_trop/T_0,4.25864)
        rho = rho_trop*np.exp(-(g/R/T_trop)*(h-h_trop))
       
    # Aerodynamic Drag
    V_min_cruise = 1.3*V_stall_cruise
    V_min_approach = 1.3*V_stall_approach
    
    C_L = 2*m*g/(rho*np.power(V_TAS,2)*S*np.cos(np.deg2rad(phi)));
    
    C_D_TO = C_D0_TO + C_D2_TO*np.power(C_L,2)
    C_D_IC = C_D0_IC + C_D2_IC*np.power(C_L,2)
    C_D_CR = C_D0_CR + C_D2_CR*np.power(C_L,2)
    C_D_AP = C_D0_AP + C_D2_AP*np.power(C_L,2) # h<8000ft, V_TAS<V_min_cruise+10kts
    C_D_LDG = C_D0_LDG + C_D2_LDG*np.power(C_L,2) # h<3000ft, V_TAS<V_min_approach+10kts
    
    if flight_phase==1:
        C_D = C_D_CR
        if h<400+GL_Origin:
            C_D = C_D_TO
        elif h<2000+GL_Origin:
            C_D = C_D_IC
    elif flight_phase==2:
        C_D = C_D_CR
    elif flight_phase==3:
        C_D = C_D_CR;
        if h<8000 and V_TAS<V_min_cruise+10:
            C_D = C_D_AP
        if h<3000 and V_TAS<V_min_approach+10:
            C_D = C_D_LDG

    DragForce = 0.5*C_D*rho*np.power(V_TAS,2)*S

    # Aerodynamic Thrust
    DT_ISA_eff = DT_ISA-C_TC4
    if C_TC5>0:
        DT_ISA_eff = max(0, DT_ISA_eff*C_TC5)
        DT_ISA_eff = min(DT_ISA_eff,0.4/C_TC5)

    T_max_climb_ISA = C_TC1*(1 - h/C_TC2 + C_TC3*h*h)
    T_max_climb = T_max_climb_ISA*(1 - C_TC5 - DT_ISA_eff)

    C_T_CR = 0.95
    T_cruise_max = C_T_CR*T_max_climb

    T_des_high = C_Tdes_high*T_max_climb
    T_des_low = C_Tdes_low*T_max_climb
    T_des_app = C_Tdes_app*T_max_climb
    T_des_ld = C_Tdes_ld*T_max_climb
    if h>h_des:
        T_des = T_des_high
    else:
        T_des = T_des_low

    if h<8e3 and V_TAS<V_min_cruise+10:
        T_des = T_des_app
    elif h<3e3 and V_TAS<V_min_approach+10:
        T_des = T_des_ld

    if flight_phase==1:
        ThrustForce = T_max_climb
    elif flight_phase==2:
        ThrustForce = T_cruise_max
    elif flight_phase==3:
        ThrustForce = T_des
    
    return DragForce, ThrustForce;