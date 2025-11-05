# ------------------------------------------------ #
# Fuel-Optimal Fleet Assignment (FOFA) Model - V2
# ------------------------------------------------ #
# DESCRIPTION:
# The following code implements the fuel-optimal fleet 
# assignment (FOFA) framework within the Brazilian domestic 
# air transportation system.
#
# VERSION:
# This code implements the framework on for the entirety
# of 2024 (i.e., aggregating all 366 days) so to enable 
# conducting a sensitivity analysis based on the 
# system-wide load factor. The framework can either be 
# implemented separately on the three largest domestic
# Brazilian airlines, or considering all eight domestic
# airlines on aggregate as a combined national airline.
# ------------------------------------------------ #

# ------------------------------------------------ #
# Step 0: User-editable Parameters
# ------------------------------------------------ #

# --> Data Directory 
directory = '/Users/andyeske/Desktop/Fall 2025/Optimization Methods/Project/Processed Datasets/'

# --> System-wide Load Factor Sensitivity Vector
# LF_vec = [0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9]
LF_vec = [0.83]

# --> Desired Airline
# Options: 
# (1): AD - Azul Linhas Aereas Brasileiras 
#      8 aircraft types, 30.2% market share by ASKs

# (2): G3 - GOL Linhas Aereas
#      3 aircraft types, 30.5% market share by ASKs

# (3): JJ - LATAM Brasil 
#      6 aircraft types, 38.7% market share by ASKs

# (4): Combined Brazilian National Airline (19 aircraft types)
#      19 aircraft types, including aicraft from:
#      -> 0S - Sideral Linhas Aereas
#      -> 2F - Azul Conecta
#      -> 2Z - Voepass
#      -> 7M - MAP Linhas Aereas
#      -> E4 - Abaete Aviacao

Option = 1

# ------------------------------------------------ #
# Step 1: Importing Packages
# ------------------------------------------------ #
import pandas as pd
import math
import os
import time
import numpy as np
import numba as nb
import random
from multiprocessing import Pool
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import gurobipy as gp
from numba import jit
from gurobipy import GRB

# ------------------------------------------------ #
# Step 1: Importing and Processing the Relevant Datasets
# ------------------------------------------------ #

# User-Option
if Option == 1: # Azul Linhas Aereas Brasileiras
    Airline = 4
elif Option == 2: # GOL Linhas Aereas
    Airline = 6
elif Option == 2: # LATAM Brasil
    Airline = 7
else: # Combined Brazilian National Airline
    Airline = 8
    
# Aircraft Types
aircraft = ['A20N','A21N','A319','A320','A321','A332','A339','AT45','AT75','AT76','B38M','B733','B735','B737','B738','B789','C208','E195','E295']

# Route Distance
file_path = open(os.path.join(directory + 'route_distances.csv'), 'r'); route_distance = pd.read_csv(file_path, header=None); 
route_distance = route_distance.to_numpy(); route_distance = np.round(route_distance[:,0])

# Flight Times on each Route
file_path = open(os.path.join(directory + 'route_flight_times.csv'), 'r'); flight_time = pd.read_csv(file_path, header=None); 
flight_time = flight_time.to_numpy(); flight_time = np.round(flight_time[:,0],3)

# Airport Codes
file_path = directory + 'Airport Codes.xlsx'; airport_codes = pd.read_excel(file_path);
runway_length = airport_codes['Primary Runway Length'].to_numpy(); runway_length = runway_length[0:169]

# Aircraft Operating Statistics
file_path = '/Users/andyeske/Desktop/Fall 2025/Optimization Methods/Project/Processed Datasets/Aircraft Statistics.xlsx'; aircraft_statistics = pd.read_excel(file_path,skiprows=1); 
aircraft_fuel = aircraft_statistics['Fuel Burn (L/km)'].to_numpy(); aircraft_fuel = aircraft_fuel[0:19]; aircraft_fuel = np.vstack(aircraft_fuel).astype(np.float64); aircraft_fuel = np.round(aircraft_fuel[:,0],3)
aircraft_seats = aircraft_statistics['Average Seats'].to_numpy(); aircraft_seats = aircraft_seats[0:19]; aircraft_seats = np.vstack(aircraft_seats).astype(np.int64); aircraft_seats = np.round(aircraft_seats[:,0],3)
aircraft_range = aircraft_statistics['Range'].to_numpy(); aircraft_range = aircraft_range[0:19]; aircraft_range = np.vstack(aircraft_range).astype(np.int64); aircraft_range = np.round(aircraft_range[:,0],3)
aircraft_stage_length = aircraft_statistics['Average Stage Length (km)'].to_numpy(); aircraft_stage_length = aircraft_stage_length[0:19]
aircraft_runways = aircraft_statistics['Take-off Length (m)'].to_numpy(); aircraft_runways = aircraft_runways[0:19]

# Airport Index Vector
file_path = open(os.path.join(directory + 'route_airport_index.csv'), 'r'); route_airports = pd.read_csv(file_path, header=None); 
route_airports = route_airports.to_numpy();
Airport_In = route_airports[:,0]-1
Airport_Out = route_airports[:,1]-1

if Airline > 7:
    # Departures on each Route
    file_path = open(os.path.join(directory + 'daily_route_departures.csv'), 'r'); total_departures = pd.read_csv(file_path, header=None); 
    total_departures = total_departures.to_numpy(); 
    total_departures = np.sum(total_departures,1); 
    total_departures = np.round(total_departures)
    l_routes = 1417
    
    # Passengers Transported on each Route
    file_path = open(os.path.join(directory + 'daily_route_passengers.csv'), 'r'); total_passengers = pd.read_csv(file_path, header=None); 
    total_passengers = total_passengers.to_numpy();
    total_passengers = np.sum(total_passengers,1)
    total_passengers = np.round(total_passengers)
    
    # Fuel Consumed on each Route
    file_path = open(os.path.join(directory + 'daily_fuel_consumed.csv'), 'r'); total_fuel = pd.read_csv(file_path, header=None); 
    total_fuel = total_fuel.to_numpy(); 
    total_fuel = np.sum(total_fuel,1); 
    total_fuel = np.round(total_fuel,3)
    
    # Aircraft Total Flying Time
    file_path = open(os.path.join(directory + 'daily_aircraft_flying_times.csv'), 'r'); total_aircraft_flying_times = pd.read_csv(file_path, header=None); 
    total_aircraft_flying_times = total_aircraft_flying_times.to_numpy(); 
    total_aircraft_flying_times = np.sum(total_aircraft_flying_times,1); 
    total_aircraft_flying_times = np.round(total_aircraft_flying_times)
    l_aircraft = 19
    
    # total RPKs and ASKs
    file_path = open(os.path.join(directory + 'Metrics.csv'), 'r'); metrics = pd.read_csv(file_path, header=None); 
    metrics = metrics.to_numpy();
    RPKs = np.sum(metrics[1,:])
    ASKs = np.sum(metrics[0,:])
    average_total_LF = np.round(RPKs/ASKs,3)
      
else:
    # Passengers Transported on each Route
    file_path = open(os.path.join(directory + 'airline_specific_avg_yearly_route_passengers.csv'), 'r'); total_passengers = pd.read_csv(file_path, header=None); 
    total_passengers = total_passengers.to_numpy(); 
    total_passengers = np.round(total_passengers[:,Airline]); 
    total_passengers = total_passengers.astype(np.int64)
    l_routes = 1417
    
    # Departures on each Route
    file_path = open(os.path.join(directory + 'airline_specific_avg_yearly_route_departures.csv'), 'r'); total_departures = pd.read_csv(file_path, header=None); 
    total_departures = total_departures.to_numpy(); total_departures = total_departures[:,Airline]
    total_departures = total_departures.astype(np.int64)
    total_departures = total_departures*(total_passengers > 0)
    
    # Fuel Consumed on each Route
    file_path = open(os.path.join(directory + 'airline_specific_avg_yearly_fuel_consumed.csv'), 'r'); total_fuel = pd.read_csv(file_path, header=None); 
    total_fuel = total_fuel.to_numpy(); total_fuel = total_fuel[:,Airline]
    total_fuel = total_fuel*(total_passengers > 0)
    
    # Aircraft Total Flying Time
    file_path = open(os.path.join(directory + 'airline_specific_avg_yearly_aircraft_flying_times.csv'), 'r'); total_aircraft_flying_times = pd.read_csv(file_path, header=None); 
    total_aircraft_flying_times = total_aircraft_flying_times.to_numpy(); total_aircraft_flying_times = total_aircraft_flying_times[:,Airline]
    l_aircraft = 19
    
    # total RPKs and ASKs
    file_path = open(os.path.join(directory + 'airline_specific_avg_yearly_metrics.csv'), 'r'); metrics = pd.read_csv(file_path, header=None); 
    metrics = metrics.to_numpy();
    RPKs = np.sum(metrics[1,Airline])
    ASKs = np.sum(metrics[0,Airline])
    average_total_LF = np.round(RPKs/ASKs,3)

# ------------------------------------------------ #
# Step 2: Building and Solving the Optimization Problem
# ------------------------------------------------ #

# Load Factor Sensitivity Analysis
reduction_vec = np.zeros([len(LF_vec),1])
LF_in = 0

# Starting the sensitivity iterations, proving multiple load factors
for target_LF in LF_vec:
    
    # Optimization Problem:
    m = gp.Model("Model") 
    
    # Number of flights on the O-D market operated by aircraft type a
    x = m.addMVar(shape=(l_routes,l_aircraft), lb=0.0, vtype=GRB.INTEGER, name="x") # Adding variables
        
    # Objective Function:
    # Minimize the fuel consumption
    m.setObjective(np.transpose(route_distance)@x@aircraft_fuel, GRB.MINIMIZE)
    
    # Constraints:
    # 1) At least the same number of passengers must be carried on each route
    constraint_1 = m.addConstr(x@aircraft_seats >= total_passengers)
    
    # 2) At least the same number of flights on each route must be preserved
    constraint_2 = m.addConstr(x.sum(1) >= total_departures)
    
    # 3) At least a minimum system-wide load factor must be attained
    new_ASKs = np.transpose(route_distance)@x@aircraft_seats
    constraint_3 = m.addConstr(RPKs <= target_LF*new_ASKs)
    
    # 4) The total flying time can't exceed the actual flying time
    constraint_4 = m.addConstr(np.transpose(flight_time)@x <= np.transpose(1.01*total_aircraft_flying_times))
    
    # 5) Aircraft must possess sufficient range to cover the route
    logic_distance = np.zeros([l_routes,l_aircraft])
    for a in range(0,l_aircraft):
        logic_distance[:,a] = route_distance > aircraft_range[a]
    non_permissible_aircraft = np.nonzero(logic_distance)
    in_itinerary = non_permissible_aircraft[0].astype(int)
    in_aircraft = non_permissible_aircraft[1].astype(int)
    if len(in_itinerary) > 0:
        constraint_5 = m.addConstr(x[in_itinerary,in_aircraft] == 0)
    
    # 6) The number of aircraft of type a into airport o must be equal to the 
    # number of aircraft of the same type a out of airport o 
    for node in range(0,169):
        o_in = np.argwhere(Airport_In == node).astype(int);
        d_in = np.argwhere(Airport_Out == node).astype(int);
        if len(o_in) > 0:
            m.addConstr(sum(x[o_in,:]) == sum(x[d_in,:]))
        
    # 7) An aircraft can only fly into an airport with a sufficiently long runway
    for node in range(0,169):
        runway = runway_length[node]
        o_in = np.argwhere(Airport_In == node).astype(int);
        d_in = np.argwhere(Airport_Out == node).astype(int);
        runway_in = np.nonzero(1*(aircraft_runways > runway))[0]
        l_runway_in = len(runway_in)
        for a_in in range(0,l_runway_in):
            m.addConstr((x[o_in,runway_in[a_in]]) == 0)
            m.addConstr((x[d_in,runway_in[a_in]]) == 0)
    
    # --> Modifying model parameters (adjust as needed)
    m.Params.Presolve = 0
    m.Params.MIPFocus= 1
    m.Params.Cuts= 1
    m.Params.Heuristics = 0.98
    m.Params.NoRelHeurTime = 1000
    m.Params.MIPGap= 0.005
    
    # Solving the optimization problem
    m.optimize() 
    x_solution = x.X
    
    # Calculating the fuel reduction for the given load factor
    original_fuel_consumption = np.sum(total_fuel)
    new_fuel_consumption = np.sum(x_solution@aircraft_fuel*route_distance)
    reduction_vec[LF_in] = 100*(new_fuel_consumption - original_fuel_consumption)/original_fuel_consumption
    
    # Updating the LF index
    LF_in = LF_in + 1
    