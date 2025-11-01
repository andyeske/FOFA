# ------------------------------------------------ #
# Fuel-Optimal Aircraft Allocation (FOAA) - Daily Basis (DB)
# ------------------------------------------------ #
# DESCRIPTION - FOAA: 
# The following code implements the fuel-optimal aircraft 
# allocation (FOAA) framework within the Brazilian domestic 
# air transportation system.
#
# VERSION - DAILY BASIS:
# This code implements the framework on a daily-basis 
# (i.e., the optimization is solved sequentially on each
# day separately), so to enable conducting a sensitivity 
# analysis based on route passenger variability. Here, 
# the load factors at each day are set to what was seen
# in practice in 2024.
# ------------------------------------------------ #

# ------------------------------------------------ #
# Step 0: User-editable Parameters
# ------------------------------------------------ #

# --> Data Directory 
directory = '/Users/andyeske/Desktop/Fall 2025/Optimization Methods/Project/Processed Datasets/'

# --> Days Sensitivity Vector
# Here, a 0: January 1, 2024, while a 99: April 9, 2024
days_vec = [0,1,2,3,4,5]

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

# Passengers Transported on each Route
file_path = open(os.path.join(directory + 'daily_route_passengers.csv'), 'r'); daily_passengers = pd.read_csv(file_path, header=None); 
daily_passengers = daily_passengers.to_numpy();
yearly_passengers = np.sum(daily_passengers,1)
l_routes = 1417

# Flight Times on each Route
file_path = open(os.path.join(directory + 'route_flight_times.csv'), 'r'); flight_time = pd.read_csv(file_path, header=None); 
flight_time = flight_time.to_numpy(); flight_time = np.round(flight_time[:,0],3)

# Fuel Consumed on each Route
file_path = open(os.path.join(directory + 'daily_fuel_consumed.csv'), 'r'); daily_fuel = pd.read_csv(file_path, header=None); 
daily_fuel = daily_fuel.to_numpy(); daily_fuel = np.round(daily_fuel,3); 
yearly_fuel = np.sum(daily_fuel,1); yearly_fuel = np.round(yearly_fuel,3)

# Route Distance
file_path = open(os.path.join(directory + 'route_distances.csv'), 'r'); route_distance = pd.read_csv(file_path, header=None); 
route_distance = route_distance.to_numpy(); route_distance = np.round(route_distance[:,0]/1000,3)

# Airport Codes
file_path = directory + 'Airport Codes.xlsx'; airport_codes = pd.read_excel(file_path);
runway_length = airport_codes['Primary Runway Length'].to_numpy(); runway_length = runway_length[0:169]

# Aircraft Total Flying Time
file_path = open(os.path.join(directory + 'daily_aircraft_flying_times.csv'), 'r'); daily_aircraft_flying_times = pd.read_csv(file_path, header=None); 
daily_aircraft_flying_times = daily_aircraft_flying_times.to_numpy(); daily_aircraft_flying_times = np.round(daily_aircraft_flying_times,3)
yearly_aircraft_flying_times = np.sum(daily_aircraft_flying_times,1); yearly_aircraft_flying_times = np.round(yearly_aircraft_flying_times,3)

# Aircraft Types
aircraft = ['A20N','A21N','A319','A320','A321','A332','A339','AT45','AT75','AT76','B38M','B733','B735','B737','B738','B789','C208','E195','E295']
l_aircraft = 19

# Aircraft Operating Statistics
file_path = '/Users/andyeske/Desktop/Fall 2025/Optimization Methods/Project/Processed Datasets/Aircraft Statistics.xlsx'; aircraft_statistics = pd.read_excel(file_path,skiprows=1); 
aircraft_fuel = aircraft_statistics['Fuel Burn (L/km)'].to_numpy(); aircraft_fuel = aircraft_fuel[0:19]; aircraft_fuel = np.vstack(aircraft_fuel).astype(np.float64); aircraft_fuel = np.round(aircraft_fuel[:,0],3)
aircraft_seats = aircraft_statistics['Average Seats'].to_numpy(); aircraft_seats = aircraft_seats[0:19]; aircraft_seats = np.vstack(aircraft_seats).astype(np.float64); aircraft_seats = np.round(aircraft_seats[:,0],3)
aircraft_range = aircraft_statistics['Range'].to_numpy(); aircraft_range = aircraft_range[0:19]; aircraft_range = np.vstack(aircraft_range).astype(np.float64); aircraft_range = np.round(aircraft_range[:,0],3)
aircraft_stage_length = aircraft_statistics['Average Stage Length (km)'].to_numpy(); aircraft_stage_length = aircraft_stage_length[0:19]
aircraft_runways = aircraft_statistics['Take-off Length (m)'].to_numpy(); aircraft_runways = aircraft_runways[0:19]

# Airport Index Vector
file_path = open(os.path.join(directory + 'route_airport_index.csv'), 'r'); route_airports = pd.read_csv(file_path, header=None); 
route_airports = route_airports.to_numpy();
Airport_In = route_airports[:,0]-1
Airport_Out = route_airports[:,1]-1

# Daily RPKs and ASKs
file_path = open(os.path.join(directory + 'metrics.csv'), 'r'); metrics = pd.read_csv(file_path, header=None); 
metrics = metrics.to_numpy();

# ------------------------------------------------ #
# Step 2: Building and Solving the Optimization Problem
# ------------------------------------------------ #

# Load Factors Computation: Base Case
RPKs = np.sum(metrics[1,:])
ASKs = np.sum(metrics[0,:])
average_daily_LF = np.round(metrics[1,:]/metrics[0,:],3)

# Optimization Days Vector
#days_vec = np.arange(366)
reduction_vec = np.zeros([len(days_vec),1])
days_in = 0

# Starting the sensitivity iterations, proving multiple days
for day in days_vec:
    
    # Optimization Problem:
    m = gp.Model("Model") 
    
    # Number of flights on the O-D market operated by aircraft type a
    x = m.addMVar(shape=(l_routes,l_aircraft), lb=0.0, vtype=GRB.INTEGER, name="x") # Adding variables
        
    # Objective Function:
    # Minimize the fuel consumption
    m.setObjective(np.transpose(route_distance)@x@aircraft_fuel, GRB.MINIMIZE)
    
    # Constraints:
    # 1) At least the same number of passengers must be carried on each route
    constraint_1 = m.addConstr(x@aircraft_seats >= daily_passengers[:,day])
    
    # 2) At least a minimum system-wide load factor must be attained
    target_LF = 0.01 + math.ceil(100*metrics[1,day]/metrics[0,day])/100
    RPKs = np.transpose(route_distance)@daily_passengers[:,day]
    ASKs = np.transpose(route_distance)@(x@aircraft_seats)
    constraint_2 = m.addConstr(RPKs <= target_LF*ASKs)
    
    # 3) The total flying time can't exceed the actual flying time
    constraint_3 = m.addConstr(np.transpose(flight_time)@x <= np.transpose(daily_aircraft_flying_times[:,day]))
    
    # 4) Aircraft must possess sufficient range to cover the route
    logic_distance = np.zeros([l_routes,19])
    for a in range(0,19):
        logic_distance[:,a] = route_distance > aircraft_range[a]
    non_permissible_aircraft = np.nonzero(logic_distance)
    in_itinerary = non_permissible_aircraft[0].astype(int)
    in_aircraft = non_permissible_aircraft[1].astype(int)
    constraint_4 = m.addConstr(x[in_itinerary,in_aircraft] == 0)
    
    # 5) The number of aircraft of type a into airport o must be equal to the 
    # number of aircraft of the same type a out of airport o 
    for node in range(0,169):
        o_in = np.argwhere(Airport_In == node).astype(int);
        d_in = np.argwhere(Airport_Out == node).astype(int);
        m.addConstr(sum(x[o_in,:]) == sum(x[d_in,:]))
        
    # 6) An aircraft can only fly into an airport with a sufficiently long runway
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
    m.Params.Presolve = 2 
    m.Params.MIPFocus= 1
    m.Params.Cuts= 1
    m.Params.Heuristics = 0.98
    m.Params.NoRelHeurTime = 500
    m.Params.MIPGap= 0.005
    
    # Solving the optimization problem
    m.optimize() 
    x_solution = x.X
    
    # Calculating the fuel reduction for the given load factor
    original_fuel_consumption = np.sum(yearly_fuel)
    new_fuel_consumption = np.sum(x_solution@aircraft_fuel*route_distance)*1000
    reduction_vec[days_in] = 100*(new_fuel_consumption - original_fuel_consumption)/original_fuel_consumption
    
    # Updating the LF index
    days_in = days_in + 1
    