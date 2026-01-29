# ------------------------------------------------ #
# Fuel-Optimal Fleet Assignment (FOFA) Model - V3
# ------------------------------------------------ #
# DESCRIPTION:
# The following code implements the fuel-optimal fleet 
# assignment (FOFA) framework within the Brazilian domestic 
# air transportation system.
#
# VERSION:
# This code implements the framework on for the entirety
# of 2024 (i.e., aggregating all 366 days). 
# The framework can either be implemented separately on 
# the three largest domestic Brazilian airlines, or 
# considering all eight domestic airlines on aggregate 
# as a combined national airline.
#
# OUTPUTS:
# - Estimated fuel consumption reduction relative to 
# the 2024 baseline.       
# - Optimized average stage lengths (in km) for each
# aircraft type operated by the chosen airline.
# ------------------------------------------------ #

# ------------------------------------------------ #
# Step 0: User-editable Parameters
# ------------------------------------------------ #

# --> Data Directory 
directory = '/Users/andyeske/Desktop/Fall 2025/Optimization Methods/Project/Processed Datasets/'

# --> Airline
# Options: 
# (1): AD - Azul Linhas Aereas Brasileiras 
#      8 aircraft types, 30.2% market share by ASKs
#      Minimum LF = 82.2%

# (2): G3 - GOL Linhas Aereas
#      3 aircraft types, 30.5% market share by ASKs
#      Minimum LF = 84.7%

# (3): JJ - LATAM Brasil 
#      6 aircraft types, 38.7% market share by ASKs
#      Minimum LF = 83.6%

# (4): RG - Combined Brazilian National Airline (19 aircraft types)
#      19 aircraft types, including aicraft from:
#      -> 0S - Sideral Linhas Aereas
#      -> 2F - Azul Conecta
#      -> 2Z - Voepass
#      -> 7M - MAP Linhas Aereas
#      -> E4 - Abaete Aviacao
#      Minimum LF = 83.4%

Option = 1

# --> System-wide Sensitivity Parameters
# Establish a maximum load factor (max_LF)
max_LF = 82.2

# Enforce a maximum aircraft availability percentage (max_avail)
max_avail = 101

# Fix the baseline fleet assignment on certain route segments (fixed_routes)
fixed_routes = []

# ------------------------------------------------ #
# Step 1: Importing Packages
# ------------------------------------------------ #
import pandas as pd
import math
import os
import time
import numpy as np
import warnings
import gurobipy as gp
from gurobipy import GRB

# ------------------------------------------------ #
# Step 1: Importing and Processing the Relevant Datasets
# ------------------------------------------------ #

# User-Option
if Option == 1: # Azul Linhas Aereas Brasileiras
    Airline = 4
    name = 'Azul Linhas Aereas Brasileiras'
    code = 'AD'
    
elif Option == 2: # GOL Linhas Aereas
    Airline = 6
    name = 'GOL Linhas Aereas'
    code = 'G3'
    
elif Option == 3: # LATAM Brasil
    Airline = 7
    name = 'LATAM Brasil'
    code = 'JJ'
    
else: # Combined Brazilian National Airline
    Airline = 8
    name = 'Combined Brazilian National Airline'
    code = 'RG'
    
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
aircraft_runways = aircraft_statistics['Take-off Length (m)'].to_numpy(); aircraft_runways = aircraft_runways[0:19]

# Baseline Fleet Assignment
file_path = '/Users/andyeske/Desktop/Fall 2025/Optimization Methods/Project/Processed Datasets/Baseline Fleet Assignment.xlsx'; 
baseline_fleet_assignment = pd.read_excel(file_path,header=None,sheet_name=code); baseline_fleet_assignment = baseline_fleet_assignment.to_numpy();

# Airport Index Vector
file_path = open(os.path.join(directory + 'route_airport_index.csv'), 'r'); route_airports = pd.read_csv(file_path, header=None); 
route_airports = route_airports.to_numpy();
Airport_In = route_airports[:,0]-1
Airport_Out = route_airports[:,1]-1

# Passengers Transported on each Route
file_path = open(os.path.join(directory + 'airline_specific_yearly_route_passengers.csv'), 'r'); total_passengers = pd.read_csv(file_path, header=None); 
total_passengers = total_passengers.to_numpy(); 
if Airline > 7: 
    total_passengers = np.round(np.sum(total_passengers,1)); 
else:
    total_passengers = np.round(total_passengers[:,Airline]); 
total_passengers = total_passengers.astype(np.int64)
l_routes = 1413
 
# Departures on each Route
total_departures = np.sum(baseline_fleet_assignment,1)
total_departures = total_departures*(total_passengers > 0)

# Fuel Consumed on each Route
file_path = open(os.path.join(directory + 'airline_specific_yearly_fuel_consumed.csv'), 'r'); total_fuel = pd.read_csv(file_path, header=None); 
total_fuel = total_fuel.to_numpy(); 
if Airline > 7: 
    total_fuel = np.sum(total_fuel,1)
else:
    total_fuel = total_fuel[:,Airline]
total_fuel = np.round(total_fuel,3)
total_fuel = total_fuel*(total_passengers > 0)
     
# Aircraft Total Flying Time
file_path = open(os.path.join(directory + 'airline_specific_yearly_aircraft_flying_times.csv'), 'r'); total_aircraft_flying_times = pd.read_csv(file_path, header=None); 
total_aircraft_flying_times = total_aircraft_flying_times.to_numpy(); 
if Airline > 7: 
    total_aircraft_flying_times = np.sum(total_aircraft_flying_times,1)
else:
    total_aircraft_flying_times = total_aircraft_flying_times[:,Airline]
total_aircraft_flying_times = np.round(total_aircraft_flying_times)
l_aircraft = 19

# Total RPKs and ASKs
file_path = open(os.path.join(directory + 'airline_specific_yearly_metrics.csv'), 'r'); metrics = pd.read_csv(file_path, header=None); 
metrics = metrics.to_numpy();
if Airline > 7: 
    RPKs = np.sum(metrics[1,:])
    ASKs = np.sum(metrics[0,:])
else:
    RPKs = np.sum(metrics[1,Airline])
    ASKs = np.sum(metrics[0,Airline])   
average_total_LF = np.round(RPKs/ASKs,3)

# Adjusting Metric Scales
route_distance = route_distance/1000
aircraft_range = aircraft_range/1000
total_fuel = total_fuel/1000
RPKs = RPKs/1000; ASKs = ASKs/1000; 

# ------------------------------------------------ #
# Step 2: Building and Solving the Optimization Problem
# ------------------------------------------------ #

# Load Factor Sensitivity Analysis
avg_stage_length_vec = np.zeros([l_aircraft+1,1])
avg_stage_length_vec_baseline = np.zeros([l_aircraft+1,1])
   
# Optimization Problem Statistics
non_zero_routes = np.sum(total_passengers > 0);
route_in = np.nonzero(total_passengers > 0)[0]
non_zero_airports_in = route_airports[route_in,:]
non_zero_airports = np.size(np.unique(non_zero_airports_in))
non_zero_aircraft = np.sum(total_aircraft_flying_times > 0);
aircraft_in = np.nonzero(total_aircraft_flying_times > 0)[0]
num_vars = non_zero_routes*non_zero_aircraft
num_constraints = 0
 
# Optimization Problem:
m = gp.Model("Model") 

# Decision Variable: Number of frequencies on the O-D market operated by aircraft type a
x = m.addMVar(shape=(l_routes,l_aircraft), lb=0.0, vtype=GRB.INTEGER, name="x") # Adding variables
    
# Objective Function: Minimize the system-wide fuel consumption
m.setObjective(np.transpose(route_distance)@x@aircraft_fuel, GRB.MINIMIZE)

# Constraints:
# 0) Fixing routes of the baseline solution
constraint_0 = m.addConstr(x[fixed_routes,:] >= baseline_fleet_assignment[fixed_routes,:])
num_constraints = num_constraints + np.size(fixed_routes)*non_zero_aircraft
    
# 1) At least the same number of passengers must be carried on each route
constraint_1 = m.addConstr(x@aircraft_seats >= total_passengers, name="constraint_1")
num_constraints = num_constraints + non_zero_routes

# 2) At least the same number of flights on each route must be preserved
constraint_2 = m.addConstr(x.sum(1) >= total_departures, name="constraint_2")
num_constraints = num_constraints + non_zero_routes

# 3) At least a minimum system-wide load factor must be attained
new_ASKs = np.transpose(route_distance)@x@aircraft_seats
constraint_3 = m.addConstr(RPKs <= max_LF*new_ASKs/100, name="constraint_3")
num_constraints = num_constraints + 1

# 4) The total flying time can't exceed the actual flying time
constraint_4 = m.addConstr(np.transpose(flight_time)@x <= np.transpose(max_avail*total_aircraft_flying_times/100), name="constraint_4")
num_constraints = num_constraints + non_zero_aircraft

# 5) Aircraft must possess sufficient range to cover the route
logic_distance = np.zeros([l_routes,l_aircraft])
for a in range(0,l_aircraft):
    logic_distance[:,a] = route_distance > aircraft_range[a]
non_permissible_aircraft = np.nonzero(logic_distance)
in_itinerary = non_permissible_aircraft[0].astype(int)
in_aircraft = non_permissible_aircraft[1].astype(int)
if len(in_itinerary) > 0:
    constraint_5 = m.addConstr(x[in_itinerary,in_aircraft] == 0, name="constraint_5")
for a in range(0,np.size(non_zero_aircraft)):
    num_constraints = num_constraints + np.sum(route_distance[route_in] > aircraft_range[a])

# 6) The number of aircraft of type a into airport o must be equal to the 
# number of aircraft of the same type a out of airport o 
for node in range(0,169):
    o_in = np.argwhere(Airport_In == node).astype(int);
    d_in = np.argwhere(Airport_Out == node).astype(int);
    if len(o_in) > 0:
        m.addConstr(sum(x[o_in,:]) == sum(x[d_in,:]))
        num_constraints = num_constraints + 1
    
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
    
    runway_in = np.nonzero(1*(aircraft_runways[aircraft_in] > runway))[0]
    num_constraints = num_constraints + len(runway_in)

# --> Modifying model parameters (adjust as needed)
m.Params.Presolve = 0
m.Params.MIPFocus= 1
m.Params.Cuts= 1
m.Params.Heuristics = 0.98
m.Params.NoRelHeurTime = 1000
m.Params.MIPGap= 0.001

# Solving the optimization problem
m.optimize() 
x_solution = x.X

# Calculating the fuel reduction for the given load factor
original_fuel_consumption = np.sum(total_fuel)
new_fuel_consumption = np.sum(x_solution@aircraft_fuel*route_distance)
reduction_vec = 100*(new_fuel_consumption - original_fuel_consumption)/original_fuel_consumption

# Calculating the new average stage length
aircraft_names = ['A320NEO','A321NEO','A319CEO','A320CEO','A321CEO','A330-200','A330-900',
                  'ATR42-500','ATR72-500','ATR72-600','B737-8','B737-300','B737-500',
                  'B737-700','B737-800','B787-9','C208','E195','E195-E2','Airline Average']
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    avg_stage_length_vec[0:19,0] = np.transpose(np.transpose(route_distance)@x_solution/np.sum(x_solution,0))
    avg_stage_length_vec[19,0] = np.sum(np.transpose(route_distance)@x_solution)/np.sum(x_solution)
    avg_stage_length_vec_baseline[0:19,0] = np.transpose(np.transpose(route_distance)@baseline_fleet_assignment/np.sum(baseline_fleet_assignment,0))
    avg_stage_length_vec_baseline[19,0] = np.sum(np.transpose(route_distance)@baseline_fleet_assignment)/np.sum(baseline_fleet_assignment)

    # Calculating constraints slack
    solution_ASKs = np.transpose(route_distance)@x_solution@aircraft_seats
    slack_LF = max_LF*solution_ASKs/100 - RPKs
    slack_LF_per = round(100*slack_LF/(max_LF*solution_ASKs/100),2)
    
    slack_uti = np.transpose(max_uti*total_aircraft_flying_times/100) - np.transpose(flight_time)@x_solution
    slack_uti_per = np.round(100*slack_uti/np.transpose(max_uti*total_aircraft_flying_times/100),2)

print("")
print("---------------- Problem Statistics -----------------")
print("")
print("---------------------- Inputs -----------------------")
print("--> Airline: " + name + " (" + code + ")")
print("--> Maximum System-wide Load Factor (LF): " + str(max_LF) + "%")
print("--> Maximum Aircraft Utilization: " + str(max_uti) + "%")
print("--> Fixed Route: " + str(fixed_routes))
print("")
print("---------------------- Outputs ----------------------")
print("--> Airline Statistics:")
print("----> Number of Routes: " + str(non_zero_routes))
print("----> Number of Airports: " + str(non_zero_airports))
print("----> Number of Aircraft Types: " + str(non_zero_aircraft))
print("----> Number of Passengers: " + str(np.sum(np.round(total_passengers/1000000,2))) + ' million')
print("----> Revenue Passenger Miles (RPKs): " + str(np.round(RPKs/1000000,2)) + ' billion')
print("----> Revenue Passenger Miles (ASKs): " + str(np.round(ASKs/1000000,2)) + ' billion')
print("----> Baseline Load Factor (LF): " + str(np.round(100*RPKs/ASKs,1)) + '%')
print("")
print("--> Fuel Consumption (FC):")
print("----> Optimized FC (L): " + str(round(new_fuel_consumption/1000)) + " million L" + " (Baseline FC (L): " + str(round(original_fuel_consumption/1000)) + " million L)")
print("----> FC Reduction (%): " + str(round(reduction_vec,2)) + "%")
print("")
print("--> Average Stage Lengths (ASL):")
for k in range(0,19):
    if np.isnan(slack_uti_per[k]) == 0:
        if slack_uti_per[k] == 100:
            print("----> " + aircraft_names[k] + " Optimized ASL (km): 0 km " + "(Baseline ASL (km): " + str(int(np.round(avg_stage_length_vec_baseline[k,0]*1000))) + " km)")
        else:
            print("----> " + aircraft_names[k] + " Optimized ASL (km): " + str(int(np.round(avg_stage_length_vec[k,0]*1000))) + " km " + "(Baseline ASL (km): " + str(int(np.round(avg_stage_length_vec_baseline[k,0]*1000))) + " km)")
print("----> " + aircraft_names[19] + " Optimized ASL (km): " + str(int(np.round(avg_stage_length_vec[19,0]*1000))) + " km " + "(Baseline ASL (km): " + str(int(np.round(avg_stage_length_vec_baseline[19,0]*1000))) + " km)")
print("")
print("-------------------- Optimization --------------------")        
print("--> Overall Statistics:")
print("----> Number of Variables: " + str(num_vars))
print("----> Number of Constraints: " + str(num_constraints))
print("----> Runtime (s): " + str(np.round(m.Runtime,2)) + 's')
print("----> Optimality Gap (%): " + str(np.round(100*m.MIPGap,2)) + '%')
print("")

print("--> Key Constraints Slack Percentage:")
print("----> LF Slack (%): " + str(slack_LF_per) + '%')
for k in range(0,19):
    if np.isnan(slack_uti_per[k]) == 0:
        print("----> " + aircraft_names[k] + " Utilization Slack (%): " + str(slack_uti_per[k]) + '%')
print("")
print("-----------------------------------------------------")
    
