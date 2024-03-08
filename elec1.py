import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import Canvas, Scrollbar
import random
from scipy.stats import weibull_min
from operator import itemgetter
from copy import deepcopy
import math

# PSO parameters
n_particles = 30
n_hours = 24
max_iter = 100
w = 0.5  # inertia
c1 = 1.5  # cognitive component
c2 = 1.5  # social component

# Genetic Algorithm parameters
pop_size = 20
mutation_rate = 0.2
crossover_rate = 0.8

# Environmental constraint example
some_environmental_limit = 2.0  # in MW


# Function to simulate wind power
def simulate_complex_wind():
    mean_wind_speed = 12  # in m/s
    turbulence_intensity = 0.1  # Ratio of standard deviation to the mean
    downtime_hours = 2  # 2 hours of downtime for maintenance
    shape_factor = 2.0  # Weibull shape factor
    scale_factor = mean_wind_speed  # Weibull scale factor

    # Generate wind speeds with Weibull distribution
    wind_speed = weibull_min.rvs(shape_factor, scale=scale_factor, size=24)
    
    # Seasonal variation
    seasonal_factor = 1 + 0.2 * np.sin((2 * np.pi / 365) * random.randint(1, 365))
    wind_speed *= seasonal_factor
    
    # Turbulence
    wind_speed += np.random.normal(0, mean_wind_speed * turbulence_intensity, 24)
    
    # Downtime for maintenance
    wind_speed[0:downtime_hours] = 0
    
    # Air density (kg/m^3)
    air_density = np.random.uniform(1.15, 1.225)
    
    # Turbine parameters
    gearbox_efficiency = 0.95  # Gearbox efficiency
    generator_efficiency = 0.97  # Generator efficiency
    rated_power = 2.5  # in MW
    cut_in_speed = 3.5  # in m/s
    cut_out_speed = 25  # in m/s
    rated_wind_speed = 12  # in m/s

    # Power Curve
    P_wind = np.piecewise(wind_speed, 
                          [wind_speed < cut_in_speed, 
                           (wind_speed >= cut_in_speed) & (wind_speed < rated_wind_speed), 
                           (wind_speed >= rated_wind_speed) & (wind_speed < cut_out_speed), 
                           wind_speed >= cut_out_speed], 
                          [0, 
                           lambda x: 0.5 * air_density * np.pi * (2 ** 2) * (x ** 3) * gearbox_efficiency * generator_efficiency,
                           rated_power * gearbox_efficiency * generator_efficiency,
                           0])
    
    return P_wind

# Function to simulate solar power
def simulate_solar():
    solar_constant = 1361  # Solar constant in W/m^2
    panel_area = 2.0  # Solar panel area in m^2
    panel_efficiency = 0.2  # Efficiency of solar panels
    temp_coefficient = -0.005  # Temperature coefficient of panel (%/C)
    nominal_temperature = 25  # Nominal temperature of panel in Celsius
    ambient_temperature = np.random.normal(25, 5, 24)  # Random ambient temperature for each hour
    
    latitude = 40.0  # Latitude of the location
    tilt_angle = latitude - 10  # Tilt angle of solar panel
    solar_angle = 30 + 40 * np.sin(np.pi * np.linspace(0, 1, 24))  # Corrected solar angle throughout the day
    
    weather_conditions = ['clear', 'cloudy', 'rainy']
    weather_today = random.choice(weather_conditions)
    
    days_in_year = 365
    day_of_year = random.randint(1, days_in_year)
    seasonal_factor = 1 + 0.3 * np.sin((2 * np.pi / days_in_year) * day_of_year)
    
    dust_factor = np.random.uniform(0.9, 1)
    degradation_factor = np.random.uniform(0.95, 1)
    
    inverter_loss = 0.05  # 5% loss in inverter
    cable_loss = 0.02  # 2% loss in cables
    
    P_solar = np.zeros(24)
    
    for t in range(24):
        angle_of_incidence = abs(solar_angle[t] - tilt_angle)
        angle_factor = np.cos(np.radians(angle_of_incidence))
        
        irradiance = solar_constant * np.sin(np.pi * solar_angle[t] / 180) * seasonal_factor
        
        if weather_today == 'clear':
            weather_factor = 1
        elif weather_today == 'cloudy':
            weather_factor = 0.6
        else:  # rainy
            weather_factor = 0.2
        
        efficiency_loss = (ambient_temperature[t] - nominal_temperature) * temp_coefficient
        adjusted_efficiency = panel_efficiency + efficiency_loss / 100
        
        system_loss = inverter_loss + cable_loss
        
        P_solar[t] = (irradiance * panel_area * adjusted_efficiency * angle_factor * 
                      weather_factor * dust_factor * degradation_factor * (1 - system_loss))
        
    return P_solar

# Function to simulate battery and load
def simulate_battery_load():
    battery_capacity = 100  # in kWh
    max_charging_rate = 20  # in kW
    max_discharging_rate = 20  # in kW
    state_of_charge = 50  # Initial state in kWh
    load = np.random.rand(24) * 100  # Dummy load data
    return battery_capacity, max_charging_rate, max_discharging_rate, state_of_charge, load

# Fitness function
def fitness(particle, P_wind, P_solar, load, battery_capacity, max_discharging_rate):
    state_of_charge = 50  # Initial state in kWh
    grid_power_sum = 0
    renewable_sum = 0
    reliability_metric = 0

    for t in range(n_hours):
        battery_action = np.clip(particle[t], -max_discharging_rate, max_discharging_rate)
        state_of_charge += battery_action
        state_of_charge = np.clip(state_of_charge, 0, battery_capacity)

        grid_power = max(0, load[t] - (P_wind[t] + P_solar[t] + battery_action))
        grid_power_sum += grid_power
        renewable_sum += P_wind[t] + P_solar[t]
        reliability_metric += state_of_charge
        
    
    environmental_penalty = 0
    if any(P_wind > some_environmental_limit):
        environmental_penalty = 1000
    
    return grid_power_sum + environmental_penalty, -renewable_sum, -reliability_metric

# New function to perform Genetic Algorithm
def genetic_algorithm():
    global w, c1, c2
    global pareto_front  # Declare pareto_front as global
    population = [(random.random(), random.random(), random.random()) for _ in range(pop_size)]
    
    for _ in range(10):  # Run GA for 10 generations
        # Evaluate fitness of each individual
        fitnesses = []
        for individual in population:
            w, c1, c2 = individual
            run_simulation()  # Will use the global pareto_front
            fitnesses.append(-len(pareto_front))  # Maximize pareto_front length
            
        # Selection
        selected_parents = sorted(zip(population, fitnesses), key=itemgetter(1), reverse=True)[:pop_size//2]
        
        # Crossover
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.choices(selected_parents, k=2, weights=[fit for _, fit in selected_parents])
            if random.random() < crossover_rate:
                child = [(a + b) / 2 for a, b in zip(parent1[0], parent2[0])]
            else:
                child = random.choice([parent1[0], parent2[0]])
            
            # Mutation
            if random.random() < mutation_rate:
                child = [param + random.gauss(0, 0.1) for param in child]
                
            new_population.append(child)
        
        population = new_population

    # Update global PSO parameters with the best-found values
    best_individual = max(selected_parents, key=itemgetter(1))[0]
    w, c1, c2 = best_individual

# Update Pareto front
def update_pareto_front(pareto_front, new_particle, new_fitness):
    dominated = False
    to_remove = []
    for i, (particle, fitness) in enumerate(pareto_front):
        if all(new_fitness >= np.array(fitness)):
            dominated = True
            break
        elif all(new_fitness <= np.array(fitness)):
            to_remove.append(i)
    
    for i in reversed(to_remove):
        pareto_front.pop(i)
    
    if not dominated:
        pareto_front.append((new_particle, new_fitness))
        
def plot_pareto_front(pareto_front):
    fig_pareto = Figure(figsize=(5, 5), dpi=100)
    ax_pareto = fig_pareto.add_subplot(111)
    ax_pareto.scatter([fit[0] for _, fit in pareto_front], [fit[1] for _, fit in pareto_front], c='r', marker='o')
    ax_pareto.set_title('Pareto Front')
    ax_pareto.set_xlabel('Grid Power (Objective 1)')
    ax_pareto.set_ylabel('Renewable Use (Objective 2)')
    return fig_pareto

# Function to run the simulation
def run_simulation():
    
    # Remove any existing widgets in the frame
    for widget in frame.winfo_children():
        widget.destroy()
        
    P_wind = simulate_complex_wind()
    P_solar = simulate_solar()
    battery_capacity, max_charging_rate, max_discharging_rate, state_of_charge, load = simulate_battery_load()

    particle_positions = np.random.uniform(-max_discharging_rate, max_charging_rate, (n_particles, n_hours))
    particle_velocities = np.random.uniform(-1, 1, (n_particles, n_hours))
    pbest_positions = np.copy(particle_positions)
    pareto_front = []
    
    best_fitness = float('inf')  # Initialize to a large value

    for i in range(n_particles):
        initial_fitness = fitness(particle_positions[i], P_wind, P_solar, load, battery_capacity, max_discharging_rate)
        update_pareto_front(pareto_front, particle_positions[i], initial_fitness)
    
    for iter in range(max_iter):
        for i in range(n_particles):
            gbest_particle, _ = pareto_front[np.random.randint(len(pareto_front))]

            inertia = w * particle_velocities[i]
            cognitive = c1 * np.random.random() * (pbest_positions[i] - particle_positions[i])
            social = c2 * np.random.random() * (gbest_particle - particle_positions[i])
            
            new_velocity = inertia + cognitive + social
            particle_velocities[i] = np.clip(new_velocity, -1, 1)
            new_position = particle_positions[i] + particle_velocities[i]
            particle_positions[i] = np.clip(new_position, -max_discharging_rate, max_charging_rate)

            new_fitness = fitness(particle_positions[i], P_wind, P_solar, load, battery_capacity, max_discharging_rate)
            update_pareto_front(pareto_front, particle_positions[i], new_fitness)
            if new_fitness[0] < best_fitness:
                best_fitness = new_fitness[0]
                pbest_positions[i] = particle_positions[i]
                
    # Plotting the results
    fig, ax = plt.subplots(4, 1, figsize=(8, 12))
    
    ax[0].plot(P_wind, label='Wind Power', color='b', linestyle='-')
    ax[0].set_title('Simulated Wind Energy')
    ax[0].set_xlabel('Hour')
    ax[0].set_ylabel('Power (kW)')
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].plot(P_solar, label='Solar Power', color='r', linestyle='--')
    ax[1].set_title('Simulated Solar Energy')
    ax[1].set_xlabel('Hour')
    ax[1].set_ylabel('Power (kW)')
    ax[1].grid(True)
    ax[1].legend()
    
    ax[2].plot(load, label='Load', color='g', linestyle='-.')
    ax[2].set_title('Simulated Load')
    ax[2].set_xlabel('Hour')
    ax[2].set_ylabel('Load (kW)')
    ax[2].grid(True)
    ax[2].legend()

    ax[3].plot(pbest_positions[np.argmin([fit[0] for _, fit in pareto_front])], label='Optimized Output', color='m', linestyle=':')
    ax[3].set_title('Optimized Output')
    ax[3].set_xlabel('Hour')
    ax[3].set_ylabel('Power (kW)')
    ax[3].grid(True)
    ax[3].legend()


    # Embed the plot in the Tkinter window
    canvas_frame = ttk.Frame(frame)
    canvas_frame.grid(row=1, column=0, columnspan=4, sticky='nsew')

    tk_canvas = Canvas(canvas_frame, bg='white')
    tk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    
    canvas = FigureCanvasTkAgg(fig, master=tk_canvas)
    canvas_widget = canvas.get_tk_widget()

    tk_canvas.create_window((0, 0), window=canvas_widget, anchor="nw")

    # Add a scrollbar
    scrollbar = Scrollbar(canvas_frame, orient="vertical", command=tk_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tk_canvas.config(yscrollcommand=scrollbar.set)

    canvas_widget.update_idletasks()
    canvas_height = canvas_widget.winfo_height()
    tk_canvas.config(scrollregion=(0, 0, 0, canvas_height))
 
    canvas.draw()
    
    #display Pareto front
    fig_pareto = plot_pareto_front(pareto_front)
    canvas_pareto = FigureCanvasTkAgg(fig_pareto, master=tk_canvas)
    canvas_pareto_widget = canvas_pareto.get_tk_widget()
    tk_canvas.create_window((0, canvas_height), window=canvas_pareto_widget, anchor="nw")

    canvas_pareto_widget.update_idletasks()
    pareto_height = canvas_pareto_widget.winfo_height()
    tk_canvas.config(scrollregion=(0, 0, 0, canvas_height + pareto_height))

    canvas_pareto.draw()
    
# Function to run the simulation with UI input
def run_simulation_with_ui():
    global n_particles, max_iter, w, c1, c2
    n_particles = int(n_particles_entry.get())
    max_iter = int(max_iter_entry.get())
    w = float(w_entry.get())
    c1 = float(c1_entry.get())
    c2 = float(c2_entry.get())
    
    # Run Genetic Algorithm to adaptively tune PSO parameters
    genetic_algorithm()
    run_simulation()  # Call the original run_simulation function


 # Create the main window
root = tk.Tk()
root.geometry("800x600")  # Adjust the size of the main window
root.title("Wind-Solar Hybrid Energy Storage System Simulation")

# Create a frame
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

frame.columnconfigure(0, weight=1)
frame.rowconfigure(1, weight=1)

# Add widgets for user interaction
run_button = ttk.Button(frame, text="Run Simulation", command=run_simulation_with_ui)
run_button.grid(row=0, column=0, columnspan=4)

# UI for weather selection
weather_var = tk.StringVar(frame)
weather_var.set('clear')  # default value
ttk.Label(frame, text="Weather: ").grid(row=4, column=0)
weather_menu = ttk.OptionMenu(frame, weather_var, 'clear', 'cloudy', 'rainy')
weather_menu.grid(row=4, column=1)

# Reset button to reset all input fields
def reset_fields():
    n_particles_entry.delete(0, tk.END)
    n_particles_entry.insert(0, '30')
    max_iter_entry.delete(0, tk.END)
    max_iter_entry.insert(0, '100')
    w_entry.delete(0, tk.END)
    w_entry.insert(0, '0.5')
    c1_entry.delete(0, tk.END)
    c1_entry.insert(0, '1.5')
    c2_entry.delete(0, tk.END)
    c2_entry.insert(0, '1.5')
    weather_var.set('clear')

reset_button = ttk.Button(frame, text="Reset", command=reset_fields)
reset_button.grid(row=4, column=3)

# UI for updating PSO parameters
ttk.Label(frame, text="Number of particles: ").grid(row=1, column=0)
n_particles_entry = ttk.Entry(frame)
n_particles_entry.insert(0, str(n_particles))
n_particles_entry.grid(row=1, column=1)

ttk.Label(frame, text="Max iterations: ").grid(row=1, column=2)
max_iter_entry = ttk.Entry(frame)
max_iter_entry.insert(0, str(max_iter))
max_iter_entry.grid(row=1, column=3)

ttk.Label(frame, text="w (inertia): ").grid(row=2, column=0)
w_entry = ttk.Entry(frame)
w_entry.insert(0, str(w))
w_entry.grid(row=2, column=1)

ttk.Label(frame, text="c1 (cognitive): ").grid(row=2, column=2)
c1_entry = ttk.Entry(frame)
c1_entry.insert(0, str(c1))
c1_entry.grid(row=2, column=3)

ttk.Label(frame, text="c2 (social): ").grid(row=3, column=0)
c2_entry = ttk.Entry(frame)
c2_entry.insert(0, str(c2))
c2_entry.grid(row=3, column=1)

# Run the Tkinter event loop
root.mainloop()
