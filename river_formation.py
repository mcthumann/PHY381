import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation parameters
lattice_size = (10, 20)  # Size of the lattice (x, y)
iterations = 100       # Number of iterations to run the simulation
precipitation_rate = .2 # Rate at which water is added to random cells
erosion_rate = 0.1        # Erosion rate (controls how much land is eroded when water flows)
land_initial_height = 10   # Initial height of the land (arbitrary units)

# Create the lattice, initialized with land heights and no water
land = np.full(lattice_size, land_initial_height, dtype=float)
water = np.zeros(lattice_size, dtype=float)

def add_precipitation(water):
    precipitation_sites = np.random.random(lattice_size) < precipitation_rate
    water[precipitation_sites] += 1
    return np.argwhere(precipitation_sites)

# Iterative function to simulate water flow from a specific site

def flow(x, y, land, water):
    # Stack to mimic recursion
    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()  # Current site to process

        # Get the current surface height (land + water) at this site
        surface_height = land[cx, cy] + water[cx, cy]

        # Define the 4 neighboring sites (periodic boundary conditions on x)
        neighbors = [(cx - 1) % lattice_size[0], (cx + 1) % lattice_size[0], (cy - 1) % lattice_size[1], (cy + 1) % lattice_size[1]]

        # Check neighboring cells and move water if possible
        for nx, ny in [(neighbors[0], cy), (neighbors[1], cy), (cx, neighbors[2]), (cx, neighbors[3])]:
            # No need to check boundaries due to periodic conditions
            neighbor_surface_height = land[nx, ny] + water[nx, ny]
            if neighbor_surface_height < surface_height and water[cx, cy] > 0:
                # Calculate how much water can flow to the neighbor
                flow_amount = min(water[cx, cy], surface_height - neighbor_surface_height)

                # Update water levels
                water[cx, cy] -= flow_amount
                water[nx, ny] += flow_amount

                # Erode land at the destination site
                if land[nx, ny] < land[cx, cy]:
                    land[nx, ny] = max(land[nx, ny] - erosion_rate * flow_amount, 0)

                # Push the neighbor to the stack to continue flowing from there
                stack.append((nx, ny))

def step(land, water):
    # Perform precipitation
    precip_sites = add_precipitation(water)

    # Flow water starting from each precipitation site
    for x, y in precip_sites:
        flow(x, y, land, water)

    return land, water

# Run the simulation for a number of iterations
for i in range(iterations):
    land, water = step(land, water)
    if i % 10 == 0:
        print(f"Iteration {i}/{iterations}")

# Plot the final state of the system
plt.figure(figsize=(10, 6))
plt.imshow(land, cmap='terrain')
plt.colorbar(label='Height (land)')
plt.title('River Network After Simulation')
plt.show()



