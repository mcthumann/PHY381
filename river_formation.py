import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

# Simulation parameters
lattice_size = (50, 50)  # Size of the lattice (x, y)
iterations = 500     # Number of iterations to run the simulation
precipitation_rate = 0.1 # Rate at which water is added to random cells
erosion_rate = .05        # Erosion rate (controls how much land is eroded when water flows)
land_initial_height = 10   # Initial height of the land (arbitrary units)
delta = .1
# max_stack = 100
flow_per_rain = lattice_size[0]*2

# Create the lattice, initialized with land heights and no water
land = np.full(lattice_size, land_initial_height, dtype=float)
land += np.random.uniform(-delta, delta, size=lattice_size)
water = np.zeros(lattice_size, dtype=float)
flow_sum = np.zeros(lattice_size, dtype=float)

def add_precipitation(water):
    precipitation_sites = np.random.random(lattice_size) < precipitation_rate
    water[precipitation_sites] += delta
    # plt.figure(figsize=(10, 6))
    # plt.imshow(water)
    # plt.show()
    return np.argwhere(precipitation_sites)

def flow(x, y, land, water):
    # Stack to mimic recursion
    stack = [(x, y)]

    while stack:
        # plt.figure(figsize=(10, 6))
        # plt.imshow(water)
        # plt.show()
        cx, cy = stack.pop()  # Current site to process
        # print("New cell " + str(cx) + " " + str(cy))
        # Get the current surface height (land + water) at this site
        surface_height = land[cx, cy] + water[cx, cy]

        # Calculate neighbor positions
        neighbors = np.array([
            [(cx - 1)%lattice_size[0], cy],  # Left
            [(cx + 1)%lattice_size[0], cy],  # Right
            [cx, cy + 1],  # Up
            [cx, cy - 1],  # Down
            [(cx - 1) % lattice_size[0], cy - 1],  # Left Down
            [(cx + 1) % lattice_size[0], cy - 1],  # Right Down
            [(cx - 1) % lattice_size[0], cy + 1],  # Left Up
            [(cx + 1) % lattice_size[0], cy + 1]  # Right Up
        ])

        # Convert neighbors to integer type (if not already)
        neighbors = neighbors.astype(int)

        # Separate neighbor x and y coordinates
        nx = neighbors[:, 0]
        ny = neighbors[:, 1]

        # Initialize neighbor surface heights to 100
        neighbor_surface_heights = np.full(nx.shape, 100.0)

        # Identify valid indices where nx and ny are within bounds
        valid_indices = (
                (nx >= 0) & (nx < lattice_size[0]) &
                (ny >= 0) & (ny < lattice_size[1])
        )

        # For valid indices, compute surface heights from land and water arrays
        neighbor_surface_heights[valid_indices] = (
                land[nx[valid_indices], ny[valid_indices]] +
                water[nx[valid_indices], ny[valid_indices]]
        )

        # Get indices that would sort the surface heights in ascending order
        sorted_indices = np.argsort(neighbor_surface_heights)
        neighbor_ratios = [.8, .1, .05, .02, .01, .005, .0025, .0025]
        # Arrange the neighbors according to sorted surface heights
        sorted_neighbors = neighbors[sorted_indices]
        # Check neighboring cells and move water if possible

        for i in range(8):
            # Check if neighbor is within bounds
            nx, ny = sorted_neighbors[i]
            ratio = neighbor_ratios[i]
            neighbor_in_bounds = 0 <= ny < lattice_size[1]
            neighbor_land_height = None
            neighbor_surface_height = None
            neighbor_water_height = None
            if neighbor_in_bounds:
                # Neighbor is within bounds
                neighbor_surface_height = land[nx, ny] + water[nx, ny]
                neighbor_land_height = land[nx, ny]
                neighbor_water_height = water[nx, ny]
            elif ny < 0:
                # Neighbor is out of bounds (water can leave the system)
                neighbor_surface_height = -100000  # Adjust based on your model
                neighbor_land_height = -100000  # Assume no land out of bounds
                neighbor_water_height = 0
            elif ny >= lattice_size[1]:
                # Neighbor is out of bounds (water cant leave the system)
                neighbor_surface_height = 1000  # Adjust based on your model
                neighbor_land_height = 1000  # Assume no land out of bounds
                neighbor_water_height = 0
            else:
                print("Neighbor error")

            # Proceed if water can flow to the neighbor
            if neighbor_surface_height < surface_height and water[cx, cy] > 0:
                # Calculate how much water can flow to the neighbor
                # if the neighbor is so low that the sum of the water wont fill the gap of the land, send all the water
                flow_amount = 0
                if land[cx, cy] > neighbor_land_height:
                    if (neighbor_water_height+water[cx, cy]) < (land[cx, cy] - neighbor_land_height):
                        flow_amount = ratio*water[cx, cy]
                    elif (neighbor_water_height+water[cx, cy]) > (land[cx, cy] - neighbor_land_height):
                        flow_amount = ratio*(water[cx, cy] - ((water[cx, cy] + neighbor_water_height) - (land[cx, cy] - neighbor_land_height))/2)
                else: # the water is higher here but the land is not...
                    flow_amount = min(water[cx, cy], ratio*((surface_height - neighbor_surface_height)/2))

                # Update water levels at current site
                water[cx, cy] -= flow_amount
                flow_sum[cx, cy] += flow_amount

                if neighbor_in_bounds:
                    # Neighbor is within bounds, update water there
                    water[nx, ny] += flow_amount

                    # Erode land at the source site if the land at the destination is lower
                    if neighbor_land_height <= land[cx, cy]:
                        land[nx, ny] = land[nx, ny] - (erosion_rate * flow_amount)

                #if neighbor_in_bounds and water[nx, ny] > delta:
                    # Push the neighbor to the stack to continue flowing from there
                    #stack.append((nx, ny))


def step(land, water, flow_per_rain):
    # Perform precipitation
    precip_sites = add_precipitation(water)

    # Flow water starting from each precipitation site
    for _ in range(flow_per_rain):
        for x, y in precip_sites:
            flow(x, y, land, water)

    return land, water

# Run the simulation for a number of iterations
for i in range(iterations):
    land, water = step(land, water, flow_per_rain)
    if i % int(iterations/10) == 0:
        plt.figure(figsize=(10, 6))
        plt.imshow(land)
        plt.colorbar()
        plt.title("land after " + str(i) + " of " + str(iterations))
        plt.show()
        # plt.figure(figsize=(10, 6))
        # plt.imshow(water)
        # plt.colorbar()
        # plt.title("water after " + str(i) + " of " + str(iterations))
        # plt.show()
        # plt.figure(figsize=(10, 6))
        # plt.imshow(land+water)
        # plt.colorbar()
        # plt.title("surface after " + str(i) + " of " + str(iterations))
        # plt.show()
        plt.figure(figsize=(10, 6))
        plt.imshow(flow_sum, norm=LogNorm(), cmap='viridis')
        plt.colorbar()
        plt.title("flow_sum after " + str(i) + " of " + str(iterations))
        plt.show()
        print(f"Iteration {i}/{iterations}")

# Plot the final state of the system
plt.figure(figsize=(10, 6))
plt.imshow(land + water)
plt.colorbar(label='Height (land)')
plt.title('land, water After Simulation')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(land)
plt.colorbar(label='Height (land)')
plt.title('land After Simulation')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(water)
plt.colorbar(label='Height (land)')
plt.title('water After Simulation')
plt.show()



