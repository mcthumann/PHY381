import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
lattice_size = (50, 50)  # Size of the lattice (x, y)
iterations = 1000     # Number of iterations to run the simulation
precipitation_rate = 1/lattice_size[0] # Rate at which water is added to random cells
erosion_rate = .2        # Erosion rate (controls how much land is eroded when water flows)
land_initial_height = 10   # Initial height of the land (arbitrary units)
delta = .1

flow_per_rain = int(lattice_size[0]/2)

# Flat land
land = np.full(lattice_size, land_initial_height, dtype=float)

# Randomize land
# land += np.random.uniform(-delta, delta, size=lattice_size)

# Gradient land
# Create an array of deltas for each column
# column_deltas = delta * (np.arange(lattice_size[1]) + 1)
# Add the column_deltas to each column of the land array
# land += column_deltas[np.newaxis, :]

# Plot land
plt.figure(figsize=(10, 6))
plt.imshow(land)
plt.show()

water = np.zeros(lattice_size, dtype=float)
flow_sum = np.zeros(lattice_size, dtype=float)

def add_precipitation(water):
    precipitation_sites = np.random.random(lattice_size) < precipitation_rate
    water[precipitation_sites] += delta
    # plt.figure(figsize=(10, 6))
    # plt.imshow(water)
    # plt.show()
    return np.argwhere(precipitation_sites)

def flow(cx, cy, land, water):

    if water[cx, cy] <= 0:
        return

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
    neighbor_surface_heights = np.full(nx.shape, 10000.0)

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
        elif ny<0:
            # Neighbor is out of bounds (water can leave the system)
            neighbor_surface_height = land[cx, cy] - delta  # Adjust based on your model
            neighbor_land_height = land[cx, cy] - delta # Assume no land out of bounds
            neighbor_water_height = 0
        elif ny >= lattice_size[1]:
            # Neighbor is out of bounds (water cant leave the system)
            neighbor_surface_height = 10000  # Adjust based on your model
            neighbor_land_height = 10000  # Assume no land out of bounds
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
                    flow_amount -= flow_amount*erosion_rate
                elif (neighbor_water_height+water[cx, cy]) > (land[cx, cy] - neighbor_land_height):
                    flow_amount = ratio*(water[cx, cy] - ((water[cx, cy] + neighbor_water_height) - (land[cx, cy] - neighbor_land_height))/2)
                    flow_amount -= flow_amount * erosion_rate
            else: # the water is higher here but the land is not...
                flow_amount = min(water[cx, cy], ratio*((surface_height - neighbor_surface_height)/2))

            # Update water levels at current site
            water[cx, cy] -= flow_amount
            flow_sum[cx, cy] += flow_amount

            if neighbor_in_bounds:
                # Neighbor is within bounds, update water there
                water[nx, ny] += flow_amount

            # Erode land at the source site if the land at the destination is lower
            land[cx, cy] = land[cx, cy] - (erosion_rate * flow_amount)


def step(land, water, flow_per_rain):
    # Perform precipitation
    add_precipitation(water)
    xs = list(range(lattice_size[0]))
    ys = list(range(lattice_size[1]))
    np.random.shuffle(xs)
    np.random.shuffle(ys)
    # Flow water starting from each precipitation site
    for _ in range(flow_per_rain):
        for x in xs:
            for y in ys:
                flow(x, y, land, water)

    return land, water

# Run the simulation for a number of iterations
for i in range(iterations):
    if i % int(iterations / 100) == 0:
        print(f"Iteration {i}/{iterations}")

    land, water = step(land, water, flow_per_rain)

    if i % int(iterations/10) == 0:
        plt.figure(figsize=(10, 6))
        plt.imshow(land)
        plt.colorbar()
        plt.title("land after " + str(i) + " of " + str(iterations))
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.imshow(flow_sum)
        plt.colorbar()
        plt.title("flow_sum after " + str(i) + " of " + str(iterations))
        plt.show()

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



