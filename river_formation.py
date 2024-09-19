import numpy as np
import matplotlib.pyplot as plt

class RiverFormation:
    """River formation model based off Kramer, Marder's paper "Evolution of river networks" 1993
    Land, and Water are maps of the same size that give, at each point, the height of soil, and the amount of water
    sitting on top of the soil respectively. The program loosely follows the following ruleset quoted from the above
    paper:
            (I) At each site of a lattice, we specify two integers, one corresponding to the height of land, the other to the height ofwater.
            (2) A lattice site is chosen randomly, and if the surface height (water plus land) is lower on a neighboring site, water units are moved to bring the surfaces as close to even as possible.
            (3) For each water unit transported out, a unit of land is dissolved away —but only if the land is lower at the destination site.
            (4) Additional water falls on a site as precipitation at random intervals.

    Parameters
    map_size (int, int): Sets the size of the corresponding land and water maps
    erosion_rate (float): Controls the rate of erosion
    land_initial_height (int): Initial height of the land plot
    precipitation (float): At each random precipitation site, this much watter is added
    land_condition (int): More of an enum to dictate the initial conditions of the land:
        0: Flat initial condition
        1: Random initial condition with randomness scaled by precipitation
        2: Sloping downward initial condition with slope set by precipitation
    """
    def __init__(self, map_size=(50, 25), erosion_rate=.1, land_initial_height=10, precipitation=0.1, land_condition=0):
        self.map_size = map_size
        self.erosion_rate = erosion_rate
        self.land_initial_height = land_initial_height
        self.precipitation = precipitation
        self.land_condition = land_condition
        # Chance of a map site being randomly chosen for
        self.precipitation_rate = 1 / map_size[0]

        # How many times to cycle through map and flow another rainfall
        #self.flow_per_rain = int(map_size[0] / 2)
        self.flow_per_rain = 10 # To speed up ...

        # Flat land
        self.land = np.full(map_size, self.land_initial_height, dtype=float)
        self.water = np.zeros(map_size, dtype=float)
        self.flow_sum = np.zeros(map_size, dtype=float)

        # Randomize land
        if self.land_condition == 1:
            self.land += np.random.uniform(-self.precipitation, self.precipitation, size=self.map_size)

        # Sloped land
        if self.land_condition == 2:
            column_deltas = self.precipitation * (np.arange(self.map_size[1]) + 1)
            self.land += column_deltas[np.newaxis, :]

    # Add some rain to sites randomly ...
    def add_precipitation(self):
        precipitation_sites = np.random.random(self.map_size) < self.precipitation_rate
        self.water[precipitation_sites] += self.precipitation
        return np.argwhere(precipitation_sites)

    # Flow the water around to the neighbors and erode away. Find valid neighbors, sort them by who to flow to
    # first, then flow alloted ratio to each neighbor if acceptable.
    def flow(self, cx, cy):

        # We dont care about waterless sites
        if self.water[cx, cy] <= 0:
            return

        surface_height = self.land[cx, cy] + self.water[cx, cy]

        cy += 1 # prepare for padding
        # pad with block / flow regions
        left_pad_land = self.land[:, [0]] - self.precipitation
        right_pad_land = self.land[:, [-1]] + self.precipitation
        self.land = np.hstack((left_pad_land, self.land, right_pad_land))

        left_pad_water = np.zeros((self.map_size[0], 1))
        right_pad_water = np.zeros((self.map_size[0], 1))
        self.water = np.hstack((left_pad_water, self.water, right_pad_water))

        # Calculate neighbor positions. Use wrap around for x.
        # For y we will have one open flow and one closed
        neighbors = np.array([
            [(cx - 1) % self.map_size[0], cy],  # Left
            [(cx + 1) % self.map_size[0], cy],  # Right
            [cx, cy + 1],  # Up
            [cx, cy - 1],  # Down
            [(cx - 1) % self.map_size[0], cy - 1],  # Left Down
            [(cx + 1) % self.map_size[0], cy - 1],  # Right Down
            [(cx - 1) % self.map_size[0], cy + 1],  # Left Up
            [(cx + 1) % self.map_size[0], cy + 1]  # Right Up
        ])

        # Try not to favor some direction ...
        np.random.shuffle(neighbors)

        # Separate neighbor x and y coordinates
        nx = neighbors[:, 0]
        ny = neighbors[:, 1]

        # For valid indices, compute surface heights from land and water arrays
        neighbor_surface_difference =((self.land[nx, ny] + self.water[nx, ny]) - surface_height)*(-1)

        neighbor_surface_difference[neighbor_surface_difference < 0] = 0
        heigh_diff_sum = np.sum(neighbor_surface_difference)
        neighbor_ratios = None
        if heigh_diff_sum > 0:
            neighbor_ratios = neighbor_surface_difference/heigh_diff_sum
        elif heigh_diff_sum ==0:
            neighbor_ratios = [.125, .125, .125, .125, .125, .125, .125, .125]
        else:
            print("Ratio error")

        for i in range(len(neighbors[0])):
            # Check if neighbor is within bounds
            nx = neighbors[i][0]
            ny = neighbors[i][1]
            ratio = neighbor_ratios[i]
            neighbor_in_bounds = 1 <= ny <= self.map_size[1]
            neighbor_land_height = self.land[nx, ny]
            neighbor_surface_height = self.land[nx, ny] + self.water[nx, ny]
            neighbor_water_height = self.water[nx, ny]

            # Proceed if water can flow to the neighbor
            if neighbor_surface_height < surface_height and self.water[cx, cy] > 0:
                # Calculate how much water can flow to the neighbor
                # if the neighbor is so low that the sum of the water wont fill the gap of the land, send all the water
                flow_amount = 0
                if self.land[cx, cy] > neighbor_land_height:
                    if (neighbor_water_height+self.water[cx, cy]) < (self.land[cx, cy] - neighbor_land_height):
                        flow_amount = ratio*self.water[cx, cy]
                        flow_amount -= flow_amount * self.erosion_rate / 2
                    elif (neighbor_water_height+self.water[cx, cy]) > (self.land[cx, cy] - neighbor_land_height):
                        flow_amount = ratio*(self.water[cx, cy] - ((self.water[cx, cy] + neighbor_water_height) -
                                                                   (self.land[cx, cy] - neighbor_land_height))/2)
                        flow_amount -= flow_amount * self.erosion_rate / 2

                else: # the water is higher here but the land is not...
                    flow_amount = min(self.water[cx, cy], ratio*((surface_height - neighbor_surface_height)/2))
                    flow_amount -= flow_amount * self.erosion_rate / 2

                # Update water levels at current site
                self.water[cx, cy] -= flow_amount
                self.flow_sum[cx, cy-1] += flow_amount

                if neighbor_in_bounds:
                    # Neighbor is within bounds, update water there
                    self.water[nx, ny] += flow_amount

                # Erode land at the source site if the land at the destination is lower
                self.land[cx, cy] -= (self.erosion_rate * flow_amount)
        self.water = self.water[:, 1:-1]
        self.land = self.land[:, 1:-1]


    def step(self):
        # Perform precipitation
        self.add_precipitation()
        xs = list(range(self.map_size[0]))
        ys = list(range(self.map_size[1]))
        np.random.shuffle(xs)
        np.random.shuffle(ys)
        # Flow water starting from each precipitation site
        for _ in range(self.flow_per_rain):
            for x in xs:
                for y in ys:
                    self.flow(x, y)

    def run(self, iterations):
        # Run the simulation for a number of iterations
        for i in range(iterations):
            if i % int(iterations / 100) == 0:
                print(f"Iteration {i}/{iterations}")

            self.step()

            if i % int(iterations/10) == 0:
                plt.figure(figsize=(10, 6))
                plt.imshow(self.land)
                plt.colorbar()
                plt.title("land after " + str(i) + " of " + str(iterations))
                plt.show()
                plt.figure(figsize=(10, 6))
                plt.imshow(self.flow_sum)
                plt.colorbar()
                plt.title("flow_sum after " + str(i) + " of " + str(iterations))
                plt.show()

basin = RiverFormation((50, 20), 0.2, 0, .1, 0)
basin.run(iterations=1000)

# Plot the final state of the system
plt.figure(figsize=(10, 6))
plt.imshow(basin.flow_sum)
plt.colorbar(label='Height (land)')
plt.title('Flow sum After Simulation')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(basin.land)
plt.colorbar(label='Height (land)')
plt.title('land After Simulation')
plt.show()



