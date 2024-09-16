# This is my (Cole Thumann's) implementations for Homeworks
import numpy as np
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

class AbelianSandpile:
    """
    An Abelian sandpile model simulation. The sandpile is initialized with a random
    number of grains at each lattice site. Then, a single grain is dropped at a random
    location. The sandpile is then allowed to evolve until it is stable. This process
    is repeated n_step times.

    A single step of the simulation consists of two stages: a random sand grain is
    dropped onto the lattice at a random location. Then, a set of avalanches occurs
    causing sandgrains to get redistributed to their neighboring locations.

    Parameters:
    n (int): The size of the grid
    grid (np.ndarray): The grid of the sandpile
    history (list): A list of the sandpile grids at each timestep
    """

    def __init__(self, n=100, random_state=None):
        self.n = n
        np.random.seed(random_state)  # Set the random seed
        self.grid = np.random.choice([0, 1, 2, 3], size=(n, n))
        self.history = [self.grid.copy()]  # Why did we need to copy the grid?

        # Need to copy here since the grid itself will be overwritten, and we do not
        # want the history to point to that soon to be overwritten grid (loosing the information
        # of the initial grid)

    def step(self):
        """
        Perform a single step of the sandpile model. Step corresponds a single sandgrain
        addition and the consequent toppling it causes.

        Returns: None
        """
        drop_site = np.random.randint(0, self.n, 2)
        self.grid[drop_site[0], drop_site[1]] += 1
        self.history.append(self.grid.copy())
        if self.grid[drop_site[0], drop_site[1]] >= 4:
            self.recursive_topple(drop_site[0], drop_site[1])

    def recursive_topple(self, i, j):
        """
        Set the grid site to 0, then add 1 to all neighbors.
        if any neighbors have met threshold, call topple recursively
        """
        self.grid[i, j] = 0
        if i + 1 < self.n:
            self.grid[i + 1, j] += 1
            if self.grid[i + 1, j] >= 4:
                self.recursive_topple(i + 1, j)
        if i - 1 >= 0:
            self.grid[i - 1, j] += 1
            if self.grid[i - 1, j] >= 4:
                self.recursive_topple(i - 1, j)
        if j + 1 < self.n:
            self.grid[i, j + 1] += 1
            if self.grid[i, j + 1] >= 4:
                self.recursive_topple(i, j + 1)
        if j - 1 >= 0:
            self.grid[i, j - 1] += 1
            if self.grid[i, j - 1] >= 4:
                self.recursive_topple(i, j - 1)

    # we use this decorator for class methods that don't require any of the attributes
    # stored in self. Notice how we don't pass self to the method
    @staticmethod
    def check_difference(grid1, grid2):
        """Check the total number of different sites between two grids"""
        return np.sum(grid1 != grid2)

    def simulate(self, n_step):
        """
        Simulate the sandpile model for n_step steps.
        """
        for i in range(n_step):
            self.step()

## Import William's solution from answer key
# from solutions.sandpile import AbelianSandpileIterative as AbelianSandpile
# from solutions.sandpile import AbelianSandpileBFS as AbelianSandpile
# from solutions.sandpile import AbelianSandpileDFS as AbelianSandpile


# Run sandpile simulation
model = AbelianSandpile(n=100, random_state=0)

plt.figure()
plt.imshow(model.grid, cmap='gray')
plt.title("Initial State")

model.simulate(10000)
plt.figure()
plt.imshow(model.grid, cmap='gray')
plt.title("Final state")




# Compute the pairwise difference between all observed snapshots. This command uses list
# comprehension, a zip generator, and argument unpacking in order to perform this task
# concisely.
all_events =  [model.check_difference(*states) for states in zip(model.history[:-1], model.history[1:])]
# remove transients before the self-organized critical state is reached
all_events = all_events[1000:]
# index each timestep by timepoint
all_events = list(enumerate(all_events))
# remove cases where an avalanche did not occur
all_avalanches = [x for x in all_events if x[1] > 1]
all_avalanche_times = [item[0] for item in all_avalanches]
all_avalanche_sizes = [item[1] for item in all_avalanches]
all_avalanche_durations = [event1 - event0 for event0, event1 in zip(all_avalanche_times[:-1], all_avalanche_times[1:])]


## Waiting time distribution
waiting_times = np.diff(np.array(all_avalanche_times))
plt.figure()
plt.semilogy()
plt.hist(waiting_times)
plt.title('Waiting Time distribution')
plt.xlabel('Waiting time')
plt.ylabel('Number of events')

## Duration distribution
log_bins = np.logspace(np.log10(2), np.log10(np.max(all_avalanche_durations)), 50) # logarithmic bins for histogram
vals, bins = np.histogram(all_avalanche_durations, bins=log_bins)
plt.figure()
plt.loglog(bins[:-1], vals, '.', markersize=10)
plt.title('Avalanche duration distribution')
plt.xlabel('Avalanche duration')
plt.ylabel('Count')

## Visualize activity of the avalanches
# Make an array storing all pairwise differences between the lattice at successive
# timepoints
all_diffs = np.abs(np.diff(np.array(model.history), axis=0))
all_diffs[all_diffs > 0] = 1
all_diffs = all_diffs[np.sum(all_diffs, axis=(1, 2)) > 1] # Filter to only keep big events
most_recent_events = np.sum(all_diffs[-100:], axis=0)
plt.figure(figsize=(5, 5))
plt.imshow(most_recent_events)
plt.title("Avalanche activity in most recent timesteps")