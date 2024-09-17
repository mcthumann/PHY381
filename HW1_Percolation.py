import numpy as np
import matplotlib.pyplot as plt

import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(1500)

class PercolationSimulation:
    """
    A simulation of a 2D directed percolation problem. Given a 2D lattice, blocked sites
    are denoted by 0s, and open sites are denoted by 1s. During a simulation, water is
    poured into the top of the grid, and allowed to percolate to the bottom. If water
    fills a lattice site, it is marked with a 2 in the grid. Water only reaches a site
    if it reaches an open site directly above, or to the immediate left or right
    of an open site.

    I've included the API for my solution below. You can use this as a starting point,
    or you can re-factor the code to your own style. Your final solution must have a
    method called percolate that creates a random lattice and runs a percolation
    simulation and
    1. returns True if the system percolates
    2. stores the original lattice in self.grid
    3. stores the water filled lattice in self.grid_filled

    + For simplicity, use the first dimension of the array as the percolation direction
    + For boundary conditions, assume that any site out of bounds is a 0 (blocked)
    + You should use numpy for this problem, although it is possible to use lists

    Attributes:
        grid (np.array): the original lattice of blocked (0) and open (1) sites
        grid_filled (np.array): the lattice after water has been poured in
        n (int): number of rows and columns in the lattice
        p (float): probability of a site being blocked in the randomly-sampled lattice
            random_state (int): random seed for the random number generator
        random_state (int): random seed for numpy's random number generator. Used to
            ensure reproducibility across random simulations. The default value of None
            will use the current state of the random number generator without resetting
            it.
    """

    def __init__(self, n=100, p=0.5, grid=None, random_state=None):
        """
        Initialize a PercolationSimulation object.

        Args:
            n (int): number of rows and columns in the lattice
            p (float): probability of a site being blocked in the randomly-sampled lattice
            random_state (int): random seed for numpy's random number generator. Used to
                ensure reproducibility across random simulations. The default value of None
                will use the current state of the random number generator without resetting
                it.
        """

        self.random_state = random_state  # the random seed

        # Initialize a random grid if one is not provided. Otherwise, use the provided
        # grid.
        if grid is None:
            self.n = n
            self.p = p
            self.grid = np.zeros((n, n))
            self._initialize_grid()
        else:
            assert len(np.unique(np.ravel(grid))) <= 2, "Grid must only contain 0s and 1s"
            self.grid = grid.astype(int)
            # override numbers if grid is provided
            self.n = grid.shape[0]
            self.p = 1 - np.mean(grid)

        # The filled grid used in the percolation calculation. Initialize to the original
        # grid. We technically don't need to copy the original grid if we want to save
        # memory, but it makes the code easier to debug if this is a separate variable
        # from self.grid.
        self.grid_filled = np.copy(self.grid)

    def _initialize_grid(self):
        """
        Sample a random lattice for the percolation simulation. This method should
        write new values to the self.grid and self.grid_filled attributes. Make sure
        to set the random seed inside this method.

        This is a helper function for the percolation algorithm, and so we denote it
        with an underscore in order to indicate that it is not a public method (it is
        used internally by the class, but end users should not call it). In other
        languages like Java, private methods are not accessible outside the class, but
        in Python, they are accessible but external usage is discouraged by convention.

        Private methods are useful for functions that are necessary to support the
        public methods (here, our percolate() method), but which we expect we might need
        to alter in the future. If we released our code as a library, others might
        build software that accesses percolate(), and so we should not alter the
        input/outputs because it's a public method
        """
        # Set seed
        np.random.seed(self.random_state)
        # initialize grid 0's and 1's with prob p
        self.grid[...] = np.random.choice([0, 1], size=(self.n, self.n), p=[self.p, 1 - self.p])
        self.grid_filled = np.copy(self.grid)

    def _flow_recursive(self, i, j):
        """
        Only used if we opt for a recursive solution.

        The recursive portion of the flow simulation. Notice how grid and grid_filled
        are used to keep track of the global state, even as our recursive calls nest
        deeper and deeper
        """
        # Dont check edges
        if i < 0 or i >= self.n:
            return None
        if j < 0 or j >= self.n:
            return None
        # Blocked
        if self.grid[i, j] == 0:
            return None
        # Water
        if self.grid_filled[i, j] == 2:
            return None

        # must be open, fill it
        self.grid_filled[i, j] = 2

        # flow to neighbors
        self._flow_recursive(i, j + 1)
        self._flow_recursive(i, j - 1)
        self._flow_recursive(i + 1, j)
        self._flow_recursive(i - 1, j)

    def _poll_neighbors(self, i, j):
        """
        Check whether there is a filled site adjacent to a site at coordinates i, j in
        self.grid_filled. Respects boundary conditions.
        """

        ####### YOUR CODE HERE  #######################################################
        raise NotImplementedError("Implement this method")
        #
        #
        # Hint: my solution is 4 lines of code in numpy, but you may get different
        # results depending on how you enforce the boundary conditions in your solution.
        # Not needed for the recursive solution
        #
        #
        ###############################################################################


    def _flow(self):
        """
        Run a percolation simulation using recursion

        This method writes to the grid and grid_filled attributes, but it does not
        return anything. In other languages like Java or C, this method would return
        void
        """
        ###############################################################################
        raise NotImplementedError("Implement this method")
        ####### YOUR CODE HERE  #######
        # Hintsmy non-recursive solution contains one row-wise for loop, which contains
        # several loops over individual lattice sites. You might need to visit each lattice
        # site more than once per row. In my implementation, split the logic of checking
        # the von neumann neighborhood into a separate method _poll_neighbors, which
        # returns a boolean indicating whether a neighbor is filled
        #
        # My recursive solution calls a second function, _flow_recursive, which takes
        # two lattice indices as arguments

        ###############################################################################

    def percolate(self):
        """
        Initialize a random lattice and then run a percolation simulation. Report results
        """
        # Since water can enter any cells in the top row, try all of them
        for i in range(self.n):
            self._flow_recursive(i, 0)

        # If any of the grids bottom row entries are filled, then return true, otherwise false
        return np.any(self.grid_filled == 2, axis=0)[-1]

# Import William's solution
# from solutions.percolation import PercolationSimulation
# from solutions.percolation_iterative import PercolationSimulation

from matplotlib.colors import LinearSegmentedColormap
def plot_percolation(mat):
    """
    Plots a percolation matrix, where 0 indicates a blocked site, 1 indicates an empty
    site, and 2 indicates a filled site
    """
    cvals  = [0, 1, 2]
    colors = [(0, 0, 0), (0.4, 0.4, 0.4), (0.372549, 0.596078, 1)]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = LinearSegmentedColormap.from_list("", tuples)
    plt.imshow(mat, cmap=cmap, vmin=0, vmax=2)


model = PercolationSimulation(n=20, random_state=0, p=0.9)
print(model.percolate())
plt.figure()
plot_percolation(model.grid_filled)
plt.show()

model2 = PercolationSimulation(n=20, random_state=0, p=0.407259)
print(str(model2.percolate()))
plt.figure()
plot_percolation(model2.grid_filled)
plt.show()

model = PercolationSimulation(n=20, random_state=0, p=0.41)
print(model.percolate())
plt.figure()
plot_percolation(model.grid_filled)
plt.show()


pvals = np.linspace(0, 1, 25) # control parameter for percolation phase transition
n_reps = 20 # number of times to repeat the simulation for each p value

all_percolations = list()
for p in pvals:
    print("Running replicate simulations for p = {}".format(p), flush=True)
    all_replicates = list()
    for i in range(n_reps):
        # Initialize the model
        model = PercolationSimulation(30, p=p)
        all_replicates.append(model.percolate())
    all_percolations.append(all_replicates)

plt.figure()
plt.plot(pvals, np.mean(np.array(all_percolations), axis=1))
plt.xlabel('Average site occupation probability')
plt.ylabel('Percolation probability')

plt.figure()
plt.plot(pvals, np.std(np.array(all_percolations), axis=1))
plt.xlabel('Standard deviation in site occupation probability')
plt.ylabel('Percolation probability')

plt.show()

## Just from curiousity, plot the distribution of cluster sizes at the percolation threshold
## why does it appear to be bimodal?
all_cluster_sizes = list()
p_c = 0.407259
n_reps = 5000
for i in range(n_reps):
    model = PercolationSimulation(100, p=p_c)
    model.percolate()
    cluster_size = np.sum(model.grid_filled == 2)
    all_cluster_sizes.append(cluster_size)

    if i % 500 == 0:
        print("Finished simulation {}".format(i), flush=True)

all_cluster_sizes = np.array(all_cluster_sizes)

plt.figure()
plt.hist(all_cluster_sizes, 50);
plt.show()