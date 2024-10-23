import warnings
import numpy as np
import matplotlib.pyplot as plt

class SpectralDecompositionPowerMethod:
    """
    Store the output vector in the object attribute self.components_ and the
    associated eigenvalue in the object attribute self.singular_values_

    Parameters
        max_iter (int): maximum number of iterations to for the calculation
        tolerance (float): fractional change in solution to stop iteration early
        gamma (float): momentum parameter for the power method
        random_state (int): random seed for reproducibility
        store_intermediate_results (bool): whether to store the intermediate results as
            the power method iterates
        stored_eigenvalues (list): If store_intermediate_results is active, a list of
            eigenvalues at each iteration
        stored_eigenvectors (list): If store_intermediate_results is active, a list of
            eigenvectors at each iteration

    """

    def __init__(self,
                 max_iter=1000,
                 tolerance=1e-5,
                 gamma=0.0,
                 random_state=None,
                 store_intermediate_results=True
                 ):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.gamma = gamma
        self.random_state = random_state
        self.store_intermediate_results = store_intermediate_results
        self.stored_eigenvalues = None
        self.stored_eigenvectors = None

        if self.store_intermediate_results:
            self.stored_eigenvalues = []
            self.stored_eigenvectors = []

        # Placeholders for the results of the calculation
        self.singular_values_ = None
        self.components_ = None

    def fit(self, A):
        """
        Perform the power method with random initialization, and optionally store
        intermediate estimates of the eigenvalue and eigenvectors at each iteration.
        You can add an early stopping criterion based on the tolerance parameter.
        """
        print(np.linalg.eigvals(A))

        # Find a random unit vector of dim A_Cols
        np.random.seed(self.random_state)
        w = np.random.rand(A.shape[0])

        # Normalize the vector
        w_n = w / np.sqrt(np.dot(w, w))  # Ensure w is a numeric array

        if self.store_intermediate_results:
            self.stored_eigenvalues.append(1.0)
            self.stored_eigenvectors.append(w_n)

        # Multiply by A and renormalize
        for i in range(self.max_iter):
            w_prev = np.copy(w_n)

            eigval = np.sqrt(np.dot(A@w_n, A@w_n))
            w_n = A@w_n / eigval

            if self.store_intermediate_results:
                self.stored_eigenvalues.append(eigval)
                self.stored_eigenvectors.append(w_n)

            if np.mean(np.sqrt((w_n - w_prev) ** 2 / w_prev ** 2)) < self.tolerance:
                warnings.warn(f"Power method converged before {self.max_iter} iterations")
                break

        self.singular_values_ = eigval
        self.components_ = w_n


## import William's solutions
# from solutions.eigen import SpectralDecompositionPowerMethod

## Use the default eigensystem calculator in numpy as a point of comparison
def eigmax_numpy(A):
    """
    Compute the maximum eigenvalue and associated eigenvector in a matrix with Numpy.
    """
    eigsys = np.linalg.eig(A)
    ind = np.abs(eigsys[0]).argmax()
    return np.real(eigsys[0][ind]), np.real(eigsys[1][:, ind])


np.random.seed(2) # for reproducibility
mm = np.random.random(size=(10, 10)) / 100
mm = np.random.normal(size=(10, 10))# / 100 # these matrices fail to converge more often

# mm += mm.T # force hermitian

print("EIGENVALS OF A")
print(np.linalg.eigvals(mm)[0])
print(np.linalg.eigvals(mm)[1])
print(np.linalg.eigvals(mm)[2])
print(np.linalg.eigvals(mm)[3])
print(np.linalg.eigvals(mm)[4])
print(np.linalg.eigvals(mm)[5])
print("EIGENVECTORS OF A")

print(np.linalg.cond(mm.T))
model = SpectralDecompositionPowerMethod(store_intermediate_results=True, gamma=0.0)
model.fit(mm);


print(f"Power method solution: {model.singular_values_}")
print(f"Numpy solution: {eigmax_numpy(mm)[0]}")

plt.figure()
plt.plot(model.stored_eigenvalues)
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue estimate")
plt.xlim(right=50)
plt.show()

plt.figure()
plt.plot(eigmax_numpy(mm)[1], model.components_, '.')
plt.xlabel("Numpy eigenvector")
plt.ylabel("Power method eigenvector")
plt.show()


# 1. Implement the power method in Python. I've included my starter code below.

    # Done

# 2. Sometimes you'll notice that the power method fails to converge to the correct solution. What is special about
# randomly-sampled matrices where this occurs? How does the direction of the starting vector affect the time it takes
# to reach a solution?

    # The failure to converge to the correct solution is due to the fact that there are two leading eigenvalues which
    # are complex conjugates of each other. In this case - the power method displays oscillatory behavior. Real
    # matrices may have eigenvalues/vectors that are imaginary or complex if they are not real symmetric or real upper
    # triangular. The direction of the starting vector is important - when it is near the dominant eigenvalue there is
    # fast convergence. When it is nearly orthogonal to the dominant eigenvector there is slow convergence. When it is
    # very near another eigenvalue that is close in magnitude to the dominant eigenvalue there may be slow convergence
    # as well.

# 3. Suppose that we interpret a given linear matrix  X  as describing a discrete-time linear dynamical system,
# yt+1=Xyt. What kind of dynamics does the power method exhibit? What about the pathological cases you discussed in the
# previous solution?

    # The power method's long time dynamics would show convergence to the direction of the eigenvector corresponding
    #  to the largest eigenvalue. The matrix X iteratively stretches the initial vector y_0. The stretching happens in
    #  the direction of all of the eigenvectors, but since the stretch is applied t number of times we see the dominant
    #  eigenvalue grow in importance exponentially. eigval_1^t >> eigval_2^t. When eig_1 > 1 we see exp growth and when
    #  eig_1 < 1 we see exp decay. The pathological cases of complex eigenvalues means we will see oscillatory behavior.
    # When eig_1 is close to eig_2 the system  may tak a longer time to reach convergence.


# 4. The power method represents a basic optimization problem, where we are searching for a convergent solution. We
# saw that our method occasionally fails to find the correct solution. One way to improve our optimization would be
# to add a momentum term of the form yt←γyt−1+(1−γ)Xyt−1|Xyt−1| Where  γ∈(0,1] . How would you modify your
# implementation of the power method, in order to allow momentum? What kinds of pathological dynamics would the
# momentum term help us avoid?

    # A momentum term would help with faster convergence when your initial vector is nearly orthogonal to the largest
    #  eigenvalue's eigenvector. Momentum terms will also help escape local minima (stuck near a second largest eigen
    #  direction) faster than the naive approach. The momentum term, intuitively, may make the case of a negative
    #  leading eigenvalue (where the sign/direction flips with every iteration) or complex eigenvalue (oscillatory)
    #  worse. We might reinforce some non-convergent behavior if not careful.

# 5. Similar to the momentum term, there is also a way to add additional damping to the update rule. What kinds of
# dynamics would that help us avoid?

    # A damping term would help ensure that the leading eigenvectors direction does not get overshot by the momentum
    # term. in these oscillatory and sign-flipping cases. It may help ease oscillatory behavior.