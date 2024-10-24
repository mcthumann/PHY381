import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os  # Import the os module
import numpy as np  # Import the numpy module
import urllib.request  # Import requests module (downloads remote files)
import matplotlib
from matplotlib.widgets import Slider

from sklearn.base import BaseEstimator, TransformerMixin

matplotlib.use('TkAgg')

def explore_von_karman():
    Re = 300  # Reynolds number, change this to 300, 600, 900, 1200

    fpath = f"../resources/von_karman_street/vortex_street_velocities_Re_{Re}.npz"
    if not os.path.exists(fpath):
        print("Data file not found, downloading now.")
        print("If this fails, please download manually through your browser")

        ## Make a directory for the data file and then download to it
        os.makedirs("../resources/von_karman_street/", exist_ok=True)
        url = f'https://github.com/williamgilpin/cphy/raw/main/resources/von_karman_street/vortex_street_velocities_Re_{Re}.npz'
        urllib.request.urlretrieve(url, fpath)
    else:
        print("Found existing data file, skipping download.")

    vfield = np.load(fpath, allow_pickle=True)  # Remove allow_pickle=True, as it's not a pickle file
    print("Velocity field data has shape: {}".format(vfield.shape))

    # Calculate the vorticity, which is the curl of the velocity field
    vort_field = np.diff(vfield, axis=1)[..., :-1, 1] + np.diff(vfield, axis=2)[:, :-1, :, 0]

    plt.rcParams['animation.embed_limit'] = 2**26

    # Assuming frames is a numpy array with shape (num_frames, height, width)
    frames = vort_field.copy()[::4]
    vmax = np.percentile(np.abs(frames), 98)

    fig, ax = plt.subplots(figsize=(6, 6))
    img = plt.imshow(frames[0], vmin=-vmax, vmax=vmax, cmap="RdBu")
    plt.xticks([]); plt.yticks([])
    # tight margins
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    def update(frame):
            img.set_array(frame)

    plt.subplots_adjust(left=0.1, bottom=0.25)
    img = ax.imshow(frames[0], cmap="RdBu")
    plt.xticks([]); plt.yticks([])

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        frame_idx = int(slider.val)
        img.set_array(frames[frame_idx])
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)
    plt.show()


# We are going to use class inheritance to define our object. The two base classes from
# scikit-learn represent placeholder objects for working with datasets. They include
# many generic methods, like fetching parameters, getting the data shape, etc.
#
# By inheriting from these classes, we ensure that our object will have access to these
# functions, even though we didn't define them ourselves. Earlier in the course
# we saw examples where we defined our own template classes. Here, we are using the
# template classes define by the scikit-learn Python library.
class PrincipalComponents(BaseEstimator, TransformerMixin):
    """
    A class for performing principal component analysis on a dataset.

    Parameters
        random_state (int): random seed for reproducibility
        components_ (numpy array): the principal components
        singular_values_ (numpy array): the singular values
    """

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):
        """
        Fit the PCA model to the data X. Store the eigenvectors in the attribute
        self.components_ and the eigenvalues in the attribute self.singular_values_

        NOTE: This method needs to return self in order to work properly with the
         scikit-learn base classes from which it inherits.

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) containing the
                data to be fit.

        Returns:
            self (PrincipalComponents): The fitted object.
        """

        # Center the data
        mean_vec = np.mean(X, axis=0)
        # X -= mean_vec # Subtracting a col vec from a matrix fills in the cols with the vec

        # Compute the covariance matrix
        cov = np.cov(X, rowvar=False)
        cov = cov.astype('float64')

        # Compute the eigenvalues
        eigenvals, eigenvecs = np.linalg.eig(cov)
        eigenvecs = eigenvecs.T

        # Sort
        indicies = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[indicies]
        eigenvecs = eigenvecs[indicies]

        self.components_ = eigenvecs
        self.singular_values_ = eigenvals

        return self


    def transform(self, X):
        """
        Transform the data X into the new basis using the PCA components.

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) containing the
                data to be transformed.

        Returns:
            X_new (np.ndarray): A 2D array of shape (n_samples, n_components) containing
                the transformed data. n_components <= n_features, depending on whether
                we truncated the eigensystem.
        """
       
        centered = X - np.mean(X, axis=0)
        return centered.dot(self.components_.T)

    def inverse_transform(self, X):
        """
        Transform from principal components space back to the original space

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_components) containing the
                data to be transformed. n_components <= n_features, depending on whether
                we truncated the eigensystem.

        Returns:
            X_new (np.ndarray): A 2D array of shape (n_samples, n_features) containing
                the transformed data.
        """
        return X.dot(self.components_) + np.mean(X, axis=0)

    ## You shouldn't need to implement this, because it gets inherited from the base
    ## class. But if you are having trouble with the inheritance, you can uncomment
    ## this and to implement it.
    # def fit_transform(self, X):
    #     self.fit(X)
    #     return self.transform(X)


Re = 300  # Reynolds number, change this to 300, 600, 900, 1200

fpath = f"../resources/von_karman_street/vortex_street_velocities_Re_{Re}.npz"
if not os.path.exists(fpath):
    print("Data file not found, downloading now.")
    print("If this fails, please download manually through your browser")

    ## Make a directory for the data file and then download to it
    os.makedirs("../resources/von_karman_street/", exist_ok=True)
    url = f'https://github.com/williamgilpin/cphy/raw/main/resources/von_karman_street/vortex_street_velocities_Re_{Re}.npz'
    urllib.request.urlretrieve(url, fpath)
else:
    print("Found existing data file, skipping download.")

vfield = np.load(fpath, allow_pickle=True)  # Remove allow_pickle=True, as it's not a pickle file
print("Velocity field data has shape: {}".format(vfield.shape))

# Calculate the vorticity, which is the curl of the velocity field
vort_field = np.diff(vfield, axis=1)[..., :-1, 1] + np.diff(vfield, axis=2)[:, :-1, :, 0]

data = np.copy(vort_field)[::3, ::2, ::2] # subsample data to reduce compute load
data_reshaped = np.reshape(data, (data.shape[0], -1))

model = PrincipalComponents()
# model = PCA()

data_transformed = model.fit_transform(data_reshaped)
principal_components = np.reshape(
    model.components_, (model.components_.shape[0], data.shape[1], data.shape[2])
)

## Look at skree plot, and identify the "elbow" indicating low dimensionality
plt.figure()
plt.plot(model.singular_values_[:50])
plt.plot(model.singular_values_[:50], '.')
plt.xlabel("Eigenvalue magnitude")
plt.ylabel("Eigenvalue rank")
plt.show()

## Plot the principal components
plt.figure(figsize=(20, 10))
for i in range(8):
    plt.subplot(1, 8, i+1)
    vscale = np.percentile(np.abs(principal_components[i]), 99)
    plt.imshow(np.real(principal_components[i]), cmap="RdBu", vmin=-vscale, vmax=vscale)
    plt.title("PC {}".format(i+1))

## Plot the movie projected onto the principal components
plt.figure(figsize=(20, 10))
for i in range(8):
    plt.subplot(8, 1, i+1)
    plt.plot(data_transformed[:, i])
    plt.ylabel("PC {} Amp".format(i+1))
plt.xlabel("Time")

## Plot the time series against each other
plt.figure()
ax = plt.axes(projection='3d')
ax.plot(data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2])

plt.show()

# 1. Download the von Karman dataset and explore it using the code below. What symmetries are present in the data? Do any
# symmetries change as we increase the Reynolds number? Note: If you are working from a fork of the course repository,
# the dataset should load automatically. Otherwise, the cells below should automatically download the data sets and
# place them in higher-level directory ../resources. If the automatic download fails, you may need to manually download
# the von Karman dataset and place it in the correct directory.

    # The data has a lot of symmetry at a Reynolds number of 300. There is right-left symmetry, and a repetition along
    #  timesteps where vorticity packets of alternating signs are forming a set intervals. As the Reynolds number
    #  increases the symmetry goes away and the process seems to be more chaotic.

# 2. Implement Principal Component Analysis in Python. I have included my outline code below; we are going to use multiple
# inheritence in order to make our implementation compatible with standard conventions for machine learning in Python.
# I recommend using numpy's built-in eigensystem solvers np.linalg.eig and np.linalg.eigh

    # Done
#
# 3. Plot the eigenvalues of the data covariance matrix in descending order. What does this tell us about the effective
# dimensionality, and thus optimal number of features, to use to represent the von Karman dataset?
#
    # We see from the plot that the leading eigenvalue is around .0128, the next is .0103, followed about an order of
    # magnitude less .0017 and .0016, and the next three are another 60ish percent drop off (.0005 and .00095) making
    # them less than 4% of the leading eigenvalue. So there's a large drop off after the first two components, a large
    # drop off after the next two, and then only a few more before reaching eigenvalues that are less than 1% of the
    # leading eig. The optimal number of features to project onto might be between 6 and 10. The fact that these first
    # 6 eigenvalues loosely come in pairs seems to signify the duality of a red vs blue vorticity cluster occupying a
    # region of the map.

# 4. Try re-running your analysis using datasets from different Reynolds numbers. How does the effective dimensionality
# of the problem change as Reynolds number increases? How is that reflected in the lower-dimensional time series
# representation?

    # As the reynolds number increases the importance of the later eigenvalues/vectors decays more and more slowly.
    # You need more dimensionality to capture the dynamics. There is less symmetry. In the lower dimensional time
    # series there is less accuracy.
#
# 5. For this problem, the principal components often appear in pairs. Can you think of a reason for this?

    # I had noticed that answering the first question, and I think it is because of the curl symmetry of the vorticies.
    # The red and blue clusters can behave the same way but are essentially identified by a different sign. So there
    # needs to be similar looking components that hint at the fact that a certain region is likely to be red -occupied
    # and a matching component that hints that the same region is likely to be blue occupied.
#
# 6. What happens if we don't subtract the feature-wise mean before calculating PCA? Why does this happen?
#
#   When the mean is not subtracted, and the mean is far from zero, then the mean explains a lof of the variance in the
#   data, and the leading eigenvalue/vector may be biased.

# 7. In Fourier analysis, we project a function onto linear combination of trigonometric basis functions. How is this
# related to principal components analysis?

    # PCA is also a projection on to an orthogonal basis. Each component sine frequency in a fourier transform
    # contributes some magnitude to the overall signal, just as the eigenbasis contributes to the representation of the
    # overall dataset