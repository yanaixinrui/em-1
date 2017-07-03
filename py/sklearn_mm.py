import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from sklearn import mixture

print(__doc__)

plt.figure(figsize=(5,5))
# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
np.random.seed(2)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

features=[' AU06_r',' AU12_r']
df = pd.read_csv('example/interrogator.csv') 
X = df.loc[:,features].dropna().values


lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
#n_components_range = range(5, 6)
cv_types = ['spherical', 'tied', 'diag', 'full']
cv_types = ['full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type,
                                      tol=1e-8,
                                      max_iter=1000,
                                      n_init=3,
                                      reg_covar=2e-3)
        gmm.fit(X)
        print('n-iter:',gmm.n_iter_)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
#spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
#spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)
plt.xlabel('Number of components')
plt.legend([b[0] for b in bars], cv_types)

# Plot the winner
plt.figure(figsize=(8,8))
splot = plt.subplot(1, 1, 1)
Y_ = clf.predict(X)

for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
    #cov = cov * np.eye(2)
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color,alpha=.5)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.75)
    splot.add_artist(ell)
    
print(clf.means_)
plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()
