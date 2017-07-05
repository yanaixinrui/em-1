from __future__ import print_function
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import mixture

print(__doc__)

"""


"""

def plot_gmm(clf, X, features, outfile, bic, to_screen=False):
    """ saves a plot of the clusters with data """
    
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])    
    plt.figure(figsize=(8,8))   
    
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color,alpha=.5)
    
        x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 50)
        y_vals = np.linspace(X[:,1].min(), X[:,1].max(), 50)
        x, y = np.meshgrid(x_vals, y_vals)    
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv = multivariate_normal(mean, cov)
    
        try:
            plt.contour(x, y, rv.pdf(pos))
        except ValueError:
            pass    

    k = clf.means_.shape[0]
    plt.title('all_frames I&W, k = ' + str(k) + '  , BIC = ' + str(bic))
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.savefig('bestClusters_' + str(k) + '.png')
    plt.close()
    
#----------------------------------     

plt.figure(figsize=(5,5))

features=[' AU06_r',' AU12_r']
outfile = 'clusters.csv'
#df = pd.read_csv('example/interrogator.csv') 
df = pd.read_csv('example/all_frames.csv') 
X = df.loc[:,features].dropna().values


lowest_bic = np.infty
bic = []
n_components_range = range(1, 12)
#n_components_range = range(5, 6)
cv_types = ['spherical', 'tied', 'diag', 'full']
cv_types = ['full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type,
                                      tol=1e-8,
                                      #tol=1e-6,
                                      max_iter=1000,
                                      #max_iter=100,
                                      n_init=3,
                                      reg_covar=2e-3)
        gmm.fit(X)
        print('n_components:', n_components, ', n-iter:', gmm.n_iter_)
        bic.append(gmm.bic(X))
        plot_gmm(gmm, X, features, 'clusterContour', bic[-1], False)
        
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
plt.savefig('BIC scores')

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

    '''
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.75)
    splot.add_artist(ell)
    '''
    x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 50)
    y_vals = np.linspace(X[:,1].min(), X[:,1].max(), 50)
    x, y = np.meshgrid(x_vals, y_vals)    
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal(mean, cov)

    try: # not sure why, was running into ValueErrors
        plt.contour(x, y, rv.pdf(pos))
    except ValueError:
        pass
    
print(clf.means_)
sigmas = np.empty(clf.covariances_.shape[0],dtype=object)
for i in range(sigmas.shape[0]):
    sigmas[i] = clf.covariances_[i]
cluster_data = np.concatenate((clf.means_,sigmas[:,np.newaxis]),axis=1)
df_clusters = pd.DataFrame(data=cluster_data,columns=features+['sigmas'])
df_clusters.to_csv(outfile,index=False)
#plt.xticks(())
#plt.yticks(())
plt.title('best GMM')
plt.xlabel(features[0])
plt.ylabel(features[1])
#plt.subplots_adjust(hspace=.35, bottom=.02)
plt.savefig('clusterContours.png')
#plt.show()
