from __future__ import print_function
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import mixture

import csv
import glob
import os
import sys
import argparse
import logging
print(__doc__)

from collections import defaultdict

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
     
#------------------------------------------------------------------------
def write_results(outfile, header, avg_dict):
    logging.info('writing output file')

    with open(outfile, 'w') as f:
        feature_data = []
        writer = csv.writer(f)
        writer.writerow(['Filename'] + list(header))
        
        for fname in avg_dict:
            row = [str(x) for x in avg_dict[fname]]
            writer.writerow([fname] + row)

#------------------------------------------------------------------------
def get_data_segments(fname, key_features, data_type='', S1=None, S2=None):
    """ returns np.arrays of S1, S2, S3 data from fname """
    
    
    #END_CHOP = 180 # number of seconds to chop from the end, 180 for deception
    END_CHOP = 120 # number of seconds to chop from the end
    logging.info('Processing ' + fname)
    
    with open(fname, 'rt') as f:
        feature_data = []
        try:
            reader = csv.reader(f)
            header = np.array(next(reader))
            header = header[key_features]
            time = float(0)
            for row in reader:
                row = [w.replace('-nan(ind)','NaN') for w in row]
                if 'nan' in row[0]:  # FIX THIS WEHN YOU ARE CALC AVE VELOCITY
                    first = row[0]
                    first = first.split('nan')
                    row[0] = first[0]
                    row.insert(1,first[1])
                if data_type == 'SHORE':
                    row = np.array([time] + row)
                else:
                    row = np.array(row)
                feature_data.append(row[key_features])
                time += 0.0333667
        finally:
            f.close()    
            
    
        feature_data = np.array(feature_data,dtype=float)  
        
        timestamp = feature_data[:,0]
        end_time = timestamp[-1] - END_CHOP  
        success = feature_data[:,2]
            
        S1_bool = np.logical_and( (timestamp > S1), (timestamp < S2) )
        S1_bool = np.logical_and(S1_bool,success) 
        S2_bool = np.logical_and( (timestamp > S2), (timestamp < end_time) )    
        S2_bool = np.logical_and(S2_bool,success) 
        S3_bool = (timestamp > end_time)
        S3_bool = np.logical_and(S3_bool,success) 
        
        S1_data=feature_data[S1_bool,3:]
        S2_data=feature_data[S2_bool,3:]
        S3_data=feature_data[S3_bool,3:]
        
        new_header = header[3:]
        
        return new_header, S1_data, S2_data, S3_data
#------------------------------------------------------------------------
def get_face_dict(inputs, file_time_dict, S1_, S2_, key_features,
                       data_type=''):  

    face_dict = {} # for every basename, gives % faces per S1, S2...
    files =  glob.glob(inputs)
    if files==[]:
        logging.error(inputs + ' - no files found')
        exit() 

    data=[]

    for fname in files:
        basename = os.path.basename(fname)
        root = '-'.join(basename.split('-')[0:6])
        if root in file_time_dict:
            S1,S2 = file_time_dict[root]
            result =  get_data_segments(fname, key_features,data_type,S1,S2)
            header, S1_datas, S2_datas, S3_datas = result

        else:
            continue                
            #logging.warning('no S1 S2 data exists for ' + fname + ',using arg.b, arg.e')
            #file_header,S1_avg,S2_avg,S3_avg,SA_avg = calc_avg(fname,data_type,S1_,S2_)
        face_dict[basename] = [list(S1_datas), list(S2_datas), list(S3_datas)] 

    return header, face_dict
#------------------------------------------------------------------------
def analyze_face_result(header, face_dict, gmm):
    logging.info('analyzing face results')
    
    
    k = gmm.n_components
    face_results = defaultdict(list)
    for fname in face_dict:
        SA_counts = np.zeros(k)
        for data in face_dict[fname]:
            data = np.array(data)
            print(fname)
            print(data.shape)
            if (data.shape[0]==0):
                continue
            data = data[~np.isnan(data).any(axis=1),:]

            counts = np.zeros(k)
            length = data.shape[0]
            y = gmm.predict(data)

            for y_i in range(k):
                counts[y_i] = (y == y_i).sum()            
            percents = counts / counts.sum()
            face_results[fname] += list(percents) 
            SA_counts += counts
        
        percents = SA_counts / SA_counts.sum()
        face_results[fname] += list(percents)    
    face_header = []
    for prefix in ['S1','S2','S3','SA']:
        for k_i in range(k):
            face_header += [prefix +'_cluster' + str(k_i)]  
    
    output_file_name = 'face_result_'+str(k)+'.gmm.csv'
    write_results(output_file_name,face_header, face_results)        
#------------------------------------------------------------------------
def analyze_face_soft_result(header, face_dict, gmm):
    logging.info('analyzing face results')
    
    
    k = gmm.n_components
    face_results = defaultdict(list)
    for fname in face_dict:
        file_data = []
        SA_counts = np.zeros(k)
        for data in face_dict[fname]:
            data = np.array(data)
            if (data.shape[0]==0):
                continue
            data = data[~np.isnan(data).any(axis=1),:]
            file_data.append(data)

            counts = np.zeros(k)
            length = data.shape[0]
            y = gmm.predict_proba(data)
          
            percents = np.nanmean(y, axis =0)
            face_results[fname] += list(percents) 
        
        if(file_data.shape == 0):
            continue
        file_data = np.vstack(file_data)
        file_predict = gmm.predict_proba(file_data)
        percents = np.nanmean(file_predict, axis=0)
        face_results[fname] += list(percents)    
        
    face_header = []
    for prefix in ['S1','S2','S3','SA']:
        for k_i in range(k):
            face_header += [prefix +'_cluster' + str(k_i)]  
    
    output_file_name = 'soft_face_result_'+str(k)+'.gmm.csv'
    write_results(output_file_name,face_header, face_results)        

#------------------------------------------------------------------------
def calculate_gmm(header, face_dict):
    plt.figure(figsize=(5,5))
    
    features=[' AU06_r',' AU12_r']
    outfile = 'clusters.csv'
    #df = pd.read_csv('example/interrogator.csv') 
    df = pd.read_csv('example/all_frames.csv') 
    X = df.loc[:,features].dropna().values
    
    #gmm_list = []
    lowest_bic = np.infty
    bic = []
    n_components_range = range(2, 3) #initially 1-12
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
            #gmm_list.append(gmm)
            print('n_components:', n_components, ', n-iter:', gmm.n_iter_)
            bic.append(gmm.bic(X))
            plot_gmm(gmm, X, features, 'clusterContour', bic[-1], False)
            
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
            
            #analyze_face_result(header, face_dict, gmm)
            analyze_face_soft_result(header, face_dict, gmm)
            print(gmm.means_)
            sigmas = np.empty(gmm.covariances_.shape[0],dtype=object)
            for i in range(sigmas.shape[0]):
                sigmas[i] = gmm.covariances_[i]
            local_cluster_data = np.concatenate((gmm.means_,sigmas[:,np.newaxis]),axis=1)
            df_clusters = pd.DataFrame(data=local_cluster_data,columns=features+['sigmas'])
            currentCluster = 'cluster_'+str(n_components)+'.csv'
            df_clusters.to_csv(currentCluster,index=False)
    
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

#write code to load in a GMM cluster result csv
#------------------------------------------------------------------------
def read_time_list(fname):
    """ reads in the S1 and S2 times from csv file fname 
        returns time_interval_dict with root filename and list of file contents
    """
    
    time_interval_dict = {}
    if(fname):
        with open(fname, 'rt') as f:
            try:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    fname = row[0]
                    root = fname.split('.')[0]                
                    time_interval_dict[root] = [float(row[1]),float(row[2])]
            finally:
                f.close()

        assert(header[0]=='Filename' or header[0]=='root')
        assert(header[1]=='S1' or header[1]=='s1')
        assert(header[2]=='S2' or header[2]=='s2')
    
    return time_interval_dict
#------------------------------------------------------------------------
def do_all(args):
    KEY_DATA = list(range(1,4)) + [400,404]
    time_interval_dict = read_time_list(args.l)
    header, face_dict = get_face_dict(args.i, time_interval_dict, args.b, args.e, \
                                     KEY_DATA, args.t)
    gmm_list = calculate_gmm(header, face_dict)
    
#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program for averaging a directory of files across columns.' 
    help_intro += ' example usage:\n\t$ ./avg_master.py -i \'example/*_openface.txt\''
    help_intro += ' -l example/list.csv -o avg.csv -t OPENFACE'
    help_intro += '\n\t$ ./avg_master.py -i \'OpenFace/*_openface.txt\''
    help_intro += ' -l example/list.csv -t OPENFACE -o  avg.csv'    
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-i', help='inputs, ex:example/*.txt', type=str, 
                        default='example/*.txt')
    parser.add_argument('-l', help='S1 S2 time list csv file', type=str, 
                        default='example/list.csv')
    parser.add_argument('-o', help='output file', type=str, default='avg.csv')
    parser.add_argument('-t', help='data type, {OPENFACE,SHORE,AFFECTIVA}', 
                        type=str, default='OPENFACE')
    parser.add_argument('-b', help='begin time', type=float, default=0)
    parser.add_argument('-e', help='end time', type=float, default=999999999)
    args = parser.parse_args()
    
    print('args: ',args.i, args.l, args.o)

    #if not os.path.isdir(args.i):
    #    logging.error(args.i + ' directory does not exist')
    #    exit()
    if args.l and (not os.path.exists(args.l)):
        logging.warning(args.l + ' list file does not exist.\n')
    
    do_all(args)
