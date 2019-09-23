#==============================================================================
# Operations on data which are different from loading / storing / augmentation
#==============================================================================

import numpy as np
import scipy.stats as sps
from load_data import load_data


def dataset_metrics (dataset, dim_variance_coll='avg', T_threshold=0.05):
    '''
    dataset: Path to dataset
    For any dataset, compute metrics L, U, D, T as given in ICCCNT paper
    For L and U:
        Assume dimensions in classes have Gaussian variances and they can be averaged/medianed over dimensions
            dim_variance_coll: 'avg' or 'med'
            ** Averaging works better **
        Prior probabilities and different variances for different classes ARE taken into account
    For T:
        Find number of d1 values below T_threshold
    '''
    
    print('Starting for dataset = {0}'.format(dataset))
    xtr, ytr, xva, yva, xte, yte = load_data(filename=dataset)
    xdata = np.concatenate((xtr,xva,xte), axis=0)
    ydata = np.concatenate((ytr,yva,yte), axis=0)
    xdata = xdata.reshape(xdata.shape[0],-1) #if xdata has multiple dimensions (like num,32,32,3 for cifar), it should be flattened to num,-1
    del (xtr,ytr,xva,yva,xte,yte)

    numlabels = ydata.shape[1]
    numdims = xdata.shape[1]
    numsamples = ydata.shape[0] # = xdata.shape[0]
    label_occurrences = np.zeros(numlabels) #how many times each label occurs

    ## Calculate label centroids ##
    label_dim_totals = np.zeros((numlabels,numdims))
    for x,y in zip(xdata,ydata):
        label = np.flatnonzero(y)[0] #identify label for a particular example
        label_occurrences[label] += 1
        label_dim_totals[label] += x
    label_dim_centroids = label_dim_totals/label_occurrences[:,np.newaxis] #Centroid of each label, i.e. filtered value. Shape (numlabels,numdims)

   ## [OPTIONAL] Visualize centroids ##
#==============================================================================
#     sidelength = int(np.sqrt(numdims))
#     for i in range(numlabels):
#         Image.fromarray((label_dim_centroids[i].reshape(sidelength,sidelength)*255).astype('uint8')).show()
#==============================================================================

    ## Calculate average variance across all dimensions for each label ##
    label_dim_squareddeviation_totals = np.zeros((numlabels,numdims))
    for x,y in zip(xdata,ydata):
        label = np.flatnonzero(y)[0] #identify label for a particular example
        label_dim_squareddeviation_totals[label] += (x-label_dim_centroids[label])**2
    label_dim_variances = label_dim_squareddeviation_totals/label_occurrences[:,np.newaxis] #Variance of each label in each dimensions. Shape (numlabels,numdims)
    if dim_variance_coll == 'avg':
        label_coll_variances = np.average(label_dim_variances, axis=1) #average variance across all dimensions for each label. Shape (numlabels,)
    elif dim_variance_coll == 'med':
        label_coll_variances = np.median(label_dim_variances, axis=1) #median variance across all dimensions for each label. Shape (numlabels,)

    ## [OPTIONAL] Plot variances across all dimensions for each label to see if mean or median makes sense ##
#==============================================================================
#     for i in range(numlabels):
#         print('Average for label {0} = {1}...'.format(i,label_coll_variances[i])
#         plt.hist(label_dim_variances[i])
#         plt.show()
# #        plt.plot(sorted(label_dim_variances[i]))
# #        plt.show()
#==============================================================================

    ## Calculate label probabilities (priors) ##
    label_probs = label_occurrences/float(numsamples) #Shape (numlabels,)

    ## Calculate centroid L2 distance (d) and average L1 distance (d1) between all pairs of labels ##
    label_d = np.zeros((numlabels,numlabels))
    label_d1 = np.zeros((numlabels,numlabels))
    for r in range(1,numlabels):
        for c in range(r):
            label_d[r,c] = np.linalg.norm((label_dim_centroids[r]-label_dim_centroids[c]), ord=2)
            label_d1[r,c] = np.linalg.norm((label_dim_centroids[r]-label_dim_centroids[c]), ord=1)/float(numdims)
    label_d += label_d.T #make d symmetric, diagonal elements stay 0
    #No need to make d1 symmetric

    ## Calculate T metric ##
    label_d1 = label_d1[np.nonzero(label_d1)]
    T = np.count_nonzero(label_d1 < T_threshold)

    ## Calculate dmin and sigmabydmin for each label
    label_dmin = np.partition(label_d, kth=1, axis=1)[:,1] #Get 2nd minimum of each row (since min is the diagonal element=0). Shape (numlabels,)
    label_sigmabydmin = label_coll_variances/label_dmin

    ## [OPTIONAL] Plot sigmabydmin
#==============================================================================
#     plt.plot(sorted(label_simgabydmin))
#     plt.show()
#==============================================================================

    ## Calculate D metric ##
    D = np.average(label_sigmabydmin)
#    Dmed = np.median(label_sigmabydmin)

    ## Calculate L and U metrics (i.e. error bounds) ##
    L = U = 0.
    for m in range(numlabels):
        L += label_probs[m]*sps.norm.sf(np.sqrt(label_dmin[m]**2/(4*label_coll_variances[m]**2)))
        U_inner = 0.
        for j in range(numlabels):
            if j==m:
                continue
            else:
                U_inner += sps.norm.sf(np.sqrt(label_d[m,c]**2/(4*label_coll_variances[m]**2)))
        U += label_probs[m]*U_inner

    return L,U,D,T
