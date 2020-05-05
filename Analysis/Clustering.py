import gc
import sys 
sys.path.append(r'M:\Python\Personal\DataAnalysis')
sys.path.append(r'C:\Development\Git\DataAnalysis')

import math
import pandas as pd

from Analysis import Plotting

import plotly.express as px
import plotly.figure_factory as ff

import hdbscan
from scipy.spatial import distance


#HDBScan with Andrews Curve Plot
def HDBScan(df, min_cluster_size = None, min_samples = None, epsilon = None, alpha = 1.0, 
            single_cluster = False, outliers = None, new_column = "Cluster", offline = None,
            transform = None):

    """
    Min_cluster_size = Litteral
    min_samples = how conservative the model is.
    epsilon = how agressive similiar clusters merge, higher equals fewer clusters. Value based on data distance units, thus value could be 50 for 50 unit distance.
    alpha = another conservative value that you probably shouldn't need to messs with, but works on a tighter scale.'
    
    Note Outlier should be a cutoff value.
    
    Allow single cluster for anomoly detection / outlier detection.
    """

    assert all([pd.api.types.is_numeric_dtype(df[column]) for column in df.columns])

    if not min_cluster_size:
        min_cluster_size = int(len(df) * 0.1)

    if not min_samples:
        min_samples = int(min_cluster_size * .333)   

    if not epsilon:
        distances = distance.cdist(df.values, df.values)
        mean = distances.mean()
        std  = distances.std()
        epsilon = float(min(mean, std))
        
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,        
        min_samples = min_samples,                  
        cluster_selection_epsilon = epsilon,        
        alpha = alpha,
        allow_single_cluster = single_cluster,
        core_dist_n_jobs = -1,
        gen_min_span_tree = True
    )    
    clusterer.fit(df)
    
    tmp = df.copy()
    tmp[new_column] = clusterer.labels_
    tmp["{}_Probabilities".format(new_column)] = clusterer.probabilities_
    tmp['{}_outlier_prob'.format(new_column)]  = clusterer.outlier_scores_
    
    if not offline:
        Plotting.dfTable(tmp.head(50), title = "Results of HDBScan")
        Plotting.dfTable(pd.DataFrame(clusterer.cluster_persistence_), title = "Cluster Persistence")
        Plotting.Title("Density Based Cluster Validity Score", description = clusterer.relative_validity_)
    else:
        print("Cluster Persistence")
        print(pd.DataFrame(clusterer.cluster_persistence_))
        print()
        print("Density Based Cluster Validity Score: {}".format(clusterer.relative_validity_))
     
    if outliers:
        
        if not isinstance(outliers, float):
            outliers = 0.85
            
        tmp['{}_isoutlier'.format(new_column)] = tmp['{}_outlier_prob'.format(new_column)].apply(lambda x: 1 if x > outliers else 0)
        
        fig = ff.create_distplot([tmp['{}_outlier_prob'.format(new_column)].values], group_labels = ["Distribution"], bin_size = 0.1)
        fig.update_layout(title = 'Outlier Probability Distribution')
        Plotting.Plot(fig, offline = offline)
        
        if len(tmp['{}_isoutlier'.format(new_column)].unique()) > 1:
            fig = px.scatter_matrix(tmp, 
                              dimensions = [column for column in df.columns], 
                              color = '{}_isoutlier'.format(new_column))
            fig.update_layout(title = "Outlier Distribution")
            Plotting.Plot(fig, offline = offline)

    tmp1 = df.copy()
    tmp1['Cluster'] = clusterer.labels_ 
    Plotting.AndrewsCurve(tmp1, 'Cluster', offline = offline)
    del tmp1
    fig = px.scatter_matrix(tmp, 
                      dimensions = [column for column in df.columns],
                      color = new_column)
    Plotting.Plot(fig, offline = offline)

    if transform:
        return tmp
    
    gc.collect()
    return 