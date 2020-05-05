import numpy as np
import pandas as pd

import sys 
#sys.path.append(r'M:\Python\Personal\DataAnalysis')
sys.path.append(r'C:\Development\Git\DataAnalysis')

from Analysis import Plotting, Stats
import plotly.graph_objects as go

from fancyimpute import KNN, SoftImpute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import sklearn.feature_selection as feature_selection
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.svm as svm

import factor_analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import calculate_kmo

from imblearn.under_sampling import (NearMiss, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule)
from imblearn.over_sampling import (SMOTE,
                                    ADASYN, 
                                    KmeansSMOTE,
                                    SVMSMOTE)

def FactorAnalysis(df, rotation = "varimax", n_factors = 10, transform = False):

    """ You want "varimax" rotation if you want orthogonal (highly differentiable) with very high and low variable loading. common
        You want "oblimin" for non-orthogonal loading. Increases eigenvalues, but reduced interpretability.
        You want "promax" if you want Oblimin on large datasets.
        
        See https://stats.idre.ucla.edu/spss/output/factor-analysis/ for increased explination. 
    """   

    assert not df.isnull().values.any(), "Data must not contain any nan or inf values"
    assert all(df.std().values > 0), "Columns used in Factor Analysis must have a non-zero Std. Dev. (aka more than a single value)"  

    def data_suitable(df, kmo_value = False, ignore = False):
        
        #Test to ensure data is not identity Matrix
        chi_square_value, p_value = calculate_bartlett_sphericity(df)
        
        # test to ensure that observed data is adquite for FA. Must be > 0.6
        kmo_all, kmo_model = calculate_kmo(df)

        if (p_value > 0.1 or kmo_model < 0.6) and ignore != True:
            raise Exception("Data is not suitable for Factor Analysis!: Identity test P value: {}.  KMO model Score: {}".format(p_value, kmo_model))
        
        if kmo_value:
            return kmo_model
        else:
            return
        
        
    print("KMO Value: {}.".format(data_suitable(df, kmo_value = True)))

    fa = FactorAnalyzer(method = "minres", 
                        rotation = rotation,
                        n_factors = n_factors)

    fa.fit(df)

    def eigenplot(df):
        df = pd.DataFrame(df)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x = df.index.values,
                y = df[0].values,
                mode = 'lines'
            )
        )
        
        
        fig.add_shape(
            type = "line",
            y0 = 1,
            x0 = 0,
            y1 = 1,
            x1 = len(df),
            line = dict(
                color = 'red',
                dash = 'dash'
            )
        )
        
        fig.update_layout(
            title = "Factor Eigenvalues",
            yaxis_title="Eigenvalue",
            xaxis_title="Factor",
            xaxis = dict(
                range = [0,df[df[0] > 0].index.values[-1]]
                )
        )
        
        fig.show()
        return

    eigenplot(fa.get_eigenvalues()[1])
    Plotting.LabeledHeatmap(fa.loadings_, y = list(df.columns), title = "Factor Loading", expand = True, height = 2000, width = 2000)

    tmp = pd.DataFrame(fa.get_factor_variance()[1:]) 
    tmp.index = ["Proportional Varience","Cumulative Varience"]
    Plotting.dfTable(tmp)

    if rotation == 'promax':
        Plotting.LabeledHeatmap(fa.phi_, title = "Factor Correlation", expand = True, height = 2000, width = 2000)
        Plotting.LabeledHeatmap(fa.structure_, y = list(df.columns), title = "Variable-Factor Correlation", expand = True, height = 2000, width = 2000)

    Plotting.LabeledHeatmap(pd.DataFrame(fa.get_communalities()).T, 
                            title = "Varience Explained",
                            x = list(df.columns), 
                            description = "The proportion of each variables varience that can be explained by the factors.", 
                            expand = True, 
                            height = 300, 
                            width = 2000)

    Plotting.LabeledHeatmap(pd.DataFrame(fa.get_uniquenesses()).T, 
                            title = "Variable Uniqueness",
                            x = list(df.columns),
                            expand = True, 
                            height = 300,
                             width = 2000)

    if transform:
        return fa.transform(df)

    return 


def OverSample(df, _class, method = 'ADASYN', strategy = 'auto', knn = 5, n_jobs = , ratio = None,  stepsize = 0.5):
   """ 
    SMOTE - Synthetic Minority Oversampling Technique
    ADASYN - Adaptive Synthetic Sampleing - Smote but focuses on minority density.
    SVMSMOTE - SMOTE but utilizes SVM to identify boundries
    KMeans - SMOTE but utilizing KMeans clustering ahead of time
    #https://towardsdatascience.com/sampling-techniques-for-extremely-imbalanced-data-part-ii-over-sampling-d61b43bc4879
    """
    Y = df[_class]
    X = df.drop(_class, axis = 1)

    if method.lower() == 'smote':
        x, y = SMOTE(stratey = strategy, k_neighbors = knn, n_jobs = n_jobs, ratio = ratio).fit_resample(X, Y)
    elif method.lower() == 'adasyn':
        x, y = ADASYN(stratey = strategy, n_neighbors = knn, n_jobs = n_jobs, ratio = ratio).fit_resample(X, Y)
    elif method.lower() == 'svmsmote':
        x, y = SVMSMOTE(stratey = strategy, k_neighbors = knn, n_jobs = n_jobs, ratio = ratio, out_step = stepsize).fit_resample(X, Y)
    elif method.lower() == 'kmeans':
        x, y = KmeansSMOTE(stratey = strategy,  k_neighbors = knn, n_jobs = n_jobs, ratio = ratio, ).fit_resample(X, Y)
    else:
        raise Exception("{} is not a valid method for OverSampling".format(method))

    df = pd.DataFrame([x, y], columns = list(df.columns) + [_class])

    fig = go.Figure()

    fig.add_trace(
    
        go.Splom(
            dimensions = [
                dict(label = column, values = df[column]) for column in df.columns
            ], 
            marker = dict(
                color = df[_class]
            )
        )
    )

    fig.show()
    
    if transform:
        return df

    return


def UnderSample(df, _class, method = 'cc', strategy = 'auto', n_jobs = 1, ratio = None, transform = None, offline = None):
    """
       NearMiss - Select values which are closest to minority class.
       TomeLinks - uses connected sets between class borders which are closest. If there are no other points closer, it assumes they are noise or borderline and remove them.
       ENN - Edited Nearest Neighbors, remove instances from majorit which are near bordeline
       NCL - NeighborhoodCleaningRule - Uses ENN to remove majority samples. Finds Nearest neighbors and if all are correctly label it keeps them.
       CC - Cluster Centroids - Finds Clusters of Majority Samples with K-means, then keeps cluster centroids of the clusters as the new majority sample.   
    """
    #https://towardsdatascience.com/sampling-techniques-for-extremely-imbalanced-data-part-i-under-sampling-a8dbc3d8d6d8

    Y = df[_class]
    X = df.drop(_class, axis = 1)

    if method.lower() == 'nearmiss':
        x, y = NearMiss(stratey = strategy, n_jobs = n_jobs, ratio = ratio).fit_resample(X, Y)
    elif method.lower() == 'tomelinks':
        x, y = TomekLinks(stratey = strategy, n_jobs = n_jobs, ratio = ratio).fit_resample(X, Y)
    elif method.lower() == 'ncl':
        x, y = NeighbourhoodCleaningRule(stratey = strategy, n_jobs = n_jobs, ratio = ratio).fit_resample(X, Y)
    elif method.lower() == 'cc':
        x, y = ClusterCentroids(stratey = strategy, n_jobs = n_jobs, ratio = ratio).fit_resample(X, Y)
    else:
        raise Exception("{} is not a valid method for UserSampling".format(method))

    df = pd.DataFrame([x, y], columns = list(df.columns) + [_class])

    fig = go.Figure()

    fig.add_trace(
    
        go.Splom(
            dimensions = [
                dict(label = column, values = df[column]) for column in df.columns
            ], 
            marker = dict(
                color = df[_class]
            )
        )
    )

    fig.show()
    
    if transform:
        return df
    
    return


def Imputer(df, method = 'Mice', knn = 5, transform = None, offline = None, save_name = None, width = None, height = None):
    """ Mice, KNN, or SoftImpute """
    
    missing_values = []
    
    for key, row in df[df.isnull().any(1)].iterrows():
        row = row[row.isnull()].index
        for item in row:
            missing_values.append((key, item))
    
    colors = pd.DataFrame(np.zeros(df.shape), columns = df.columns)
    for val in missing_values:
        colors.loc[val] = 1
    
        
    if method.lower() == 'mice':
        model = IterativeImputer(random_state = 123)
        df    = pd.DataFrame(model.fit_transform(df), columns = df.columns)

    elif method.lower() == 'KNN':
        model  = KNN(k=knn)
        df     = pd.DataFrame(model.fit_transform(df), columns = df.columns)
        
    elif method.lower() == 'soft':
        model  = SoftImpute()
        df     = pd.DataFrame(model.fit_transform(df), columns = df.columns)

    else:
        raise Exception("{} is not a valid method for Impute".format(method))        

    Plotting.Splom(df, colors, offline = offline, save_name = save_name, width = None, height = None)
    
    if transform:
        return df
    
    return



class FeatureAnalysis():
    
    def __init__(self, df, class_, kind = 'classification'):
        """This class generates a Feature selection DF, which can be viewed to identify the best variables for modeling
            Recall - chi2 shows error between expected and observed - thus lower = better, 
                     f_score tests if the means between two value are different, if significant = they are similair. 
                     mutual infomration shows how much randomness there is between the variables, thus lower = better
        """
        
        self.y = df[class_]
        self.x = df[[column for column in df.columns if column != class_]] 
        
        lst = []
        combine = dict()
        for column in self.x.columns:
            combine['column'] = column
            combine.update(Stats.BoxPlotStats(self.x[column]))
            combine['outlier_count'] = len(self.x[(self.x[column] > combine['d2']) | (self.x[column] < combine['d1'])]) 
            combine['std_perc'] = combine['std']/(combine['max'] - combine['min'])
            lst.append(combine.copy())
        
        self.stats = pd.DataFrame(lst).set_index('column')
        self.n_jobs = -1 if len(self.x.columns) > 100 else 1
        self.type = kind
        return

    def classification_analysis(self):

        tmp = dict()
        #linear
        tmp['logic'] = feature_selection.RFECV(lm.LogisticRegression(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['ridge'] = feature_selection.RFECV(lm.RidgeClassifier(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['SGD'] = feature_selection.RFECV(lm.SGDClassifier(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['lm_svm'] = feature_selection.RFECV(svm.LinearSVC(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_

        #non-linear
        tmp['ADABoost'] = feature_selection.RFECV(ensemble.AdaBoostClassifier(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['RandomForest'] = feature_selection.RFECV(ensemble.RandomForestClassifier(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_

        #stats
        chi = feature_selection.chi2(self.x,self.y)
        tmp['chi2'] = chi[0]
        tmp['chi2_pval'] = chi[1]
        fscore = feature_selection.f_classif(self.x,self.y)
        tmp['f_score'] = fscore[0]
        tmp['f_pval'] = fscore[1]
        tmp['MIC'] = feature_selection.mutual_info_classif(self.x,self.y)

        return tmp


    def regression_analysis(self):

        tmp = dict()
        #linear
        tmp['logic'] = feature_selection.RFECV(lm.LinearRegression(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['ridge'] = feature_selection.RFECV(lm.Ridge(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['SGD'] = feature_selection.RFECV(lm.SGDRegressor(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['lm_svm'] = feature_selection.RFECV(svm.LinearSVR(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_

        #non-linear
        tmp['ADABoost'] = feature_selection.RFECV(ensemble.AdaBoostRegressor(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_
        tmp['RandomForest'] = feature_selection.RFECV(ensemble.RandomForestRegressor(), cv=5, n_jobs = self.n_jobs).fit(self.x,self.y).ranking_

        #stats        
        fscore = feature_selection.f_regression(self.x,self.y)
        tmp['f_score'] = fscore[0]
        tmp['f_pval'] = fscore[1]
        tmp['MIC'] = feature_selection.mutual_info_regression(self.x,self.y)

        return tmp           

    def run(self):
        if self.type.lower() == 'classification':
            return pd.concat([self.stats, pd.DataFrame(self.classification_analysis())], axis=1)
        else:
            return pd.concat([self.stats, pd.DataFrame(self.regression_analysis())], axis=1)




# development
def CentroidModeling():
    """
        This process utilizes clustering algorthms, then replaces data with centroids. Returns data which
        is significantly slimmed down, but makes the modeling easier.
    """
    return

def IterativeBalanceSampling(df, ):
    """ This function creates a several different sample sets based on varying methodologies of undersampling
    
    """

    #oversampling
    for method in ['smote', 'svmsmote', 'adasyn', 'kmeans']:
        print(method)

    #undersampling
    for method in ['nearmiss', 'tomelink', 'ncl', 'cc', 'none']:
        print(method)

    return

def UnBalancedCrossValidation(model, df, _class, n_splits = 10, method = 'all'):
    """
        This process handles overfitting and data balancing such that CrossValidation can be utilized. 
        
        Note only use K = 5 if comparing models

        This process is based off of research found here: https://www.researchgate.net/publication/328315720_Cross-Validation_for_Imbalanced_Datasets_Avoiding_Overoptimistic_and_Overfitting_Approaches
    """

    df = df.sample(frac = 1).reset_index(drop = True)
    tmp = IterativeBalanceSampling(df)

    return
