import pandas as pd
import sklearn
import autosklearn

class AutoML(object):

    """
    AutoML requires being run on Linux or Google Colab. 
    To run on google Colab, the install is:

    !apt-get install swig -y
    !pip install Cython numpy
    !pip install auto-sklearn

    it is recommended to let it run for a day total and up to 30 minutes for a single run
    
    for more info: https://www.automl.org/, https://automl.github.io/auto-sklearn/master/index.html
    """

    def __init__(df, y, max_min = 30, max_model_min = 10, split = None, train_size = .75, dataset_name = None, column_types = None):

        if split:
            self.X = df[[column for column in df.columns if column != y]]
            self.Y = df[y]
        else:
            self.X = df
            self.Y = y

        if not column_types:
            column_types = list()
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    column_types.append('numerical')
                else:
                    column_types.append('categorical')

        self.column_types = column_types

        self.x_train, self.y_train, self.x_test, self.y_test = \
            sklearn.model_selection.train_test_split(X, Y, train_size = train_size, random_state = 321)

        self.name = dataset_name
        

    def Classification(self, n_jobs = -1, max_mem = 12000):
        self.model = \
            autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task = 60*max_min, # run auto-sklearn run time
                per_run_time_limit = 60*max_model_min, # Most time spent on any single model type
                tmp_folder='/tmp/autosklearn_{}'.format(self.name),
                output_folder='/tmp/autosklearn_{}'.format(self.name),
                disable_evaluator_output=False,
                resampling_strategy='partial-cv',
                resampling_strategy_arguments=dict(folds = 5, shuffle = True),
                ensemble_memory_limit = max_mem / 4,
                ml_memory_limit = max_mem,
                n_jobs=jobs,
                delete_output_folder_after_terminate=False, 
                delete_tmp_folder_after_terminate=True,
            )

        self.model  = self.model.fit(self.x_train.copy(), self.y_train.copy(), dataset_name = self.name)
        self.model  = self.model.refit(self.x_train.copy(), self.y_train.copy())
        self.pred_proba = self.model.pred_proba(self.x_test)
        self.pred = self.model.predict(self.x_test)
        self.model_weights = self.model.get_models_from_weights()
        self.params = self.model.get_params()
        self.cv_results_ = pd.DataFrame(self.classification.cv_results_)
        self.sprint_stats = self.classification.sprint_statistics()
        self.results = self.classification.show_models()
        return


    def Regression(self, n_jobs = -1, max_mem = 12000):
        self.model = \
            autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task = 60*max_min, # run auto-sklearn run time
                per_run_time_limit = 60*max_model_min, # Most time spent on any single model type
                tmp_folder='/tmp/autosklearn_{}'.format(self.name),
                output_folder='/tmp/autosklearn_{}'.format(self.name),
                disable_evaluator_output=False,
                resampling_strategy='partial-cv',
                resampling_strategy_arguments=dict(folds = 5, shuffle = True),
                ensemble_memory_limit = max_mem / 4,
                ml_memory_limit = max_mem,
                n_jobs=jobs,
                delete_output_folder_after_terminate=False, 
                delete_tmp_folder_after_terminate=True,
            )
        self.model  = self.model.fit(self.x_train.copy(), self.y_train.copy(), dataset_name = self.name)
        self.model  = self.model.refit(self.x_train.copy(), self.y_train.copy())
        self.pred = self.classification.predict(self.x_test)
        self.cv_results_ = pd.DataFrame(self.classification.cv_results_)
        self.sprint_stats = self.classification.sprint_statistics()
        self.results = self.classification.show_models()
        self.model_weights = self.model.get_models_from_weights()
        self.params = self.model.get_params()
        return


import ConfigSpace as CS 
from hpbandster.core.worker import Worker


def SEM(df, mod, optomizer = 'MLW'):
    """
    Structural Equation Models
    https://bitbucket.org/herrberg/semopy/src/master/
    https://arxiv.org/pdf/1905.09376.pdf

    #The Measurement Model is akin to the factor analysis. A latent variable is based on the observed values, In this case ind60 is a latent variable and x1, x2, and x3 are observd variables. Ind60 is effectively loaded by these variables. 
    #The Regressions are how the latent variables interact to predict


    Mod Example:
    # measurement model 
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
    # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
    # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8

    #Optomizers:
    MLW - Maximum Likelihood
    GLS - General Least Squares
    ULS - Inweighted Least Squares
    L-BGFS-B - LBGF limited Memory
    Adam
    SGD 
    Nestrov - An accelerated Stochastic Gradient Method


    Good ways to test accuracy, look at Covarience Matrix of exodrogenous variables and covariance values of the model. They should effectively be the same.
    """
    import semopy

    ### Add thing to check degrees of Freedom


    model = Model(mod)
    model.load_dataset(df)

    opt = Optimizer(model)
    loss = opt_mlw.optimize(objective=optomizer) 

    print('Resultant objective functions {} values are:'.format(optomizer))
    print('{}: {:.3f}'.format(optomizer, loss))

    print(semopy.inspector.inspect(opt, mode='list'))

    return 


#HyperParemeterSelection
class BOHB(object):
    """ 
    Baysian optomized Hyperbandings. Current best method for identifying hyperparameters. 
    """

    class Workers(Worker):
        
        def __init__(self, *args, sleep_interval = 0, **kwargs):
            super().__init__(*args, **kwargs)

            self.sleep_interval = sleep_interval

        def compute(self, config, budget, **kwargs):


#OptimalInputAlgothm
def SMAC():
    """ 
    -Is a tool designed to optomize parameters for any algorthmn.
    -Deterministic of Probalistic 
    -Utilizes a sped up Bayesian Optomization to between two options performing better.
    -Requires swig and thus also requires using a linux system.
    https://github.com/automl/SMAC3
    https://github.com/automl/SMAC3/blob/master/examples/SMAC4BO_rosenbrock.py
    """
    return

#ParameterImportance
def PyImp():
    """
     -Utilizes SMAC outputs to determine most important parameters. 
    https://github.com/automl/ParameterImportance
    """
    return