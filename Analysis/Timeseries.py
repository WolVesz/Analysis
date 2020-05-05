
import sys 
sys.path.append(r'M:\Python\Personal\DataAnalysis')
sys.path.append(r'C:\Development\Git\DataAnalysis')

from Analysis import Plotting

import pandas as pd
import plotly.graph_objects as go
from pandas.plotting import autocorrelation_plot

def AutoCorrelationPlot(df, title = 'Autocorrelation'):
    """ 
    Creates a graph of Auto Correlation. This implies the number of lag cycles and their level of correlation.
    """
    
    assert len(df.columns) == 1, "Requires Dataframe with one column with Index = Datatime"
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be of type datetime"
        
    ax = pd.plotting.autocorrelation_plot(df)
    
    upper99 = ax.lines[0].get_data()[1]
    upper95 = ax.lines[1].get_data()[1]
    lower95 = ax.lines[3].get_data()[1]
    lower99 = ax.lines[4].get_data()[1]
    data    = ax.lines[5].get_data()[1]
    
    x = list(range(1,len(data)))
    y = data
    
    fig = Go.Figure()

    fig.add_trace(
        
        go.Scatter(
            x = x,
            y = y,           
            mode = 'lines'
        )
        
    )
    
    #upper99
    fig.add_shape(
        type = 'line',
        x0 = 0,
        x1 = x[-1],
        y0 = upper99[0], 
        y1 = upper99[1],
        line = dict(
            color = 'red'
        )
    )

    #upper95
    fig.add_shape(
        type = 'line',
        x0 = 0,
        x1 = x[-1],
        y0 = upper95[0], 
        y1 = upper95[1],
        line = dict(
            color = 'black', 
            dash = 'dash'
        )
    )
    
    #lower95
    fig.add_shape(
        type = 'line',
        x0 = 0,
        x1 = x[-1],
        y0 = lower95[0], 
        y1 = lower95[1],
        line = dict(
            color = 'black', 
            dash = 'dash'
        )
    )
    
    #lower99
    fig.add_shape(
        type = 'line',
        x0 = 0,
        x1 = x[-1],
        y0 = lower99[0], 
        y1 = lower99[1],
        line = dict(
            color = 'red'
        )
    )
        
    fig.update_layout(
        title = title, 
        xaxis_rangeslider_visible = True
    )    
    
    Plotting.Plot(fig, offline = True)


class ARIMA:
    
    import pmdarima as pm
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    import plotly.graph_objects as go
    
    def __init__(self, series, X = None, period = None, lag = None):
        self.data = series
        self.diff = data.diff().dropna()
        self.lag  = int(len(series)/2) if not lag else lag
        
        if X:
            self.X = X
        
        if period:
            self.period = period

        return
    
    def setup(self, period = None, lag = None, regression = 'ct'):
        
        if period:
            self.period = period
        
        if lag:
            self.lag = lag
            
        assert self.lag, "This function requires a Lag value to be set to work."
        assert self.period, "This function requires a Period value to be set to work."
        
        res = seasonal_decompose(series, period = self.period)
        fig = res.plot()
        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.show()
        
        sm.graphics.tsa.plot_acf(series, lags=self.lag)
        sm.graphics.tsa.plot_pacf(series, lags=self.lab)
        
        adf_val = adfuller(series, max_lag = self.lag, regresion = regression, autolag = 'AIC')
        print("Augmented Dickey-Fuller")
        print("This test is used to determine if the Timeseries is stationary. A p-value of < 0.05 implies that the TS is stationary.")
        print('ADF Statistic: {}, P-value: {}'.format(adf_val.adf, adf_val.pvalue))
        print("Number of Lags used during AIC: {}".format(adf_val.usedlag))
        
        print('KPSS Test')
        print('This test is used to determin if the Timeseries is starionary. A ')
        
        return

    
    def AutoARIMA(self, X = None, duration = 60, max_iter = 200, period = None, pred_periods = 10, start_params = None, full_period_x = None, x_name = None, y_name = None):
        """
        Parameters
        ----------
        X : Series or Array, optional
            Exogenious Variable to apply to the model.
        Duration : int, optional
            Max time spent checking variables. Should really never be needed. The Default is 60 Seconds.
        max_iter : Int, optional
            The number of iterations allowed. The default is 200. Should really never be needed.
        period : Int, optional
            The number of periods in each seasonality cycle. This is a require variable but can be added
            during initialization. The default is None.
        pred_periods : Int, optional
            The number of values to predict. The default is 10.
        start_params : Tuple, optional
            If you want to define your own p,d,q to start the search with. The default is None.
        full_period_x : Series or Array Like Object, optional
            The values to be inputted into X axis of ARIMA graph. The default is None.
        x_name : String, optional
            Title of X-axis. The default is None.
        y_name : String, optional
            Title of Y-axis. The default is None.

        Returns
        -------
        None.

        """
        if X:
            self.X = X
        
        if period:
            self.period = period
         
        with pm.arima.StepwiseContext(max_dur=duration):
            model = pm.auto_arima(series,
                                  exogenous = self.X,
                                  stepwise=True,
                                  start_p = 0,
                                  start_d = 0,
                                  start_q = 0,
                                  max_p = 5, 
                                  max_d = 2, 
                                  max_q = 5,
                                  
                                  seasonal = True,
                                  start_P = 0,
                                  start_Q = 0,
                                  max_P = 2,
                                  max_D = 1,
                                  max_Q = 2, 
                                  m = self.period,
                                  
                                  start_parms = start_params,
                                  
                                  maxiter = max_iter,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  trace = True
                                  )
            
            self.insample = model.predict_in_sample(return_conf_int=True)
            self.pred   = model.predict(pred_periods, return_conf_int=True)
            self.params = model.get_params
            
            if not full_period_x:
                full_period_x = list(range(0,len(series) + pred_periods))
                
            
            fig = go.Figure()
            
            #lowerbound
            fig.add_trace(
                go.Scatter(
                    x = full_period_x,
                    y = list(insample[-1][:,0]) + list(pred[-1][:,0]),
                    name = 'Lower Confidence Bound',
                    mode = 'lines', 
                    line = dict(
                        color = 'grey', 
                    )
                )
            )
            
            #upperbound
            fig.add_trace(
                go.Scatter(
                    x = full_period_x,
                    y = list(insample[-1][:,1]) + list(pred[-1][:,1]),
                    name = 'Upper Confidence Bound',
                    mode = 'lines',
                    line = dict(
                        color = 'grey', 
                    ),
                    fill = 'tonexty',
                    fillcolor = 'rgba(128, 128, 128, 0.2)'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x = full_period_x,
                    y = series.values,
                    name = 'Actual Data',
                    mode = 'lines',
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x = full_period_x,
                    y = insample[0],
                    name = 'Predicted Insample',
                    mode = 'lines',
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x = full_period_x[pred_periods * - 1:],
                    y = pred[0],
                    name = 'Predicted Forcast',
                    mode = 'lines',
                )
            )          
            
            
            fig.update_layout(
                title = 'ARIMA',
                xaxis = dict(
                    range = [full_period_x[period], full_period_x[-1]]
                ), 
                xaxis_title = x_name,
                yaxis_title = y_name
            )
            
            Plotting.Plot(fig, offline = True)
            
            model.plot_diagnostics()
            print(model.summary())
            
        return
    
    
    def save_model(self, save_name):
        import pickle
        
        with open(save_name, 'wb') as f:
            pickle.dump(self.model, f)
            
        print('File Saved: {}'.format(save_name))
        return

        
def Prophet():

    return