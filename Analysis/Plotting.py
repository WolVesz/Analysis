#pip install scikit-misc
import sys 
sys.path.append(os.getenv("Analysis"))

import time

import pandas as pd

from Analysis import Stats
from Utilities import Basic

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display, HTML
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt

init_notebook_mode(connected=True)

import hdbscan
from skmisc.loess import loess
from sklearn.preprocessing import MinMaxScaler

config = dict(showLink = False, displaylogo = False, editable = False)

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
#plotly.offline.init_notebook_mode(connected=True)


def Plot(fig, save_name = None, offline = None, config = config):

    if offline:
        plotly.offline.plot(fig, config = config)

    if save_name:
        plotly.offline.plot(fig, filename='{}.html'.format(save_name), config = config)

    fig.show()

    return

def loess_plot(x, y, title = '', name = None, frac = .75, weights = None, width = None, height = None, save_name = None, offline = None):
    
    X = x
    
    if pd.api.types.is_datetime64_any_dtype(x):
         X = [time.mktime(t.timetuple()) for t in x]
    
    l = loess(X,y, normalize = True, span = frac, weights = weights)
    l.fit()
    pred = l.predict(X, stderror=True)
    conf = pred.confidence()

    blue = '#348EA9'
    orange = '#F48B37'
    green = '#52BA9B'
    red = '#EF4846'

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = x,
            y = y,
            name = name,
            mode = "markers",
            marker = dict(
                color = "blue"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = conf.upper,
            mode = 'lines',
            line_color = 'grey',
            name = 'Upper 95% CI'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = conf.lower,
            mode = 'lines',
            line_color = 'grey', 
            name = 'Lower 95% CI'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = pred.values,
            name = 'Localized Regression', 
            mode = 'lines', 
            line_color = 'red'
        )
    )

    fig.update_layout(
        title = title,
        width = width if width else 800, 
        height = height if height else 500
    )

    Plot(fig, offline = offline, save_name = save_name)

    return




def Robust_loess_plot(x, y, title = '', name = None, frac = .75, min_cluster_size = None, weights = None, width = None, height = None, save_name = None, offline = None):
    
    if not isinstance(x, list):
        X = x.values
    
    if pd.api.types.is_datetime64_any_dtype(x):
         X = [time.mktime(t.timetuple()) for t in x]
    
    if min_cluster_size and not weights:

        scaler = MinMaxScaler()

        clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
        clusterer.fit(scaler.fit_transform(pd.DataFrame([X, y.values]).T))
        weights = clusterer.probabilities_ +.1

    l = loess(X,y, normalize = True, span = frac, weights = weights)
    l.fit()
    pred = l.predict(X, stderror=True)
    conf = pred.confidence()

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x = x,
            y = y,
            name = name,
            mode = "markers",
            marker = dict(
                color = "blue"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = conf.upper,
            mode = 'lines',
            line_color = 'grey',
            name = 'Upper 95% CI'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = conf.lower,
            mode = 'lines',
            line_color = 'grey', 
            name = 'Lower 95% CI'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = pred.values,
            name = 'Localized Regression', 
            mode = 'lines', 
            line_color = 'red'
        )
    )

    fig.update_layout(
        title = title,
        width = width if width else 800, 
        height = height if width else 500
    )

    Plot(fig, offline = offline, save_name = save_name)

    return




def DistributionSeries(data, y_names, title = None, xaxis_title = None, colors = None, offline = None, save_name = None):

    "data must be added as a list of each distribution you want to show. y_names are the associated labels"

    if not colors:
        colors = plotly.colors.n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(data), colortype='rgb')

    fig = go.Figure()

    for data_line, name, color in zip(data, y_names, colors):
        fig.add_trace(
            go.Violin(x=data_line, 
                line_color=color, 
                name = name, 
                spanmode = 'hard',
                box_visible=True, 
                meanline_visible=True,
                box = dict(
                    line = dict(
                        color = "black"
                        )
                    ),
                meanline = dict(
                    color = "black"
                )
            )
        )

    fig.update_traces(orientation='h', side='positive', width=2.5, points=False)
    fig.update_layout(xaxis_showgrid=True, 
                      xaxis_zeroline=False, 
                      height = len(data)*100 + 200,
                      title = title,
                      xaxis_title = xaxis_title)

    Plot(fig, offline = offline, save_name = save_name)

    return fig






def CorrelationHeatmap(df, width = None, height = None, mic = False):
    display(HTML("<style>.container { width:100% !important; }</style>"))

    df = df.select_dtypes(include=['int64','float64'])
    #stats
    correlations = Stats.Correlations(df, mic = mic)
    
    if mic:
        options = ['Pearson', 'Spearman', 'Kendall', 'MICe']
    else:
        options = ['Pearson', 'Spearman', 'Kendall']
    
    #dropdown
    textbox = widgets.Dropdown(
        description='Correlation:   ',
        value='Pearson',
        options= options
    )

    #empty Trace
    trace = go.Heatmap(
        x = df.columns, 
        y = df.columns,
        z = correlations["Pearson"],
        type = 'heatmap',
        colorscale = plotly.colors.n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')
    )

    #attched Trace to figure
    fig = go.FigureWidget(
        data   = [trace],
        layout = go.Layout(
            title = "Pearson",
            width = width if width else 800,
            height = height if height else 800
        ),
    )

    fig.update_layout(
        yaxis = dict(
            autorange = 'reversed'
        )
    )
    
    #update mechanism
    def response(change):
        
        if textbox.value:
            
            tmp_df = correlations[textbox.value]
            
            with fig.batch_update():
                fig.data[0].z = tmp_df.values
                fig.layout.title = textbox.value    
                
    textbox.observe(response, names = "value")
    display(widgets.VBox([textbox, fig]))

    return 



def Title(title = None, description = None):
    if title:
        display(HTML("<h2>{}</h2>".format(title)))

    if description:
        display(HTML("<p>{}</p>".format(description)))

    return    





def dfTable(df, title = None, description = None):

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        df = pd.DataFrame(df)

    Title(title, description)

    display(HTML(df.to_html()))
    return



def ColumnDetails(df):
    lst = []
    for column in df.columns:
        combine = dict()
        combine['column'] = column
        combine['type']   = str(df[column].dtype)
        combine['count']  = len(df[column].dropna())
        combine['unique'] = len(df[column].unique())
        if pd.api.types.is_numeric_dtype(df[column]):
            combine.update(Stats.BoxPlotStats(df[column]))
            combine['outlier_count'] = len(df[(df[column] > combine['d2']) | (df[column] < combine['d1'])]) 
            combine['std_perc'] = combine['std']/(combine['max'] - combine['min'])
        lst.append(combine.copy())
    
    output = pd.DataFrame(lst)
    output = output.set_index('column')
    
    dfTable(output, title = 'Data Description')
    return 



def LabeledHeatmap(df, x = None, y = None, width = None, height = None, expand = False, title = None, description = None, colors = None,):

    if expand:
        display(HTML("<style>.container { width:100% !important; }</style>"))

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        df = pd.DataFrame(df)

    if not x:
        x = list(df.columns)

    if not y:
        y = list(df.index.values)

    if not colors:
        colors = 'Picnic'

    Title(title, description)

    df = df.round(2)

    fig = ff.create_annotated_heatmap(z = df.values, x = x, y = y, annotation_text = df.values, colorscale=colors, hoverinfo = 'all')
    fig.update_layout(
        width = width if width else 800, 
        height = height if height else 800,
        yaxis = dict(
            autorange = 'reversed'
        )
    )

    Plot(fig, offline = offline, save_name = save_name)

    return 




def Splom(df, colors, title = None, width = None, height = None, offline = None, save_name = None):

    """
        This plot is an n*n Splom designed to handle cell specific colors
        I recommend generating the colors in a manner similiar to:

        missing_values = []
    
        for key, row in df[df.isnull().any(1)].iterrows():
            row = row[row.isnull()].index
            for item in row:
                missing_values.append((key, item))

        colors = pd.DataFrame(np.zeros(df.shape), columns = df.columns)
        for val in missing_values:
            colors.loc[val] = 1      
    """

    fig = make_subplots(rows = len(df.columns), 
                        cols = len(df.columns), 
                        shared_xaxes = True, 
                        shared_yaxes = True,
                        vertical_spacing = 0.01, 
                        horizontal_spacing = 0.01,
                        column_titles = list(df.columns), 
                        row_titles = list(df.columns)
                    )

    for count1, col1 in enumerate(df.columns):
        for count2, col2 in enumerate(df.columns):

            tmp_colors = colors[[col1, col2]].sum(axis = 1)
            
            fig.add_trace(

                go.Scatter(
                    x  = df[col1],
                    y  = df[col2],
                    mode = 'markers', 
                    marker = dict(
                        color = tmp_colors
                    ), 
                    name = None
                ),
                row = count1 + 1, 
                col = count2 + 1
            )   

    if not width and not height:
        width = 120*len(df.columns)
        height = 120*len(df.columns)

    fig.update_layout(
        title = title, 
        showlegend = False,
        height = height if height else None, 
        width = width if width else None
    )

    for i in range(0, len(df.columns)):
        fig.layout.annotations[i]["yref"] = "paper"
        fig.layout.annotations[i]["xref"] = "paper"
        fig.layout.annotations[i]["y"] = -0.06
        
        fig.layout.annotations[i + len(df.columns)]["yref"] = "paper"
        fig.layout.annotations[i + len(df.columns)]["xref"] = "paper"
        fig.layout.annotations[i + len(df.columns)]["x"] = -0.06
        
    Plot(fig, offline = offline, save_name = save_name)

    return



def AndrewsCurve(df, class_, title = 'Andrews Curve', colors = None, offline = None, save_name = None):
    """
    Identifies Data which is similiar between columns, colors the classes. Similiar Curves = Similiar Data,
    thus you will want similiar colors to be aligned with similiar curves.
    """

    if not colors:
        colors = plotly.colors.n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(df[class_].unique()), colortype='rgb')
    
    ax = pd.plotting.andrews_curves(df, class_column = class_)
    
    activated = dict()
    for val in df[class_].unique():
        activated[val] = False
    
    fig = go.Figure()
    
    for key, row in df.iterrows():

        x = ax.lines[key].get_data()[0]
        y = ax.lines[key].get_data()[1]
        
        fig.add_trace(
        
            go.Scatter(
                x = x,
                y = y,
                mode = 'lines',
                name = str(row[class_]),
                showlegend = True if activated[row[class_]] == False else False,
                line = dict(
                    color = colors[list(df[class_].unique()).index(row[class_])]
                )
            )
        )
        
        activated[row[class_]] = True
    
    fig.update_layout(
        title = title,
        legend_title = 'Category'
    )
    
    Plot(fig, offline = offline, save_name = save_name)    

    plt.ioff()
    return


def PairwiseDensityPlot(df, X = None, Y = None, scatter = True, width = 800, height = 800):

    if not X:
        X = df.select_dtypes(include = ['bool', 'float', 'int', 'category']).columns.values
    
    if not Y:
        Y = df.select_dtypes(include = ['bool', 'float', 'int', 'category']).columns.values
        
    assert Basic.isCollection(X), "X needs to be a collection type. If inputting only a single value, add it as a list."
    assert Basic.isCollection(Y), "Y needs to be a collection type. If inputting only a single value, add it as a list."
    
    x = df[X[0]].values
    y = df[Y[0]].values
    
    x_dropdown = widgets.Dropdown(
        options=X,
        value=X[0],
        description='X Axis Value:',
    )
        
    y_dropdown = widgets.Dropdown(
        options=Y,
        value=Y[0],
        description='Y Axis Value:',
    )        

    container = widgets.HBox([x_dropdown, y_dropdown])
    
    #start of figure
    traces = []

    traces.append(
        go.Histogram2dContour(
            x = x,
            y = y,
            colorscale = 'Blues',
            reversescale = True,
            xaxis = 'x',
            yaxis = 'y',
            name = 'Contour', 
            showscale = False
        )
    )
    
    if scatter:
        #with enough datapoints, this will slow down the graph build or make it look fugly. 
        traces.append(
            go.Scatter(
                x = x,
                y = y,
                xaxis = 'x',
                yaxis = 'y',
                mode = 'markers',
                marker = dict(
                    color = 'rgba(0,0,0,0.2)',
                    size = 2
                ),
                name = 'Data', 
                hoverinfo='skip'
            )
        )
        
    traces.append(
        go.Histogram(
            y = y,
            xaxis = 'x2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            name = 'Histogram',
            orientation='h'
        )
    )
    traces.append(
        go.Histogram(
            x = x,
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ), 
            name = 'Histogram'
        )
    )
    layout = go.Layout(
        autosize = True,
        width = width, 
        height = height, 
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        bargap = 0,
        showlegend = False,
        margin = dict(
            l = 10, 
            r = 10,
            t = 10, 
            b = 10
        )
    )
    
    g = go.FigureWidget(data = traces, layout = layout)
    
    def response(change):
        x = df[x_dropdown.value].values
        y = df[y_dropdown.value].values
        
        update_range = 4 if scatter else 3
        yhist        = 1 if scatter else 2
        
        with g.batch_update():
            for i in range(0, update_range):
                g.data[i].x = x
                g.data[i].y = y
                g.layout.xaxis.title = x_dropdown.value
                g.layout.yaxis.title = y_dropdown.value
            
            
    x_dropdown.observe(response, names="value")
    y_dropdown.observe(response, names="value")
    
    Title(title = "Interactive Density Plot", description = "Select values for X and Y axis to view varying densities.")
    display(widgets.VBox([container, g]))
    
    return

