import json
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
from scipy.stats import norm
import dash_defer_js_import as dji


external_stylesheets = ['https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css']

external_scripts = ['https://code.jquery.com/jquery-3.2.1.slim.min.js',
                    'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js',
                    'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js']

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets = external_stylesheets, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],)

mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
'''

def exact_solution(r, sigma, Strike, MaxTime, MaxPrice, MinPrice, Option, price_steps = 100):

    option_price = np.zeros((price_steps,MaxTime))
    SigmaSquared = sigma**2
    StockPrice = np.linspace(MinPrice, MaxPrice, price_steps)
    Tau = np.linspace(0.01, 1, MaxTime)  # This is the transformed time space
    [Tau, S] = np.meshgrid(Tau, StockPrice)
    x = np.log(S/Strike) + (r - 0.5*SigmaSquared)*Tau
    d1 = (x + SigmaSquared * Tau) / (sigma * np.sqrt(Tau))
    d2 = (x) / (sigma * np.sqrt(Tau))

    if Option == 'Call':
        option_price = Strike*np.exp(x + 0.5*SigmaSquared*Tau)*norm.cdf(d1) - Strike*norm.cdf(d2)

    elif Option == 'Put':
        option_price = -Strike * np.exp(x + 0.5 * SigmaSquared * Tau) * norm.cdf(-d1) + Strike * norm.cdf(-d2)
    return option_price

axis_template = {
    "showbackground": True,
    "backgroundcolor": "rgb(241, 236, 236)",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

plot_layout = {
    "title": "",
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": False,
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1.2, "z": 1},
        "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
        "annotations": [],
    },
}

app.layout = html.Div(children=[
    html.Div([
    html.H1(children='The Black Scholes Equation',  className="header__title")], className='container'),
    html.Div([

    html.Div([
        html.Div( [html.Div([ dcc.Graph(id = "graph-with-slider")] , style={'leftmargin': 100, }, className="six columns")], className='graphContainer'),

        html.Div([

            html.H6(children='The Black Scholes Partial Differential Equation is:',
                    style={'textAlign': 'center'}),

            html.Div(id='static', children=["$$ V_{t} + rSV_{x} + 0.5\sigma S^2 V_{xx} - rV = 0 $$"]),

            html.H6(children='Change the parameters values to see the changes reflected in the manifold. ', style={'textAlign': 'center'}),

            dcc.RadioItems( id = "option_type",
                options=[
                    {'label': 'Call', 'value': 'Call'},
                    {'label': 'Put', 'value': 'Put'},],
                    value='Call',
                    labelStyle={'display': 'inline-block'},
                    inputStyle={"margin-right": "5px", "margin-left":"30px"} ,
                    style={"padding": "10px", "max-width": "800px", "margin": "auto", 'textAlign': 'center'}),

            html.H6(children='Interest Rate: r', style={'textAlign': 'center'}),

            html.Div(dcc.Slider( id='interest-slider', min=0, max=0.5, step=0.1, value=0.3, marks={
                                    0: '0',
                                    0.1: '0.1',
                                    0.2: '0.2',
                                    0.3: '0.3',
                                    0.4: '0.4',
                                    0.5: '0.5'}
                                ), className="pb-20"),

            html.H6(children='Volatility: sigma', style={'textAlign': 'center'}),

            html.Div(dcc.Slider( id='vol-slider', min=0, max=1, step=0.1, value=0.5, marks={
                                    0: '0',
                                    0.2: '0.2',
                                    0.4: '0.4',
                                    0.6: '0.6',
                                    0.8: '0.8',
                                    1: '1'}
                                 ), className="pb-20"),

            html.H6(children='Strike Price: k', style={'textAlign': 'center'}),

            html.Div(dcc.Slider(id='strike-slider',  min=5, max=15, step=1, value=10, marks={
                                    5: '5',
                                    7.5: '7.5',
                                    10: '10',
                                    12.5: '12.5',
                                    15: '15'}
                                ),className="pb-20"),
        ], className="six columns")], className="container")], className="row"), mathjax_script

    ])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('interest-slider', 'value'),
     Input('vol-slider', 'value'),
     Input('strike-slider', 'value'),
     Input('option_type', 'value')])

def update_figure(rho, vol, k, type):
    r= rho
    sigma = vol
    Strike = k
    MaxTime = 100
    MaxPrice = 20
    MinPrice = 1
    Option =  type

    z_data= exact_solution(r, sigma, Strike, MaxTime, MaxPrice, MinPrice, Option, price_steps = 100)

    fig = go.Figure(data=[go.Surface(z=z_data)],)

    camera = dict(
        up=dict(x=0, y=0, z=0.8),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.3, y=-1.3, z=1.3)
    )

    fig.update_layout(transition_duration=500, scene_camera=camera, autosize=False,
                  width=550, height=500,
                  margin=dict(l=100, r=25, b=65, t=50), xaxis_title="Stock Price", yaxis_title="Time to Expiration",)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
