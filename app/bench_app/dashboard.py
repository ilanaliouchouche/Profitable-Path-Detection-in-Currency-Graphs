import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os

DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DIR, 'results.csv')

IMAGE_PATH = os.path.join(DIR, '/assets/psaclay.png')
print(IMAGE_PATH)
image = html.Img(src=IMAGE_PATH, style={'width': '200px',
                                                 'position': 'absolute',
                                                 'top': '20px',
                                                 'left': '20px'})

df = pd.read_csv(CSV_PATH)

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#2a2a2a', 'padding': '20px'},
                      children=[
    html.Div([image]),
    html.H1(children='Algorithm Performance Benchmark', style={
        'textAlign': 'center',
        'color': '#ffffff',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '2.5em',
        'marginBottom': '30px'
    }),

    html.P('By Ilan, David & Jiren',
           style={'fontStyle': 'italic',
                  'fontSize': '0.8em',
                  'textAlign': 'center',
                  'color': '#ffffff',
                  'marginTop': '-30px'}),

    html.Div([
        dcc.Graph(
            id='time-vs-nodes',
            style={'padding': '20px'}
        ),
    ], style={'marginBottom': '50px'}),

    html.Div([
        dcc.Graph(
            id='edges-vs-nodes',
            style={'display': 'inline-block', 'width': '48%'}
        ),
        dcc.Graph(
            id='visited-nodes-vs-nodes',
            style={'display': 'inline-block', 'width': '48%'}
        )
    ])
])


@app.callback(
    [Output('time-vs-nodes', 'figure'),
     Output('edges-vs-nodes', 'figure'),
     Output('visited-nodes-vs-nodes', 'figure')],
    [Input('time-vs-nodes', 'id')]
)
def update_graphs(_):
    fig_time = go.Figure()

    algorithms = df['Algorithm'].unique()
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        fig_time.add_trace(go.Scatter(
            x=algo_data['Number of Nodes'],
            y=algo_data['Avg Time (s)'],
            mode='lines',
            name=f'{algo} (Avg Time)',
            line=dict(width=2)
        ))

        fig_time.add_trace(go.Scatter(
            x=list(algo_data['Number of Nodes']) + list(
                algo_data['Number of Nodes'])[::-1],
            y=list(algo_data['CI Upper Time (s)']) + list(
                algo_data['CI Lower Time (s)'])[::-1],
            fill='toself',
            fillcolor='rgba(68, 68, 68, 0.9)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{algo} (CI)',
        ))

    fig_time.update_layout(
        title='Execution Time vs Number of Nodes',
        title_x=0.5,
        title_font_size=24,
        title_font_family='Arial',
        font=dict(family="Arial", size=14, color='#ffffff'),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2e2e2e',
        xaxis=dict(title='Number of Nodes (N)', color='#ffffff',
                   gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title='Execution Time (seconds)', color='#ffffff',
                   gridcolor='rgba(255, 255, 255, 0.1)'),
        legend=dict(font=dict(color='#ffffff')),
    )

    fig_edges = px.line(df, x='Number of Nodes', y='Avg Edges Traversed',
                        color='Algorithm',
                        title='Edges Traversed vs Number of Nodes',
                        template='plotly_dark')

    fig_edges.update_layout(
        title_font_size=24,
        title_x=0.5,
        title_font_family='Arial',
        font=dict(family="Arial", size=14, color='#ffffff'),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2e2e2e',
        xaxis=dict(title='Number of Nodes (N)', color='#ffffff',
                   gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title='Edges Traversed', color='#ffffff',
                   gridcolor='rgba(255, 255, 255, 0.1)'),
        legend=dict(font=dict(color='#ffffff'))
    )

    fig_nodes = px.line(df, x='Number of Nodes', y='Avg Nodes Visited',
                        color='Algorithm',
                        title='Nodes Visited vs Number of Nodes',
                        template='plotly_dark')

    fig_nodes.update_layout(
        title_font_size=24,
        title_x=0.5,
        title_font_family='Arial',
        font=dict(family="Arial", size=14, color='#ffffff'),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2e2e2e',
        xaxis=dict(title='Number of Nodes (N)', color='#ffffff',
                   gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title='Nodes Visited', color='#ffffff',
                   gridcolor='rgba(255, 255, 255, 0.1)'),
        legend=dict(font=dict(color='#ffffff'))
    )

    return fig_time, fig_edges, fig_nodes


if __name__ == '__main__':

    app.run_server(
        host="127.0.0.1",
        port="8050"
    )
