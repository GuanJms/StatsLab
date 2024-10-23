import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.graph_objs as go

# Simulation parameters
np.random.seed(42)  # for reproducibility
n_simulations = 1000  # Number of simulations
alpha = 0.05  # Significance level


# Function to simulate data
def simulate_data(means, cov_matrix, n_samples, n_groups):
    data = []
    group_labels = []
    for i in range(n_groups):
        # Generate data for each group separately
        group_data = np.random.multivariate_normal(mean=[means[i]], cov=cov_matrix[i:i + 1, i:i + 1], size=n_samples)
        data.append(group_data)
        group_labels += [i] * n_samples
    return np.concatenate(data, axis=0), group_labels


# Function to perform ANOVA test
def run_anova(data, labels):
    df = pd.DataFrame({"data": data.flatten(), "group": labels})
    model = ols('data ~ C(group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_value = anova_table['PR(>F)'][0]
    return p_value


# Function to conduct power analysis for a given covariance
def power_analysis(n_simulations, means, cov_matrix, n_samples, n_groups, alpha):
    false_negative_count = 0
    for _ in range(n_simulations):
        data, labels = simulate_data(means, cov_matrix, n_samples, n_groups)
        p_value = run_anova(data, labels)

        # If there is a difference in group means but ANOVA doesn't detect it, increase false negatives
        if p_value > alpha:
            false_negative_count += 1

    power = false_negative_count / n_simulations
    return power


# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Power Simulation with Covariance Adjustment"),

    html.Label("Number of groups:"),
    dcc.Input(id='n-groups', type='number', value=3, min=2, max=10, step=1),

    html.Label("Group size:"),
    dcc.Input(id='group-size', type='number', value=50, min=10, max=1000, step=1),

    html.Label("Variance (diagonal):"),
    dcc.Slider(
        id='var-slider',
        min=0.1,
        max=2,
        step=0.1,
        value=1,
        marks={i: str(i) for i in np.arange(0.1, 2.1, 0.2)},
    ),

    html.Label("Covariance (off-diagonal):"),
    dcc.Slider(
        id='cov-slider',
        min=0,
        max=1,
        step=0.01,
        value=0.5,
        marks={i: str(i) for i in np.arange(0, 1.1, 0.1)},
    ),

    html.Button('Run Simulation', id='run-button'),
    dcc.Graph(id='power-graph'),

    html.Div(id='output-power', style={'margin-top': '20px'})
])


# Callback for running the simulation and updating the graph
@app.callback(
    [Output('power-graph', 'figure'), Output('output-power', 'children')],
    [Input('run-button', 'n_clicks'),
     Input('var-slider', 'value'),
     Input('cov-slider', 'value'),
     Input('n-groups', 'value'),
     Input('group-size', 'value')]
)
def update_output(n_clicks, variance, cov_value, n_groups, group_size):
    if n_clicks is None:
        return {}, "Adjust the sliders and run the simulation."

    # Means for the groups
    means = np.arange(n_groups)

    # Simulate over a range of covariance values from 0 to the selected value
    cov_range = np.linspace(0, cov_value, 10)

    power_values = []
    covariance_values = []

    for cov in cov_range:
        # Create a covariance matrix with the given variance and covariance
        cov_matrix = np.full((n_groups, n_groups), cov)
        np.fill_diagonal(cov_matrix, variance)  # Diagonal is the variance

        # Run the power analysis for this covariance
        power = power_analysis(n_simulations, means, cov_matrix, group_size, n_groups, alpha)

        power_values.append(power)
        covariance_values.append(cov)

    # Create a figure to show the power as a function of covariance
    fig = go.Figure(data=[
        go.Scatter(x=covariance_values, y=power_values, mode='lines+markers')
    ])
    fig.update_layout(
        title="Power vs Covariance",
        yaxis_title="Power (False Negative Rate)",
        xaxis_title="Covariance",
        yaxis_range=[0, 1]
    )

    return fig, f"Power simulation completed. Adjusted variance: {variance}. Covariance range: 0 to {cov_value}."


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
