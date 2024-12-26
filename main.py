import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import yfinance as yf
import dash_bootstrap_components as dbc

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    stock_data.reset_index(inplace=True)
    
    # Flatten the MultiIndex columns
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    
    print("Fetched columns:", stock_data.columns)
    return stock_data

def process_stock_data(stock_data, ticker):
    if stock_data is None:
        print("Error: No stock data provided to process.")
        return None

    # Check if 'Date_' column exists instead of 'Date'
    if 'Date_' not in stock_data.columns:
        print("Error: 'Date_' column not found in stock_data")
        return None

    stock_data.rename(columns={'Date_': 'Date'}, inplace=True)  # Standardize to 'Date'

    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    for col in numeric_columns:
        if col in stock_data.columns:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
        else:
            print(f"Warning: '{col}' column not found in stock_data")
    
    stock_data.rename(columns={
        'Close': f'Close_{ticker}',
        'High': f'High_{ticker}',
        'Low': f'Low_{ticker}',
        'Open': f'Open_{ticker}',
        'Volume': f'Volume_{ticker}'
    }, inplace=True)
    
    stock_data.dropna(inplace=True)
    return stock_data


def perform_eda(data, ticker):
    print(data.describe())
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data[f'Close_{ticker}'], label='Close Price')
    plt.title('Time Series of Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def perform_linear_regression(data, ticker):
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    X = data[['Date_ordinal']]
    y = data[f'Close_{ticker}']

    model = LinearRegression()
    model.fit(X, y)

    data['Predicted_Close'] = model.predict(X)

    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data[f'Close_{ticker}'], label='Actual Close Price')
    plt.plot(data['Date'], data['Predicted_Close'], label='Predicted Close Price', linestyle='--')
    plt.title('Linear Regression on Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def perform_linear_regression_with_prediction(data, ticker, days_to_predict=30):
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    X = data[['Date_ordinal']]
    y = data[f'Close_{ticker}']

    model = LinearRegression()
    model.fit(X, y)

    # Predict for historical data
    data['Predicted_Close'] = model.predict(X)

    # Predict for future data
    last_date = data['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_dates_ordinal = [date.toordinal() for date in future_dates]
    future_predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))

    future_data = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})

    # Combine historical and future predictions
    combined_data = pd.concat([data[['Date', 'Predicted_Close']], future_data], ignore_index=True)

    return combined_data


def perform_decision_tree_regression(data, ticker, days_to_predict=30):
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    X = data[['Date_ordinal']]
    y = data[f'Close_{ticker}']

    model = DecisionTreeRegressor()
    model.fit(X, y)

    # Predict for historical data
    data['Predicted_Close_DT'] = model.predict(X)

    # Predict for future data
    last_date = data['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_dates_ordinal = [date.toordinal() for date in future_dates]
    future_predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))

    future_data = pd.DataFrame({'Date': future_dates, 'Predicted_Close_DT': future_predictions})

    # Combine historical and future predictions
    combined_data = pd.concat([data[['Date', 'Predicted_Close_DT']], future_data], ignore_index=True)

    return combined_data

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("Stock Data Analysis", className="text-center my-4"))),
        dbc.Row(
            [
                dbc.Col(dcc.Input(id='stock-ticker', type='text', value='NVDA', placeholder='Enter stock ticker', className="form-control"), width=6),
                dbc.Col(html.Button('Submit', id='submit-button', n_clicks=0, className="btn btn-primary"), width=2),
            ],
            className="mb-4"
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='time-series-plot'), width=6),
                dbc.Col(dcc.Graph(id='correlation-matrix'), width=6),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='linear-regression-plot'), width=6),
                dbc.Col(dcc.Graph(id='decision-tree-plot'), width=6),
            ],
        ),
    ],
    fluid=True,
    className="bg-light p-4",
)

@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('correlation-matrix', 'figure'),
     Output('linear-regression-plot', 'figure'),
     Output('decision-tree-plot', 'figure')],
    [Input('submit-button', 'n_clicks'),
     Input('stock-ticker', 'value')]
)
def update_graphs(n_clicks, ticker):
    stock_data = fetch_stock_data(ticker)
    stock_data = process_stock_data(stock_data, ticker)

    if stock_data is None or f'Close_{ticker}' not in stock_data.columns:
        print("Error: Stock data could not be processed or column missing.")
        return {}, {}, {}, {}

    # Generate time-series plot
    fig1 = px.line(
        stock_data, 
        x='Date', 
        y=f'Close_{ticker}', 
        title=f'Time Series of {ticker} Close Price',
        template="plotly_white"
    )
    fig1.update_traces(line=dict(color='blue', width=2))
    fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # Generate correlation matrix
    corr_matrix = stock_data.corr()
    fig2 = px.imshow(
        corr_matrix, 
        text_auto=True, 
        title="Correlation Matrix", 
        template="plotly_white"
    )
    fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # Linear regression with full predictions
    combined_data_lr = perform_linear_regression_with_prediction(stock_data, ticker)
    fig3 = px.line(
        combined_data_lr, 
        x='Date', 
        y='Predicted_Close', 
        title='Linear Regression Predictions',
        template="plotly_white"
    )
    fig3.add_scatter(
        x=stock_data['Date'], 
        y=stock_data[f'Close_{ticker}'], 
        mode='lines', 
        name='Actual',
        line=dict(color='green')
    )

    # Decision tree regression with full predictions
    combined_data_dt = perform_decision_tree_regression(stock_data, ticker)
    fig4 = px.line(
        combined_data_dt, 
        x='Date', 
        y='Predicted_Close_DT', 
        title='Decision Tree Regression Predictions',
        template="plotly_white"
    )
    fig4.add_scatter(
        x=stock_data['Date'], 
        y=stock_data[f'Close_{ticker}'], 
        mode='lines', 
        name='Actual',
        line=dict(color='green')
    )

    return fig1, fig2, fig3, fig4



if __name__ == '__main__':
    app.run_server(debug=True)
