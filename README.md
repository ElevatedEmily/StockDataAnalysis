# Stock Data Analysis Dashboard
The Stock Data Analysis Dashboard is a web application built using Python, Dash, and Plotly. It allows users to analyze stock data interactively, perform linear regression and decision tree regression for predictive modeling, and visualize the data through dynamic plots. The application fetches historical stock data from Yahoo Finance and provides insights such as time series trends, correlation matrices, and future predictions.
# Features
Interactive Dashboard: User-friendly interface built with Dash and Bootstrap for minimalistic design.
Historical Data Visualization: Time series plots of stock closing prices.
Predictive Analysis: Linear regression and decision tree regression models for forecasting future stock prices.
Correlation Insights: Visual correlation matrix to understand relationships between stock attributes.
Customizable Inputs: Fetch data for any stock ticker symbol.
# Setup
Prerequisites

Ensure the following are installed on your system:

  Python 3.8 or higher
  pip (Python package manager)

Installation

  Clone the repository:

    git clone https://github.com/your-username/stock-data-analysis-dashboard.git
    cd stock-data-analysis-dashboard

  Create a virtual environment (optional but recommended):

    python -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    venv\Scripts\activate     # For Windows

Install the required packages:

    pip install -r requirements.txt

Run the application:

    python main.py

Open your browser and navigate to:

    http://127.0.0.1:8050/
