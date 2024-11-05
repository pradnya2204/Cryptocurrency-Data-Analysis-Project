import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import ruptures as rpt

# Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def app():
    st.title("Cryptocurrency Forecasting")

    crypto_symbol = st.text_input("Enter a cryptocurrency symbol (e.g., BTC, ETH):")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if crypto_symbol and start_date and end_date:
        try:
            # Fetch data using yfinance
            crypto_data = yf.download(f"{crypto_symbol}-USD", start=start_date, end=end_date)

            if not crypto_data.empty:
                # Display historical data
                st.subheader("Cryptocurrency Historical Data:")
                st.write(crypto_data[['High', 'Low', 'Open', 'Close', 'Volume']])

                # Line graph for Closing Price
                st.subheader("Closing Price vs Time")
                fig = px.line(crypto_data, x=crypto_data.index, y='Close', title=f"{crypto_symbol} Closing Price Over Time")
                fig.update_traces(line=dict(color='green'))
                fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
                st.plotly_chart(fig)

                # Moving Averages
                st.subheader("100-day and 200-day Moving Averages")
                crypto_data['100MA'] = crypto_data['Close'].rolling(window=100).mean()
                crypto_data['200MA'] = crypto_data['Close'].rolling(window=200).mean()

                fig_ma = px.line(crypto_data, x=crypto_data.index, y=['100MA', '200MA'],
                                 title=f"{crypto_symbol} Moving Averages Over Time")
                fig_ma.update_traces(line=dict(color='green'), selector=dict(name='100MA'))
                fig_ma.update_traces(line=dict(color='blue'), selector=dict(name='200MA'))
                fig_ma.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white',
                                     xaxis_title='Date', yaxis_title='Moving Average', xaxis_gridcolor='gray', yaxis_gridcolor='gray')
                st.plotly_chart(fig_ma)

                # Prepare data for LSTM
                X = crypto_data[['High', 'Low', 'Open', 'Volume']]
                y = crypto_data['Close']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Normalize data
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_train = scaler_X.fit_transform(X_train)
                y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
                X_test = scaler_X.transform(X_test)

                X_train = torch.FloatTensor(X_train).view(-1, 1, 4)
                y_train = torch.FloatTensor(y_train)

                # Initialize LSTM model
                model = LSTMModel(input_size=4, hidden_size=64, num_layers=2, output_size=1)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Train the model
                model.train()
                for epoch in range(50):
                    outputs = model(X_train)
                    optimizer.zero_grad()
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                # Make predictions
                model.eval()
                with torch.no_grad():
                    future_X = torch.FloatTensor(X_train[-1].view(1, 1, -1))
                    predicted_y = []
                    for _ in range(365):  # Predicting for the next 365 days
                        future_y = model(future_X)
                        future_X = torch.cat((future_X[:, :, 1:], future_y.view(1, 1, -1)), dim=2)
                        predicted_y.append(future_y.item())
                    predicted_y = scaler_y.inverse_transform(np.array(predicted_y).reshape(-1, 1))

                # Visualization for LSTM prediction
                st.subheader("Prediction")
                future_dates = pd.date_range(start=end_date, periods=365, freq='D')
                fig_pred = px.line(x=future_dates, y=predicted_y.flatten(),
                                   title=f"{crypto_symbol} Predicted Closing Price for the Next Year")
                fig_pred.update_traces(line=dict(color='green'))
                fig_pred.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white',
                                       xaxis_title='Date', yaxis_title='Predicted Price', xaxis_gridcolor='gray', yaxis_gridcolor='gray')
                st.plotly_chart(fig_pred)

                # PELT Change Point Detection
                crypto_close = crypto_data[['Close']]
                algo = rpt.Pelt(model="l1").fit(crypto_close)
                result = algo.predict(pen=1.5)

                # Visualization for Change Points
                plt.figure(figsize=(10, 6))
                plt.plot(crypto_data.index, crypto_close, label="Actual Data", color='blue')
                plt.plot(crypto_data.index[result], crypto_close.iloc[result], 'ro', markersize=8, label="Change Points", color='red')
                plt.title(f"{crypto_symbol} Price with PELT Change Points")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                st.pyplot(plt)

            else:
                st.write(f"No data available for {crypto_symbol} between {start_date} and {end_date}")
        except Exception as e:
            st.write(f"Error: {str(e)}")
    else:
        st.write("Please enter a cryptocurrency symbol, start date, and end date.")

if __name__ == '__main__':
    app()
