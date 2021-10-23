# Team-A

# Project : CyptoPredictor
An application that predicts the prices of cryptocurrencies and recommends trade/holding of the currency to  the consumer.

Features provided:
1) Show the correlation of Closing Prices for each of the crypto currencies 
to Understand the impact of one on the another in the market.
This analysis can also be externed to understand the correlation between Cryptocurrencies and the Stck Indices (S&P 500) of United States, Forex conversions with USD/EUR/INR

2) Prediction of Prices of Bitcoin on an unseen test data by the LSTM Model.
This analysis can be compared wiht the Ichimoku Cloud, Relative Strength Index (RSI), and Exponential Moving Averages (EMA) 

Steps taken towards the development of oyr solution

1. Data Collection : Obtained real-time data from Investing.com using the investpy library for Bitcoin, Dogecoin, Etherium andTether USDT.
2. Exploratory Data Analysis to Visualize the trend.
- Measuring Correlation of Closing Prices of Cryptocurrency using Pearson's Correlation Co-effiecient
- Percentage Increase in 7 months
- How many coins could have been bought for $2000?
- How much money could have been made in these 7 months?
3. Prediction using LSTM
- Data pre-processing to normalize all the values using MinMaxScalar function from Sklearn
- Building and training LSTM model for Prediction.
- Testing the model with new set of data. 
- Obtaining the predicitons of prices of 1 crypto currency at a time.

Execute the project with the following Steps:
```
pip install requirements.txt
```
```
streamlit run C:/Users/sites/PycharmProjects/Team-A(hackathon)/home.py
```
