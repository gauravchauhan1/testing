import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
import matplotlib.pyplot as plt
import investpy
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



# App title
st.markdown('''
# CRYPTO PREDICTOR
- For full analysis, observe the charts
**Credits**
- App built by TeamA : Error 404 not found
- Built in `Python` using `streamlit`, `pandas` and `datetime`
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2021,1,1))
end_date = st.sidebar.date_input("End date", datetime.date(2021,7,1))


data = investpy.get_crypto_historical_data(crypto='bitcoin',
                                           from_date=str(start_date.strftime('%d/%m/%Y')),
                                           to_date=str(end_date.strftime('%d/%m/%Y')))

data1 = investpy.get_crypto_historical_data(crypto='dogecoin',
                                           from_date=str(start_date.strftime('%d/%m/%Y')),
                                           to_date=str(end_date.strftime('%d/%m/%Y')))

data2 = investpy.get_crypto_historical_data(crypto='ethereum',
                                           from_date=str(start_date.strftime('%d/%m/%Y')),
                                           to_date=str(end_date.strftime('%d/%m/%Y')))

data3 = investpy.get_crypto_historical_data(crypto='tether',
                                           from_date=str(start_date.strftime('%d/%m/%Y')),
                                           to_date=str(end_date.strftime('%d/%m/%Y')))

# Retrieving tickers data
lis = ("FULL ANALYSIS","BITCOIN","DOGECOIN","ETHERUEM","TETHER-USDT")
ticker_list = pd.DataFrame(lis)
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol


training_data = data.drop(['Currency'], axis = 1)
if tickerSymbol=="FULL ANALYSIS":
    df = pd.DataFrame({'BTC': data.Close,
                   'DOG': data1.Close,
                   'ETH': data2.Close,
                   'USDT': data3.Close})
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()
    rspine = ax2.spines['right']
    rspine.set_position(('axes', 1.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    fig.subplots_adjust(right=0.7)

    #df['BTC'].plot(ax=ax1, style='b-')
    #df['DOG'].plot(ax=ax2, style='g-')
    df['BTC'].plot(ax=ax1, style='b-')
    df['DOG'].plot(ax=ax1, style='r-', secondary_y=True)
    df['USDT'].plot(ax=ax2, style='g-')

    # legend
    ax2.legend([ax1.get_lines()[0],
                ax1.right_ax.get_lines()[0],
                ax2.get_lines()[0]],
               ['BTC','DOG','USDT'])
    st.write("**Visualize Relative Changes of Closing Prices**")
    st.pyplot(fig)


    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, fmt = '.4f', mask=mask, center=0, square=True, linewidths=.5)
    st.write("**Effects of cryptocurrency on each other**")
    st.pyplot(f)

    df_return = df.apply(lambda x: x / x[0])
    df_return.plot(grid=True, figsize=(15, 10)).axhline(y = 1, color = "black", lw = 2)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("**Profit/loss in 7 months**")
    st.pyplot()


    df_perc = df_return.tail(1) * 100
    ax = sns.barplot(data=df_perc)
    st.write("**Market value **")
    st.write(df_perc)

    budget = 2000 # USD
    df_coins = budget/df.head(1)

    ax = sns.barplot(data=df_coins)
    st.write("**Potential gains/loss per 2000 dollar**")
    df_coins
    st.write("**Percentage Increase in 7 months**")
    st.pyplot()

    df_profit = df_return.tail(1) * budget

    ax = sns.barplot(data=df_profit)
    st.write("**Percentage Increase in 7 months**")
    df_profit
    st.write("**How much money could have been made?**")
    st.pyplot()
elif tickerSymbol == "BITCOIN":
    #MinMaxScaler is used to normalize the data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    Y_train = []
    training_data.shape[0]
    for i in range(60, training_data.shape[0]):
         X_train.append(training_data[i-60:i])
         Y_train.append(training_data[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    trained_model = tf.keras.models.load_model('team_A_model')

    part_60_days = data.tail(60)
    df= part_60_days.append(data, ignore_index = True)
    df = df.drop(['Currency'], axis = 1)
    inputs = scaler.transform(df)

    X_test = []
    Y_test = []
    for i in range (60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    Y_pred = trained_model.predict(X_test)
    scale = 1/5.18164146e-05
    Y_test = Y_test*scale
    Y_pred = Y_pred*scale
    plt.figure(figsize=(14,5))
    plt.plot(Y_test, color = 'red', label = 'Real Bitcoin Price')
    plt.plot(Y_pred, color = 'green', label = 'Predicted Bitcoin Price')
    plt.title('Bitcoin Price Prediction using LSTM')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.write("**Our model predictions for bitcoin for future investment**")
    st.pyplot()
elif tickerSymbol == "DOGECOIN":
    #MinMaxScaler is used to normalize the data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    Y_train = []
    training_data.shape[0]
    for i in range(60, training_data.shape[0]):
         X_train.append(training_data[i-60:i])
         Y_train.append(training_data[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    trained_model = tf.keras.models.load_model('team_A_model')

    part_60_days = data1.tail(60)
    df= part_60_days.append(data1, ignore_index = True)
    df = df.drop(['Currency'], axis = 1)
    inputs = scaler.transform(df)


    X_test = []
    Y_test = []
    for i in range (60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    Y_pred = trained_model.predict(X_test)
    scale = 1/5.18164146e-05
    Y_test = Y_test*scale
    Y_pred = Y_pred*scale
    plt.figure(figsize=(14,5))
    plt.plot(Y_test, color = 'red', label = 'Real Doigcoin Price')
    plt.plot(Y_pred, color = 'green', label = 'Predicted Dogcoin Price')
    plt.title('Dogcoin Price Prediction using LSTM')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.write("**Our model predictions for Dogecoin for future investment**")
    st.pyplot()
elif tickerSymbol == "ETHERUEM":
    data2 = investpy.get_crypto_historical_data(crypto='ethereum',
                                           from_date='01/01/2021',
                                           to_date='31/07/2021')

    training_data = data2.drop(['Currency'], axis = 1)

    #MinMaxScaler is used to normalize the data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    Y_train = []
    training_data.shape[0]
    for i in range(60, training_data.shape[0]):
         X_train.append(training_data[i-60:i])
         Y_train.append(training_data[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    trained_model = tf.keras.models.load_model('team_A_model')

    part_60_days = data2.tail(60)
    df= part_60_days.append(data2, ignore_index = True)
    df = df.drop(['Currency'], axis = 1)
    inputs = scaler.transform(df)

    X_test = []
    Y_test = []
    for i in range (60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    Y_pred = trained_model.predict(X_test)

    scale = 1/5.18164146e-05
    Y_test = Y_test*scale
    Y_pred = Y_pred*scale

    plt.figure(figsize=(14,5))
    plt.plot(Y_test, color = 'red', label = 'Real Ethereum Price')
    plt.plot(Y_pred, color = 'green', label = 'Predicted Ethereum Price')
    plt.title('Ethereum Price Prediction using LSTM')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.write("**Our model predictions for etheruem for future investment**")
    st.pyplot()
elif tickerSymbol == "TETHER-USDT":
    data3 = investpy.get_crypto_historical_data(crypto='tether',
                                           from_date='01/01/2021',
                                           to_date='31/07/2021')

    training_data = data3.drop(['Currency'], axis = 1)

    #MinMaxScaler is used to normalize the data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    Y_train = []
    training_data.shape[0]
    for i in range(60, training_data.shape[0]):
         X_train.append(training_data[i-60:i])
         Y_train.append(training_data[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    trained_model = tf.keras.models.load_model('team_A_model')

    part_60_days = data3.tail(60)
    df= part_60_days.append(data3, ignore_index = True)
    df = df.drop(['Currency'], axis = 1)
    inputs = scaler.transform(df)

    X_test = []
    Y_test = []
    for i in range (60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    Y_pred = trained_model.predict(X_test)

    scale = 1/5.18164146e-05
    Y_test = Y_test*scale
    Y_pred = Y_pred*scale

    plt.figure(figsize=(14,5))
    plt.plot(Y_test, color = 'red', label = 'Real Tether USDT Price')
    plt.plot(Y_pred, color = 'green', label = 'Predicted Tether USDT Price')
    plt.title('Tether USDT Price Prediction using LSTM')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.write("**Our model predictions for tether usdt for future investment**")
    st.pyplot()











