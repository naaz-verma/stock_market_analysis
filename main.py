import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
from constants import Models
from sklearn.preprocessing import MinMaxScaler
from doctest import Example
import profile
from re import A
import numpy as np
import pandas as pd
import streamlit as st


# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report


@st.cache(allow_output_mutation=True)
def load_model_pred():
    model_v = tf.keras.models.load_model(Models.pred_model)

    return model_v


ops_selected = st.sidebar.selectbox(
    "options",
    ("Home", "Algorithmic Trading", "Prediction")
)

if ops_selected == "Home":
    st.title("Home Page")
    st.write('Welcome to Stocklysis!')
    st.write("Choose a company stock ticker and analyse - ")
    st.write("1. Stock price prediction calculated from previous 10 years of data.")
    st.write('2. Algo-trading choices based on high and low values of stocks.')
    st.write(" ")
    st.write('Know best of your risks and profits in stocks using these predictive analyses.')


elif ops_selected == "Algorithmic Trading":
    start = '2010-01-01'
    end = '2020-12-31'
    b = st.text_input("company name")
    DF = data.DataReader(b, 'yahoo', start, end)
    st.write('Stocks data of 10 years of ' + b + ' company.')
    st.write(DF.head())
    st.write('Working on Adjusted Closing values of the stocks.')
    st.line_chart(DF['Adj Close'])
    sma30 = pd.DataFrame()
    sma30['Adj Close'] = DF['Adj Close'].rolling(window=30).mean()
    sma100 = pd.DataFrame()
    sma100['Adj Close'] = DF['Adj Close'].rolling(window=100).mean()
    DF2 = DF
    dt = pd.DataFrame(data=DF2['Adj Close'], columns=['Adj Close'])
    dt['SMA30'] = sma30
    dt['SMA100'] = sma100
    st.write('Comparison graph between actual stocks, SMA30 and SMA100')
    st.line_chart(dt)


    def buy_sell(dt):
        sigpricebuy = []
        sigpricesell = []
        flag = -1

        for i in range(len(dt)):
            if (dt['SMA30'][i] > dt['SMA100'][i]):
                if flag != 1:
                    sigpricebuy.append(dt['Adj Close'][i])
                    sigpricesell.append(np.nan)
                    flag = 1
                else:
                    sigpricebuy.append(np.nan)
                    sigpricesell.append(np.nan)
            elif (dt['SMA30'][i] < dt['SMA100'][i]):
                if flag != 0:
                    sigpricebuy.append(np.nan)
                    sigpricesell.append(dt['Adj Close'][i])
                    flag = 0
                else:
                    sigpricebuy.append(np.nan)
                    sigpricesell.append(np.nan)
            else:
                sigpricebuy.append(np.nan)
                sigpricesell.append(np.nan)

        return (sigpricebuy, sigpricesell)


    buy_sell = buy_sell(dt)
    dt['Buy_Signal_Price'] = buy_sell[0]
    dt['Sell_Signal_Price'] = buy_sell[1]
    st.write('Graph of final buy and sell timings.')

    st.line_chart(dt)


elif ops_selected == "Prediction":
    start = '2010-01-01'
    end = '2020-12-31'
    a = st.text_input("company name")
    df = data.DataReader(a, 'yahoo', start, end)
    st.write('Stocks data of 10 years of ' + a + ' company.')
    st.write(df.head())

    df = df.reset_index()

    df = df.drop(['Date', 'Adj Close'], axis=1)

    st.write("Graph of closing values of stocks of the company.")
    x = (df["Close"])
    st.line_chart(x)
    df1 = df
    ma100 = df1.Close.rolling(100).mean()
    df1['MA100'] = ma100
    df1.drop(['High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)

    st.write('Calculating teh MA100 values for prediction.')
    st.write(df1[100:105])
    st.write('Graph for values between closing and MA100.')
    st.line_chart(df1)
    st.write("Calculating teh MA200 values for prediction.")
    ma200 = df.Close.rolling(200).mean()
    df1['MA200'] = ma200
    st.write(df1[200:205])
    st.write('Graph for values between closing, MA100, MA200.')
    st.line_chart(df1)

    model_v = load_model_pred()

    train = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    test = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
    sc = MinMaxScaler(feature_range=(0, 1))
    train_arr = sc.fit_transform(train)

    x_train = []
    y_train = []

    for i in range(100, train_arr.shape[0]):
        x_train.append(train_arr[i - 100:i])
        y_train.append(train_arr[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    past_100_days = train.tail(100)
    final_df = past_100_days.append(test, ignore_index=True)
    input_data = sc.fit_transform(final_df)
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model_v.predict(x_test)
    st.write(y_predicted.shape)
    scale_factor = 1 / 0.00988704

    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    final_df = pd.DataFrame(data=y_test, columns=['Actual'])
    final_df['Predicted'] = y_predicted

    st.write("Prediction of closing values of stocks vs actual closing price")
    st.line_chart(final_df)
