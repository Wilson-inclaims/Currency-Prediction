import yfinance as yf
import tkinter.ttk as ttk
import tkinter
from tkinter import *
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import Ridge
from datetime import datetime
import datetime as dt
from tkinter import messagebox
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
#import matplotlib.finance
import mpl_finance
from mpl_finance import candlestick_ohlc
knn = KNeighborsClassifier(n_neighbors=1)
#importing pairs from .txt file
PAIRS_FILE_NAME = 'pairs.txt'
pairs_file = open(PAIRS_FILE_NAME)
pairs = [line.strip()  for line in pairs_file.readlines()]
pairs_file.close()
scale = StandardScaler()
normalList = list()
tomorrowList = list()
tomorrowList_result = list()
tkinter.max = 0
tkinter.timeframe = ""
interval = ""#### 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d
def dataDownload(pair):
    pair = pair[:3] + pair[4:] + "=X"
    print("pair: ", pair)
    bt = yf.Ticker(pair)

    #return yf.download(pair, period = "60d",interval= "1h", group_by = "ticker")

    interval = tkinter.timeframe
    if interval != "1d":
      return bt.history(period="60d", interval= interval)
    else:
       return bt.history(period="max", interval=interval)

def Analysis(pair):
    tomorrow_datetime = ""
    open_total = list()
    close_total = list()
    high_total = list()
    low_total = list()
    tkinter.pair = variable.get()
    q = dataDownload(pair)

    df = pd.DataFrame(q)
    print(df)

    if tkinter.timeframe != "1d":
     date = df.reset_index()['Datetime']
     open_prediction = predict_open(df)
     close_prediction = predict_close(df)
     high_prediction = predict_high(df)
     low_prediction =  predict_low(df)
    else:
        date = df.reset_index()['Date']
        #print(len(date))
        open_prediction = predict_open_1d(df)
        close_prediction = predict_close_1d(df)
        high_prediction = predict_high_1d(df)
        low_prediction = predict_low_1d(df)
    #print(open_prediction)
    #print(close_prediction)
    #print(high_prediction)
    #print(low_prediction)
    openn = df[['Open']]
    close = df[['Close']]
    high = df[['High']]
    low = df[['Low']]


    normalList = calc_normal()
    today_year = date[len(date) - 1].strftime("%Y")
    #print(today_year)
    today_month = date[len(date) - 1].strftime("%m")
    today_day =  date[len(date) - 1].strftime("%d")
    length = len(date)

    for i in range(7):
        if int(date[length - i -1].strftime("%Y")) < int(today_year):
            break

        tomorrowList_result.append(date[length - i * tkinter.max -1])
        open_total.append(openn.iloc[length - i * tkinter.max -1][0])
        close_total.append(close.iloc[length - i * tkinter.max -1][0])
        high_total.append(high.iloc[length - i * tkinter.max -1][0])
        low_total.append(low.iloc[length - i * tkinter.max-1][0])
    mean_open = 0
    mean_close = 0
    mean_high = 0
    mean_low = 0

    for i in range(5):
            tomorrow = calc_tomorrow(int(today_year),int(today_month),int(today_day))

            open_total.append(open_prediction[i][0])
            close_total.append(close_prediction[i][0])
            high_total.append(high_prediction[i][0])
            low_total.append(low_prediction[i][0])

            today_year = tomorrow[0]
            today_month = tomorrow[1]
            today_day = tomorrow[2]

            tomorrow_datetime_str = today_year+ "-" +today_month +"-"+today_day+ " " + "00:00:00"
            tomorrow_datetime = datetime.strptime(tomorrow_datetime_str,"%Y-%m-%d %H:%M:%S")
            tomorrowList_result.append(tomorrow_datetime)



    with open('prediction_result.txt', 'w+') as filehandle:
        Big_openList = list()
        Big_closeList = list()
        Big_highList = list()
        Big_lowList = list()
        for  j in range(5):
            Big_openList = Big_openList + open_prediction[j]
            Big_closeList = Big_closeList + close_prediction[j]
            Big_highList = Big_highList + high_prediction[j]
            Big_lowList = Big_lowList + low_prediction[j]
        print(Big_openList)
        filehandle.write('DateTime\t')
        filehandle.write('Open\t')
        filehandle.write('Close\t')
        filehandle.write('High\t')
        filehandle.write('Low\n')
        for j in range(len(tomorrowList)):
            filehandle.write('%s\t' % str(tomorrowList[j]))
            filehandle.write('%s\t' % str(Big_openList[j]))
            filehandle.write('%s\t' % str(Big_closeList[j]))
            filehandle.write('%s\t' % str(Big_highList[j]))
            filehandle.write('%s\n' % str(Big_lowList[j]))


    print(tomorrowList_result)
    print(open_total)
    print(close_total)
    print(high_total)
    print(low_total)
    df1 = pd.DataFrame(list(zip(tomorrowList_result, open_total, close_total, high_total, low_total)),
                       columns=['DateTime', 'Open', 'Close', 'High', 'Low'])

    graph_data(df1)
    #print(open_prediction)
    #print(close_prediction)
    #print(high_prediction)
    #print(low_prediction)
def calc_tomorrow(year,month,day):
    tomorrowList = list()
    if day == 31:
        if month == 12:
           month =  1
           day = 1
           year = year+1
           tomorrowList.append(str(year))
           tomorrowList.append(str(month))
           tomorrowList.append(str(day))
           return tomorrowList
        else:
         month = month + 1
         day = 1
         tomorrowList.append(str(year))
         tomorrowList.append(str(month))
         tomorrowList.append(str(day))
         return tomorrowList
    if day == 30:
        if month == 4 or month ==6 or month ==9 or month == 11:
            month =month+1
            day = 1
            tomorrowList.append(str(year))
            tomorrowList.append(str(month))
            tomorrowList.append(str(day))
            return tomorrowList
    day = day +1
    tomorrowList.append(str(year))
    tomorrowList.append(str(month))
    tomorrowList.append(str(day))
    return tomorrowList

def graph_data(df):
    plt.style.use('ggplot')
    #df['DateTime'] = df['DateTime'].map(mdates.date2num)
    ohlc = df.loc[:, ['DateTime', 'Open', 'High', 'Low', 'Close']]
    ohlc['DateTime'] = pd.to_datetime(ohlc['DateTime'], utc=True)
    ohlc['DateTime'] = ohlc['DateTime'].apply(mdates.date2num)
    ohlc = ohlc.astype(float)

    # Creating Subplots
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8);

    # Customize graph.
    ##########################
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(tkinter.pair)

    # Format time.
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(15))
    ax.grid(True)
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.gcf().autofmt_xdate()  # Beautify the x-labels
    plt.autoscale(tight=True)
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()

    # Save graph to file.
    plt.savefig('mpl_finance-prediction.png')
def calc_normal():
    if tkinter.timeframe == "1m":
        frame = 1
    elif tkinter.timeframe == "2m":
        frame = 2
    elif tkinter.timeframe == "5m":
        frame = 5
    elif tkinter.timeframe == "15m":
        frame = 15
    elif tkinter.timeframe == "30m":
        frame = 30
    elif tkinter.timeframe == "60m":
        frame = 60
    elif tkinter.timeframe == "1d":
        frame = 24*60
    normal_min = 0
    normal_hour = 0
    normalList = list()
    tkinter.max = int(24*60/frame)
    for i in range(tkinter.max):
        if normal_min >= 60:
            normal_hour += 1
            normal_min = 0
        if normal_min < 10:
            normal_minstr = "0" + str(normal_min)
        else:
            normal_minstr = str(normal_min)
        if normal_hour < 10:
            normal_hourstr = "0" + str(normal_hour)
        else:
            normal_hourstr = str(normal_hour)
        normal = normal_hourstr + normal_minstr
        normalList.append(normal)
        normal_min = normal_min + frame
    return normalList



def predict_open(df):
    result_open = list()
    timeList = list()
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    open_prediction1 = list()
    open_prediction2 = list()
    open_prediction3 = list()
    open_prediction4 = list()
    open_prediction5 = list()
    totalvalueList = list()
    date_index = 0
    time_index = 0

    timecountList = list()

    open = df[['Open']]
    date = df.reset_index()['Datetime']
    real_date = ""
    first_md = date[0].strftime("%m") + date[0].strftime("%d")
    #print(first_md)
    ###making dateList(md:HM)
    for i in range(len(date)):
        dateDict = dict()

        openList.append(open.iloc[i][0])
        date_md = date[i].strftime("%m") + date[i].strftime("%d")

        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        #print(date_time)
        if first_md == date_md:
            timeList.append(date_time)
            if i == len(date) -1:
                timecountList.append(len(timeList))
                dateDict[first_md] = timeList
                dateList.append(dateDict)
        else:
            #print(len(timeList))
            timecountList.append(len(timeList))
            dateDict[first_md] = timeList
            first_md = date_md
            dateList.append(dateDict)
            timeList.clear()
            timeList.append(date_time)


    #print(dateList)
    ###get same data in the past
    for i in range(len(date)):
        total_index = 0
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        #print(date_md)
        #print(date_time)
        for j in range(len(dateList)):

               for key in dateList[j].keys():
                   #print(key)
                   real_date = key


               if date_md == real_date:
                    #print(real_date)
                    #print(j)
                    date_index = j
                    break

        #print(date_index)

        if date_index <= 0:
               open1List.append(0)
        else:
               for value in dateList[date_index - 1].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-1):
                             total_index = total_index + timecountList[t] - 1
                         total_index += time_index
                         open1List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open1List.append(0)

        if date_index <= 1:
               open2List.append(0)
        else:
               for value in dateList[date_index - 2].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-2):
                             total_index = total_index + timecountList[t] - 1
                         total_index += time_index
                         open2List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open2List.append(0)
        if date_index <= 2:
               open3List.append(0)
        else:
               for value in dateList[date_index - 3].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-3):
                             total_index = total_index + timecountList[t] - 1
                         total_index += time_index
                         open3List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open3List.append(0)

        if date_index <= 3:
               open4List.append(0)
        else:
               for value in dateList[date_index - 4].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-4):
                             total_index = total_index + timecountList[t] -1
                         total_index += time_index
                         open4List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open4List.append(0)

        if date_index <= 4:
               open5List.append(0)
        else:
               for value in dateList[date_index - 5].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-5):
                             total_index = total_index + timecountList[t] - 1
                         total_index += time_index
                         open5List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open5List.append(0)
    #print(openList)
    #print(open1List)
    #print(open2List)
    #print(open3List)
    #print(open4List)
    #print(open5List)
    #making dataframe
    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['Open', 'Open1', 'Open2', 'Open3', 'Open4', 'Open5'])
    #making test data
    test_open1 = list()
    test_open2 = list()
    test_open3 = list()
    test_open4 = list()
    test_open5 = list()

    open1_dateindex = len(dateList) - 1
    index = 0
    for i in range(len(dateList) - 1):
        index = index + timecountList[i]
    normalList = calc_normal()
    print(normalList)
    #print(dateList[len(timecountList) - 1])
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 1].values():
            if normalList[i] in value:
                open1_index = value.index(normalList[i])

                for j in range(len(timecountList) - 1):
                    total_index = total_index + timecountList[j] -1
                total_index += open1_index
                #print(total_index)
                test_open1.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open1.append(0)


        for value in dateList[len(timecountList) - 2].values():
            if normalList[i] in value:
                open2_index = value.index(normalList[i])
                for j in range(len(timecountList) - 2):
                    total_index = total_index + timecountList[j] - 1
                total_index += open2_index
                test_open2.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open2.append(0)


        for value in dateList[len(timecountList) - 3].values():
            if normalList[i] in value:
                open3_index = value.index(normalList[i])
                for j in range(len(timecountList) - 3):
                    total_index = total_index + timecountList[j] -1
                total_index += open3_index
                test_open3.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open3.append(0)


        for value in dateList[len(timecountList) - 4].values():
            if normalList[i] in value:
                open4_index = value.index(normalList[i])
                for j in range(len(timecountList) - 4):
                    total_index = total_index + timecountList[j] - 1
                total_index += open4_index
                test_open4.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open4.append(0)

        for value in dateList[len(timecountList) - 5].values():
            if normalList[i] in value:
                open5_index = value.index(normalList[i])
                for j in range(len(timecountList) - 5):
                    total_index = total_index + timecountList[j] -1
                total_index += open5_index
                test_open5.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open5.append(0)

    #print(test_open1)
    #print(test_open2)
    #print(test_open3)
    #print(test_open4)
    #print(test_open5)
    #prediction
    open_prediction1 = predict_opennextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5)
    result_open.append(open_prediction1)
    #print(open_prediction1)
    open_prediction2 = predict_opennextday(df1,open_prediction1,test_open1,test_open2,test_open3,test_open4)
    result_open.append(open_prediction2)
    #print(open_prediction2)
    open_prediction3 = predict_opennextday(df1, open_prediction1,open_prediction2,test_open1, test_open2, test_open3)
    result_open.append(open_prediction3)
    #print(open_prediction3)
    open_prediction4 = predict_opennextday(df1, open_prediction1, open_prediction2, open_prediction3 , test_open1, test_open2)
    result_open.append(open_prediction4)
    #print(open_prediction4)
    open_prediction5 = predict_opennextday(df1, open_prediction1, open_prediction2, open_prediction3, open_prediction4, test_open1)
    result_open.append(open_prediction5)
    #print(open_prediction5)
    return result_open
def predict_opennextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5):
    open_result = list()
    X = df1[["Open1","Open2","Open3","Open4","Open5"]]
    y = df1[["Open"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    for i in range(len(test_open1)):
        test_data = [[test_open1[i],test_open2[i],test_open3[i],test_open4[i],test_open5[i]]]
        prediction = regr.predict(test_data)
        open_result.append(np.reshape(prediction, 1)[0])
    print(open_result)
    return open_result

def predict_close(df):
    result_open = list()
    timeList = list()
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    open_prediction1 = list()
    open_prediction2 = list()
    open_prediction3 = list()
    open_prediction4 = list()
    open_prediction5 = list()
    totalvalueList = list()
    date_index = 0
    time_index = 0

    timecountList = list()

    open = df[['Close']]
    date = df.reset_index()['Datetime']
    real_date = ""
    first_md = date[0].strftime("%m") + date[0].strftime("%d")
    #print(first_md)

    for i in range(len(date)):
        dateDict = dict()

        openList.append(open.iloc[i][0])
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        if first_md == date_md:
            timeList.append(date_time)
            if i == len(date) -1:
                timecountList.append(len(timeList))
                dateDict[first_md] = timeList
                dateList.append(dateDict)
        else:
            #print(len(timeList))
            timecountList.append(len(timeList))
            dateDict[first_md] = timeList
            first_md = date_md
            dateList.append(dateDict)
            timeList.clear()
            timeList.append(date_time)


    #print(dateList)
    for i in range(len(date)):
        total_index = 0
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        #print(date_md)
        #print(date_time)
        for j in range(len(dateList)):

               for key in dateList[j].keys():
                   #print(key)
                   real_date = key


               if date_md == real_date:
                    #print(real_date)
                    #print(j)
                    date_index = j
                    break

        #print(date_index)

        if date_index <= 0:
               open1List.append(0)
        else:
               for value in dateList[date_index - 1].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-1):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open1List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open1List.append(0)

        if date_index <= 1:
               open2List.append(0)
        else:
               for value in dateList[date_index - 2].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-2):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open2List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open2List.append(0)
        if date_index <= 2:
               open3List.append(0)
        else:
               for value in dateList[date_index - 3].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-3):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open3List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open3List.append(0)

        if date_index <= 3:
               open4List.append(0)
        else:
               for value in dateList[date_index - 4].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-4):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open4List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open4List.append(0)

        if date_index <= 4:
               open5List.append(0)
        else:
               for value in dateList[date_index - 5].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-5):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open5List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open5List.append(0)
    #print(openList)
    #print(open1List)
    #print(open2List)
    #print(open3List)
    #print(open4List)
    #print(open5List)

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['Close', 'Close1', 'Close2', 'Close3', 'Close4', 'Close5'])

    test_open1 = list()
    test_open2 = list()
    test_open3 = list()
    test_open4 = list()
    test_open5 = list()

    open1_dateindex = len(dateList) - 1
    index = 0
    for i in range(len(dateList) - 1):
        index = index + timecountList[i]

    normalList = calc_normal()
    #print(normalList)
    #print(dateList[len(timecountList) - 1])
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 1].values():
            if normalList[i] in value:
                open1_index = value.index(normalList[i])

                for j in range(len(timecountList) - 1):
                    total_index = total_index + timecountList[j]
                total_index += open1_index
                #print(total_index)
                test_open1.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open1.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 2].values():
            if normalList[i] in value:
                open2_index = value.index(normalList[i])
                for j in range(len(timecountList) - 2):
                    total_index = total_index + timecountList[j]
                total_index += open2_index
                test_open2.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open2.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 3].values():
            if normalList[i] in value:
                open3_index = value.index(normalList[i])
                for j in range(len(timecountList) - 3):
                    total_index = total_index + timecountList[j]
                total_index += open3_index
                test_open3.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open3.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 4].values():
            if normalList[i] in value:
                open4_index = value.index(normalList[i])
                for j in range(len(timecountList) - 4):
                    total_index = total_index + timecountList[j]
                total_index += open4_index
                test_open4.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open4.append(0)
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 5].values():
            if normalList[i] in value:
                open5_index = value.index(normalList[i])
                for j in range(len(timecountList) - 5):
                    total_index = total_index + timecountList[j]
                total_index += open5_index
                test_open5.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open5.append(0)

    #print(test_open1)
    #print(test_open2)
    #print(test_open3)
    #print(test_open4)
    #print(test_open5)
    open_prediction1 = predict_closenextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5)
    result_open.append(open_prediction1)
    #print(open_prediction1)
    open_prediction2 = predict_closenextday(df1,open_prediction1,test_open1,test_open2,test_open3,test_open4)
    result_open.append(open_prediction2)
    #print(open_prediction2)
    open_prediction3 = predict_closenextday(df1, open_prediction1,open_prediction2,test_open1, test_open2, test_open3)
    result_open.append(open_prediction3)
    #print(open_prediction3)
    open_prediction4 = predict_closenextday(df1, open_prediction1, open_prediction2, open_prediction3 , test_open1, test_open2)
    result_open.append(open_prediction4)
    #print(open_prediction4)
    open_prediction5 = predict_closenextday(df1, open_prediction1, open_prediction2, open_prediction3, open_prediction4, test_open1)
    result_open.append(open_prediction5)
    #print(open_prediction5)
    return result_open
def predict_closenextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5):
    open_result = list()
    X = df1[["Close1","Close2","Close3","Close4","Close5"]]
    y = df1[["Close"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    for i in range(len(test_open1)):
        test_data = [[test_open1[i],test_open2[i],test_open3[i],test_open4[i],test_open5[i]]]
        prediction = regr.predict(test_data)
        open_result.append(np.reshape(prediction, 1)[0])
    return open_result

def predict_high(df):
    result_open = list()
    timeList = list()
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    open_prediction1 = list()
    open_prediction2 = list()
    open_prediction3 = list()
    open_prediction4 = list()
    open_prediction5 = list()
    totalvalueList = list()
    date_index = 0
    time_index = 0

    timecountList = list()

    open = df[['High']]
    date = df.reset_index()['Datetime']
    real_date = ""
    first_md = date[0].strftime("%m") + date[0].strftime("%d")
    #print(first_md)

    for i in range(len(date)):
        dateDict = dict()

        openList.append(open.iloc[i][0])
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        if first_md == date_md:
            timeList.append(date_time)
            if i == len(date) -1:
                timecountList.append(len(timeList))
                dateDict[first_md] = timeList
                dateList.append(dateDict)
        else:
            #print(len(timeList))
            timecountList.append(len(timeList))
            dateDict[first_md] = timeList
            first_md = date_md
            dateList.append(dateDict)
            timeList.clear()
            timeList.append(date_time)


    #print(dateList)
    for i in range(len(date)):
        total_index = 0
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        #print(date_md)
        #print(date_time)
        for j in range(len(dateList)):

               for key in dateList[j].keys():
                   #print(key)
                   real_date = key


               if date_md == real_date:
                    #print(real_date)
                    #print(j)
                    date_index = j
                    break

        #print(date_index)

        if date_index <= 0:
               open1List.append(0)
        else:
               for value in dateList[date_index - 1].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-1):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open1List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open1List.append(0)

        if date_index <= 1:
               open2List.append(0)
        else:
               for value in dateList[date_index - 2].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-2):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open2List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open2List.append(0)
        if date_index <= 2:
               open3List.append(0)
        else:
               for value in dateList[date_index - 3].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-3):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open3List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open3List.append(0)

        if date_index <= 3:
               open4List.append(0)
        else:
               for value in dateList[date_index - 4].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-4):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open4List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open4List.append(0)

        if date_index <= 4:
               open5List.append(0)
        else:
               for value in dateList[date_index - 5].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-5):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open5List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open5List.append(0)
    #print(openList)
    ##print(open1List)
    #print(open2List)
    #print(open3List)
    #print(open4List)
    #print(open5List)

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['High', 'High1', 'High2', 'High3', 'High4', 'High5'])

    test_open1 = list()
    test_open2 = list()
    test_open3 = list()
    test_open4 = list()
    test_open5 = list()

    open1_dateindex = len(dateList) - 1
    index = 0
    for i in range(len(dateList) - 1):
        index = index + timecountList[i]
    normalList = calc_normal()
    #print(normalList)
    #print(dateList[len(timecountList) - 1])
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 1].values():
            if normalList[i] in value:
                open1_index = value.index(normalList[i])

                for j in range(len(timecountList) - 1):
                    total_index = total_index + timecountList[j]
                total_index += open1_index
                #print(total_index)
                test_open1.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open1.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 2].values():
            if normalList[i] in value:
                open2_index = value.index(normalList[i])
                for j in range(len(timecountList) - 2):
                    total_index = total_index + timecountList[j]
                total_index += open2_index
                test_open2.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open2.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 3].values():
            if normalList[i] in value:
                open3_index = value.index(normalList[i])
                for j in range(len(timecountList) - 3):
                    total_index = total_index + timecountList[j]
                total_index += open3_index
                test_open3.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open3.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 4].values():
            if normalList[i] in value:
                open4_index = value.index(normalList[i])
                for j in range(len(timecountList) - 4):
                    total_index = total_index + timecountList[j]
                total_index += open4_index
                test_open4.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open4.append(0)
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 5].values():
            if normalList[i] in value:
                open5_index = value.index(normalList[i])
                for j in range(len(timecountList) - 5):
                    total_index = total_index + timecountList[j]
                total_index += open5_index
                test_open5.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open5.append(0)

    #print(test_open1)
    #print(test_open2)
    #print(test_open3)
    #print(test_open4)
    #print(test_open5)
    open_prediction1 = predict_highnextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5)
    result_open.append(open_prediction1)
    #print(open_prediction1)
    open_prediction2 = predict_highnextday(df1,open_prediction1,test_open1,test_open2,test_open3,test_open4)
    result_open.append(open_prediction2)
    #print(open_prediction2)
    open_prediction3 = predict_highnextday(df1, open_prediction1,open_prediction2,test_open1, test_open2, test_open3)
    result_open.append(open_prediction3)
    #print(open_prediction3)
    open_prediction4 = predict_highnextday(df1, open_prediction1, open_prediction2, open_prediction3 , test_open1, test_open2)
    result_open.append(open_prediction4)
    #print(open_prediction4)
    open_prediction5 = predict_highnextday(df1, open_prediction1, open_prediction2, open_prediction3, open_prediction4, test_open1)
    result_open.append(open_prediction5)
    #print(open_prediction5)
    return result_open
def predict_highnextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5):
    open_result = list()
    X = df1[["High1","High2","High3","High4","High5"]]
    y = df1[["High"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    for i in range(len(test_open1)):
        test_data = [[test_open1[i],test_open2[i],test_open3[i],test_open4[i],test_open5[i]]]
        prediction = regr.predict(test_data)
        open_result.append(np.reshape(prediction, 1)[0])
    return open_result

def predict_low(df):
    result_open = list()
    timeList = list()
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    open_prediction1 = list()
    open_prediction2 = list()
    open_prediction3 = list()
    open_prediction4 = list()
    open_prediction5 = list()
    totalvalueList = list()
    date_index = 0
    time_index = 0

    timecountList = list()

    open = df[['Low']]
    date = df.reset_index()['Datetime']
    real_date = ""
    first_md = date[0].strftime("%m") + date[0].strftime("%d")
    #print(first_md)

    for i in range(len(date)):
        dateDict = dict()

        openList.append(open.iloc[i][0])
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        if first_md == date_md:
            timeList.append(date_time)
            if i == len(date) -1:
                timecountList.append(len(timeList))
                dateDict[first_md] = timeList
                dateList.append(dateDict)
        else:
            #print(len(timeList))
            timecountList.append(len(timeList))
            dateDict[first_md] = timeList
            first_md = date_md
            dateList.append(dateDict)
            timeList.clear()
            timeList.append(date_time)


    #print(dateList)
    for i in range(len(date)):
        total_index = 0
        date_md = date[i].strftime("%m") + date[i].strftime("%d")
        date_time = date[i].strftime("%H") + date[i].strftime("%M")
        #print(date_md)
        #print(date_time)
        for j in range(len(dateList)):

               for key in dateList[j].keys():
                   #print(key)
                   real_date = key


               if date_md == real_date:
                    #print(real_date)
                    #print(j)
                    date_index = j
                    break

        #print(date_index)

        if date_index <= 0:
               open1List.append(0)
        else:
               for value in dateList[date_index - 1].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-1):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open1List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open1List.append(0)

        if date_index <= 1:
               open2List.append(0)
        else:
               for value in dateList[date_index - 2].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-2):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open2List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open2List.append(0)
        if date_index <= 2:
               open3List.append(0)
        else:
               for value in dateList[date_index - 3].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-3):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open3List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open3List.append(0)

        if date_index <= 3:
               open4List.append(0)
        else:
               for value in dateList[date_index - 4].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-4):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open4List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open4List.append(0)

        if date_index <= 4:
               open5List.append(0)
        else:
               for value in dateList[date_index - 5].values():
                   #print(value)

                     if date_time in value:
                         time_index = value.index(date_time)
                         for t in range(date_index-5):
                             total_index = total_index + timecountList[t]
                         total_index += time_index
                         open5List.append(open.iloc[total_index][0])
                         total_index = 0
                     else:
                         open5List.append(0)
    #print(openList)
    #print(open1List)
    #print(open2List)
    #print(open3List)
    #print(open4List)
    #print(open5List)

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['Low', 'Low1', 'Low2', 'Low3', 'Low4', 'Low5'])

    test_open1 = list()
    test_open2 = list()
    test_open3 = list()
    test_open4 = list()
    test_open5 = list()

    open1_dateindex = len(dateList) - 1
    index = 0
    for i in range(len(dateList) - 1):
        index = index + timecountList[i]
    normalList = calc_normal()
    #print(normalList)
    #print(dateList[len(timecountList) - 1])
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 1].values():
            if normalList[i] in value:
                open1_index = value.index(normalList[i])

                for j in range(len(timecountList) - 1):
                    total_index = total_index + timecountList[j]
                total_index += open1_index
                #print(total_index)
                test_open1.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open1.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 2].values():
            if normalList[i] in value:
                open2_index = value.index(normalList[i])
                for j in range(len(timecountList) - 2):
                    total_index = total_index + timecountList[j]
                total_index += open2_index
                test_open2.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open2.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 3].values():
            if normalList[i] in value:
                open3_index = value.index(normalList[i])
                for j in range(len(timecountList) - 3):
                    total_index = total_index + timecountList[j]
                total_index += open3_index
                test_open3.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open3.append(0)

    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 4].values():
            if normalList[i] in value:
                open4_index = value.index(normalList[i])
                for j in range(len(timecountList) - 4):
                    total_index = total_index + timecountList[j]
                total_index += open4_index
                test_open4.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open4.append(0)
    for i in range(len(normalList)):
        for value in dateList[len(timecountList) - 5].values():
            if normalList[i] in value:
                open5_index = value.index(normalList[i])
                for j in range(len(timecountList) - 5):
                    total_index = total_index + timecountList[j]
                total_index += open5_index
                test_open5.append(open.iloc[total_index][0])
                total_index = 0
            else:
                test_open5.append(0)

    #print(test_open1)
    #print(test_open2)
    #print(test_open3)
    #print(test_open4)
    #print(test_open5)
    open_prediction1 = predict_lownextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5)
    result_open.append(open_prediction1)
    #print(open_prediction1)
    open_prediction2 = predict_lownextday(df1,open_prediction1,test_open1,test_open2,test_open3,test_open4)
    result_open.append(open_prediction2)
    #print(open_prediction2)
    open_prediction3 = predict_lownextday(df1, open_prediction1,open_prediction2,test_open1, test_open2, test_open3)
    result_open.append(open_prediction3)
    #print(open_prediction3)
    open_prediction4 = predict_lownextday(df1, open_prediction1, open_prediction2, open_prediction3 , test_open1, test_open2)
    result_open.append(open_prediction4)
    #print(open_prediction4)
    open_prediction5 = predict_lownextday(df1, open_prediction1, open_prediction2, open_prediction3, open_prediction4, test_open1)
    result_open.append(open_prediction5)
    #print(open_prediction5)
    return result_open
def predict_lownextday(df1,test_open1,test_open2,test_open3,test_open4,test_open5):
    open_result = list()
    X = df1[["Low1","Low2","Low3","Low4","Low5"]]
    y = df1[["Low"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    for i in range(len(test_open1)):
        test_data = [[test_open1[i],test_open2[i],test_open3[i],test_open4[i],test_open5[i]]]
        prediction = regr.predict(test_data)
        open_result.append(np.reshape(prediction, 1)[0])
    return open_result
def predict_open_1d(df):
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    totalvalueList = list()
    resultList = list()
    date = df.reset_index()['Date']
    open = df[['Open']]
    for i in range(len(date)):
        openList.append(open.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            open1List.append(0)
        else:
            open1List.append(open.iloc[i - 1][0])

        if i <= 1:
            open2List.append(0)
        else:
            open2List.append(open.iloc[i - 2][0])

        if i <= 2:
            open3List.append(0)
        else:
            open3List.append(open.iloc[i - 3][0])

        if i <= 3:
            open4List.append(0)
        else:
            open4List.append(open.iloc[i - 4][0])

        if i <= 4:
            open5List.append(0)
        else:
            open5List.append(open.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                        columns=['Open', 'Open1', 'Open2', 'Open3', 'Open4', 'Open5'])


    firstyear = dateList[0].strftime("%y")
    lastyear  = dateList[len(date)-1].strftime("%y")
    listnum = int(lastyear) - int(firstyear) + 1
    for i in range(listnum):
        tmplist = list()
        valuelist = list()
        totalList.append(tmplist)
        totalvalueList.append(valuelist)
    for i in range(listnum):
        for j in range(len(date)):
            if int(dateList[j].strftime("%y")) == int(firstyear) + i:
                totalList[i].append(dateList[j].strftime("%m%d"))
    #print(totalList)
    for i in range(len(date)):

     thatday_md = dateList[i].strftime("%m%d")

     for j in range(listnum):
         location = 0
         if thatday_md in totalList[j]:
             index = totalList[j].index(thatday_md)
             for k in range(j):
                location += len(totalList[k]) - 1

             location += index
             #print(location)
             totalvalueList[j].append(open.iloc[location][0])
         else:
             totalvalueList[j].append(0)

    #print(totalvalueList[0])
    sameDate = "Opensame"
    sameDateList = list()
    for i in range(listnum):
         column = sameDate + str(i)
         df1[column] = totalvalueList[i]
    df2 = df1[["Open1"]]
    count = 1
    for col in df1.columns:
        if count > 2:
            df2[col] = df1[[col]]
        count += 1
    #print(df2)
    X = df2
    y = df1[["Open"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #print(regr.coef_)
    ####get predict data##############

    open1 = open.iloc[len(date) - 1][0]
    open2 = open.iloc[len(date) - 2][0]
    open3 = open.iloc[len(date) - 3][0]
    open4 = open.iloc[len(date) - 4][0]
    open5 = open.iloc[len(date) - 5][0]
    for i in range(5):
     tempList = list()

     if i == 1:
         open1 = resultList[0][0]
         open2 = open.iloc[len(date) - 1][0]
         open3 = open.iloc[len(date) - 2][0]
         open4 = open.iloc[len(date) - 3][0]
         open5 = open.iloc[len(date) - 4][0]
     if i == 2:
         open1 = resultList[1][0]
         open2 = resultList[0][0]
         open3 = open.iloc[len(date) - 1][0]
         open4 = open.iloc[len(date) - 2][0]
         open5 = open.iloc[len(date) - 3][0]
     if i == 3:
         open1 = resultList[2][0]
         open2 = resultList[1][0]
         open3 = resultList[0][0]
         open4 = open.iloc[len(date) - 1][0]
         open5 = open.iloc[len(date) - 2][0]
     if i == 4:
         open1 = resultList[3][0]
         open2 = resultList[2][0]
         open3 = resultList[1][0]
         open4 = resultList[0][0]
         open5 = open.iloc[len(date) - 1][0]
     #print(open1)
     #print(open2)
     #print(open3)
     #print(open4)
     #print(open5)
     tomorrow = dt.date.today() + dt.timedelta(days=i)
     tomorrow_md = tomorrow.strftime('%m%d')
     tomorrowValueList = list()
     for j in range(listnum):
        location = 0
        if tomorrow_md in totalList[j]:
            index = totalList[j].index(tomorrow_md)
            for k in range(j):
                location += len(totalList[k]) - 1

            location += index
            # print(location)
            tomorrowValueList.append(open.iloc[location][0])
        else:
            tomorrowValueList.append(0)
     test_data = list()
     test_data.append(open1)
     test_data.append(open2)
     test_data.append(open3)
     test_data.append(open4)
     test_data.append(open5)
     test_data = test_data + tomorrowValueList
     #print(test_data)
     test_df = pd.DataFrame(test_data, columns = ["Test"])
     #print(np.reshape(test_data,(1,len(test_data))))
     tomorrow_open = regr.predict(np.reshape(test_data,(1,len(test_data))))
     print(tomorrow_open)
     #print(tomorrow_open)
     tomorrow_result = np.reshape(tomorrow_open,1)[0]
     tempList.append(tomorrow_result)
     resultList.append(tempList)
    return resultList
def predict_close_1d(df):
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    totalvalueList = list()
    resultList = list()
    date = df.reset_index()['Date']
    open = df[['Close']]
    for i in range(len(date)):
        openList.append(open.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            open1List.append(0)
        else:
            open1List.append(open.iloc[i - 1][0])

        if i <= 1:
            open2List.append(0)
        else:
            open2List.append(open.iloc[i - 2][0])

        if i <= 2:
            open3List.append(0)
        else:
            open3List.append(open.iloc[i - 3][0])

        if i <= 3:
            open4List.append(0)
        else:
            open4List.append(open.iloc[i - 4][0])

        if i <= 4:
            open5List.append(0)
        else:
            open5List.append(open.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['Close', 'Close1', 'Close2', 'Close3', 'Close4', 'Close5'])

    firstyear = dateList[0].strftime("%y")
    lastyear = dateList[len(date) - 1].strftime("%y")
    listnum = int(lastyear) - int(firstyear) + 1
    for i in range(listnum):
        tmplist = list()
        valuelist = list()
        totalList.append(tmplist)
        totalvalueList.append(valuelist)
    for i in range(listnum):
        for j in range(len(date)):
            if int(dateList[j].strftime("%y")) == int(firstyear) + i:
                totalList[i].append(dateList[j].strftime("%m%d"))
    # print(totalList)
    for i in range(len(date)):

        thatday_md = dateList[i].strftime("%m%d")

        for j in range(listnum):
            location = 0
            if thatday_md in totalList[j]:
                index = totalList[j].index(thatday_md)
                for k in range(j):
                    location += len(totalList[k]) - 1

                location += index
                # print(location)
                totalvalueList[j].append(open.iloc[location][0])
            else:
                totalvalueList[j].append(0)

    # print(totalvalueList[0])
    sameDate = "Closesame"
    sameDateList = list()
    for i in range(listnum):
        column = sameDate + str(i)
        df1[column] = totalvalueList[i]
    df2 = df1[["Close1"]]
    count = 1
    for col in df1.columns:
        if count > 2:
            df2[col] = df1[[col]]
        count += 1
    # print(df2)
    X = df2
    y = df1[["Close"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    # print(regr.coef_)
    ####get predict data##############

    open1 = open.iloc[len(date) - 1][0]
    open2 = open.iloc[len(date) - 2][0]
    open3 = open.iloc[len(date) - 3][0]
    open4 = open.iloc[len(date) - 4][0]
    open5 = open.iloc[len(date) - 5][0]
    for i in range(5):
        tempList = list()

        if i == 1:
            open1 = resultList[0][0]
            open2 = open.iloc[len(date) - 1][0]
            open3 = open.iloc[len(date) - 2][0]
            open4 = open.iloc[len(date) - 3][0]
            open5 = open.iloc[len(date) - 4][0]
        if i == 2:
            open1 = resultList[1][0]
            open2 = resultList[0][0]
            open3 = open.iloc[len(date) - 1][0]
            open4 = open.iloc[len(date) - 2][0]
            open5 = open.iloc[len(date) - 3][0]
        if i == 3:
            open1 = resultList[2][0]
            open2 = resultList[1][0]
            open3 = resultList[0][0]
            open4 = open.iloc[len(date) - 1][0]
            open5 = open.iloc[len(date) - 2][0]
        if i == 4:
            open1 = resultList[3][0]
            open2 = resultList[2][0]
            open3 = resultList[1][0]
            open4 = resultList[0][0]
            open5 = open.iloc[len(date) - 1][0]

        tomorrow = dt.date.today() + dt.timedelta(days=i)
        tomorrow_md = tomorrow.strftime('%m%d')
        tomorrowValueList = list()
        for j in range(listnum):
            location = 0
            if tomorrow_md in totalList[j]:
                index = totalList[j].index(tomorrow_md)
                for k in range(j):
                    location += len(totalList[k]) - 1

                location += index
                # print(location)
                tomorrowValueList.append(open.iloc[location][0])
            else:
                tomorrowValueList.append(0)
        test_data = list()
        test_data.append(open1)
        test_data.append(open2)
        test_data.append(open3)
        test_data.append(open4)
        test_data.append(open5)
        test_data = test_data + tomorrowValueList
        test_df = pd.DataFrame(test_data, columns=["Test"])
        # print(np.reshape(test_data,(1,len(test_data))))
        tomorrow_open = regr.predict(np.reshape(test_data, (1, len(test_data))))
        print(tomorrow_open)
        # print(tomorrow_open)
        tomorrow_result = np.reshape(tomorrow_open, 1)[0]
        tempList.append(tomorrow_result)
        resultList.append(tempList)
    return resultList
def predict_high_1d(df):
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    totalvalueList = list()
    resultList = list()
    date = df.reset_index()['Date']
    open = df[['High']]
    for i in range(len(date)):
        openList.append(open.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            open1List.append(0)
        else:
            open1List.append(open.iloc[i - 1][0])

        if i <= 1:
            open2List.append(0)
        else:
            open2List.append(open.iloc[i - 2][0])

        if i <= 2:
            open3List.append(0)
        else:
            open3List.append(open.iloc[i - 3][0])

        if i <= 3:
            open4List.append(0)
        else:
            open4List.append(open.iloc[i - 4][0])

        if i <= 4:
            open5List.append(0)
        else:
            open5List.append(open.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['High', 'High1', 'High2', 'High3', 'High4', 'High5'])

    firstyear = dateList[0].strftime("%y")
    lastyear = dateList[len(date) - 1].strftime("%y")
    listnum = int(lastyear) - int(firstyear) + 1
    for i in range(listnum):
        tmplist = list()
        valuelist = list()
        totalList.append(tmplist)
        totalvalueList.append(valuelist)
    for i in range(listnum):
        for j in range(len(date)):
            if int(dateList[j].strftime("%y")) == int(firstyear) + i:
                totalList[i].append(dateList[j].strftime("%m%d"))
    # print(totalList)
    for i in range(len(date)):

        thatday_md = dateList[i].strftime("%m%d")

        for j in range(listnum):
            location = 0
            if thatday_md in totalList[j]:
                index = totalList[j].index(thatday_md)
                for k in range(j):
                    location += len(totalList[k]) - 1

                location += index
                # print(location)
                totalvalueList[j].append(open.iloc[location][0])
            else:
                totalvalueList[j].append(0)

    # print(totalvalueList[0])
    sameDate = "Highsame"
    sameDateList = list()
    for i in range(listnum):
        column = sameDate + str(i)
        df1[column] = totalvalueList[i]
    df2 = df1[["High1"]]
    count = 1
    for col in df1.columns:
        if count > 2:
            df2[col] = df1[[col]]
        count += 1
    # print(df2)
    X = df2
    y = df1[["High"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    # print(regr.coef_)
    ####get predict data##############

    open1 = open.iloc[len(date) - 1][0]
    open2 = open.iloc[len(date) - 2][0]
    open3 = open.iloc[len(date) - 3][0]
    open4 = open.iloc[len(date) - 4][0]
    open5 = open.iloc[len(date) - 5][0]
    for i in range(5):
        tempList = list()

        if i == 1:
            open1 = resultList[0][0]
            open2 = open.iloc[len(date) - 1][0]
            open3 = open.iloc[len(date) - 2][0]
            open4 = open.iloc[len(date) - 3][0]
            open5 = open.iloc[len(date) - 4][0]
        if i == 2:
            open1 = resultList[1][0]
            open2 = resultList[0][0]
            open3 = open.iloc[len(date) - 1][0]
            open4 = open.iloc[len(date) - 2][0]
            open5 = open.iloc[len(date) - 3][0]
        if i == 3:
            open1 = resultList[2][0]
            open2 = resultList[1][0]
            open3 = resultList[0][0]
            open4 = open.iloc[len(date) - 1][0]
            open5 = open.iloc[len(date) - 2][0]
        if i == 4:
            open1 = resultList[3][0]
            open2 = resultList[2][0]
            open3 = resultList[1][0]
            open4 = resultList[0][0]
            open5 = open.iloc[len(date) - 1][0]

        tomorrow = dt.date.today() + dt.timedelta(days=i)
        tomorrow_md = tomorrow.strftime('%m%d')
        tomorrowValueList = list()
        for j in range(listnum):
            location = 0
            if tomorrow_md in totalList[j]:
                index = totalList[j].index(tomorrow_md)
                for k in range(j):
                    location += len(totalList[k]) - 1

                location += index
                # print(location)
                tomorrowValueList.append(open.iloc[location][0])
            else:
                tomorrowValueList.append(0)
        test_data = list()
        test_data.append(open1)
        test_data.append(open2)
        test_data.append(open3)
        test_data.append(open4)
        test_data.append(open5)
        test_data = test_data + tomorrowValueList
        test_df = pd.DataFrame(test_data, columns=["Test"])
        # print(np.reshape(test_data,(1,len(test_data))))
        tomorrow_open = regr.predict(np.reshape(test_data, (1, len(test_data))))
        print(tomorrow_open)
        # print(tomorrow_open)
        tomorrow_result = np.reshape(tomorrow_open, 1)[0]
        tempList.append(tomorrow_result)
        resultList.append(tempList)
    return resultList
def predict_low_1d(df):
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    totalvalueList = list()
    resultList = list()
    date = df.reset_index()['Date']
    open = df[['Low']]
    for i in range(len(date)):
        openList.append(open.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            open1List.append(0)
        else:
            open1List.append(open.iloc[i - 1][0])

        if i <= 1:
            open2List.append(0)
        else:
            open2List.append(open.iloc[i - 2][0])

        if i <= 2:
            open3List.append(0)
        else:
            open3List.append(open.iloc[i - 3][0])

        if i <= 3:
            open4List.append(0)
        else:
            open4List.append(open.iloc[i - 4][0])

        if i <= 4:
            open5List.append(0)
        else:
            open5List.append(open.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(openList, open1List, open2List, open3List, open4List, open5List)),
                       columns=['Low', 'Low1', 'Low2', 'Low3', 'Low4', 'Low5'])

    firstyear = dateList[0].strftime("%y")
    lastyear = dateList[len(date) - 1].strftime("%y")
    listnum = int(lastyear) - int(firstyear) + 1
    for i in range(listnum):
        tmplist = list()
        valuelist = list()
        totalList.append(tmplist)
        totalvalueList.append(valuelist)
    for i in range(listnum):
        for j in range(len(date)):
            if int(dateList[j].strftime("%y")) == int(firstyear) + i:
                totalList[i].append(dateList[j].strftime("%m%d"))
    # print(totalList)
    for i in range(len(date)):

        thatday_md = dateList[i].strftime("%m%d")

        for j in range(listnum):
            location = 0
            if thatday_md in totalList[j]:
                index = totalList[j].index(thatday_md)
                for k in range(j):
                    location += len(totalList[k]) - 1

                location += index
                # print(location)
                totalvalueList[j].append(open.iloc[location][0])
            else:
                totalvalueList[j].append(0)

    # print(totalvalueList[0])
    sameDate = "Lowsame"
    sameDateList = list()
    for i in range(listnum):
        column = sameDate + str(i)
        df1[column] = totalvalueList[i]
    df2 = df1[["Low1"]]
    count = 1
    for col in df1.columns:
        if count > 2:
            df2[col] = df1[[col]]
        count += 1
    # print(df2)
    X = df2
    y = df1[["Low"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    # print(regr.coef_)
    ####get predict data##############

    open1 = open.iloc[len(date) - 1][0]
    open2 = open.iloc[len(date) - 2][0]
    open3 = open.iloc[len(date) - 3][0]
    open4 = open.iloc[len(date) - 4][0]
    open5 = open.iloc[len(date) - 5][0]
    for i in range(5):
        tempList = list()

        if i == 1:
            open1 = resultList[0][0]
            open2 = open.iloc[len(date) - 1][0]
            open3 = open.iloc[len(date) - 2][0]
            open4 = open.iloc[len(date) - 3][0]
            open5 = open.iloc[len(date) - 4][0]
        if i == 2:
            open1 = resultList[1][0]
            open2 = resultList[0][0]
            open3 = open.iloc[len(date) - 1][0]
            open4 = open.iloc[len(date) - 2][0]
            open5 = open.iloc[len(date) - 3][0]
        if i == 3:
            open1 = resultList[2][0]
            open2 = resultList[1][0]
            open3 = resultList[0][0]
            open4 = open.iloc[len(date) - 1][0]
            open5 = open.iloc[len(date) - 2][0]
        if i == 4:
            open1 = resultList[3][0]
            open2 = resultList[2][0]
            open3 = resultList[1][0]
            open4 = resultList[0][0]
            open5 = open.iloc[len(date) - 1][0]

        tomorrow = dt.date.today() + dt.timedelta(days=i)
        tomorrow_md = tomorrow.strftime('%m%d')
        tomorrowValueList = list()
        for j in range(listnum):
            location = 0
            if tomorrow_md in totalList[j]:
                index = totalList[j].index(tomorrow_md)
                for k in range(j):
                    location += len(totalList[k]) - 1

                location += index
                # print(location)
                tomorrowValueList.append(open.iloc[location][0])
            else:
                tomorrowValueList.append(0)
        test_data = list()
        test_data.append(open1)
        test_data.append(open2)
        test_data.append(open3)
        test_data.append(open4)
        test_data.append(open5)
        test_data = test_data + tomorrowValueList
        test_df = pd.DataFrame(test_data, columns=["Test"])
        # print(np.reshape(test_data,(1,len(test_data))))
        tomorrow_open = regr.predict(np.reshape(test_data, (1, len(test_data))))
        print(tomorrow_open)
        # print(tomorrow_open)
        tomorrow_result = np.reshape(tomorrow_open, 1)[0]
        tempList.append(tomorrow_result)
        resultList.append(tempList)
    return resultList
def on_select(event=None):
    if event:  # <-- this works only with bind because `command=` doesn't send event
        tkinter.timeframe =  event.widget.get()




OPTIONS = pairs
master = Tk()
#master.geometry("220x150")
variable = StringVar(master)#1m, 2m, 5m, 15m, 30m, 60m
cb = ttk.Combobox(master, values=("1m", "2m", "5m", "15m", "30m","1d"))
cb.set(" ")
cb.pack()
cb.bind('<<ComboboxSelected>>', on_select)
variable.set(OPTIONS[14])
w = OptionMenu(master, variable, *OPTIONS)
w.pack()


button = Button(master, text="OK", command=(lambda: Analysis(variable.get())))
#button.place(x= 50,y = 50)
button.pack()
mainloop()