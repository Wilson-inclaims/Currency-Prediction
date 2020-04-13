import yfinance as yf
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
knn = KNeighborsClassifier(n_neighbors=1)
#importing pairs from .txt file
PAIRS_FILE_NAME = 'pairs.txt'
pairs_file = open(PAIRS_FILE_NAME)
pairs = [line.strip()  for line in pairs_file.readlines()]
pairs_file.close()
scale = StandardScaler()


#converting currency pair name in string to format accepted by yfinance module and returning dowloaded data (maximum recorded period)
def dataDownload(pair):
    pair = pair[:3] + pair[4:] + "=X"
    print("pair: ", pair)
    return yf.download(pair, period = "max", group_by = "ticker")
    
    
#creating a drop-down list allowing user to select the currency pair to analyse and automaticly downloading data to analyse


def Analysis(pair):
    q = dataDownload(pair)
    df = pd.DataFrame(q)
    #print(df)
    open_prediction = predict_open(df)
    close_prediction = predict_close(df)
    high_prediction = predict_high(df)
    low_prediction = predict_low(df)
    print(open_prediction)
    print(close_prediction)
    print(high_prediction)
    print(low_prediction)
    info = "Open Prediction: {0}, Close Prediction: {1}, High Prediction: {2}, Low Prediction: {3}"
    messagebox.showinfo("Tomorrow Prediction",info.format(open_prediction,close_prediction,high_prediction,low_prediction))
def predict_open(df):
    dateList = list()
    openList = list()
    open1List = list()
    open2List = list()
    open3List = list()
    open4List = list()
    open5List = list()
    totalList = list()
    totalvalueList = list()
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
    tomorrow = dt.date.today() + dt.timedelta(days=1)
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
    test_df = pd.DataFrame(test_data, columns = ["Test"])
    #print(np.reshape(test_data,(1,len(test_data))))
    tomorrow_open = regr.predict(np.reshape(test_data,(1,len(test_data))))
    #print(tomorrow_open)
    return np.reshape(tomorrow_open,1)[0]
def predict_close(df):
    dateList = list()
    closeList = list()
    close1List = list()
    close2List = list()
    close3List = list()
    close4List = list()
    close5List = list()
    totalList = list()
    totalvalueList = list()
    date = df.reset_index()['Date']
    close = df[['Close']]
    for i in range(len(date)):
        closeList.append(close.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            close1List.append(0)
        else:
            close1List.append(close.iloc[i - 1][0])

        if i <= 1:
            close2List.append(0)
        else:
            close2List.append(close.iloc[i - 2][0])

        if i <= 2:
            close3List.append(0)
        else:
            close3List.append(close.iloc[i - 3][0])

        if i <= 3:
            close4List.append(0)
        else:
            close4List.append(close.iloc[i - 4][0])

        if i <= 4:
            close5List.append(0)
        else:
            close5List.append(close.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(closeList, close1List, close2List, close3List, close4List, close5List)),
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
                # print(location)
                totalvalueList[j].append(close.iloc[location][0])
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
    #print(df2)
    X = df2
    y = df1[["Close"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #print(regr.coef_)
    ####get predict data##############
    close1 = close.iloc[len(date) - 1][0]
    close2 = close.iloc[len(date) - 2][0]
    close3 = close.iloc[len(date) - 3][0]
    close4 = close.iloc[len(date) - 4][0]
    close5 = close.iloc[len(date) - 5][0]
    tomorrow = dt.date.today() + dt.timedelta(days=1)
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
            tomorrowValueList.append(close.iloc[location][0])
        else:
            tomorrowValueList.append(0)
    test_data = list()
    test_data.append(close1)
    test_data.append(close2)
    test_data.append(close3)
    test_data.append(close4)
    test_data.append(close5)
    test_data = test_data + tomorrowValueList
    test_df = pd.DataFrame(test_data, columns=["Test"])
    #print(np.reshape(test_data, (1, len(test_data))))
    tomorrow_close = regr.predict(np.reshape(test_data, (1, len(test_data))))
    #print(tomorrow_close)
    return np.reshape(tomorrow_close,1)[0]
def predict_high(df):
    dateList = list()
    highList = list()
    high1List = list()
    high2List = list()
    high3List = list()
    high4List = list()
    high5List = list()
    totalList = list()
    totalvalueList = list()
    date = df.reset_index()['Date']
    high = df[['High']]
    for i in range(len(date)):
        highList.append(high.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            high1List.append(0)
        else:
            high1List.append(high.iloc[i - 1][0])

        if i <= 1:
            high2List.append(0)
        else:
            high2List.append(high.iloc[i - 2][0])

        if i <= 2:
            high3List.append(0)
        else:
            high3List.append(high.iloc[i - 3][0])

        if i <= 3:
            high4List.append(0)
        else:
            high4List.append(high.iloc[i - 4][0])

        if i <= 4:
            high5List.append(0)
        else:
            high5List.append(high.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(highList, high1List, high2List, high3List, high4List, high5List)),
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
                # print(location)
                totalvalueList[j].append(high.iloc[location][0])
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
    #print(df2)
    X = df2
    y = df1[["High"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #print(regr.coef_)
    ####get predict data##############
    high1 = high.iloc[len(date) - 1][0]
    high2 = high.iloc[len(date) - 2][0]
    high3 = high.iloc[len(date) - 3][0]
    high4 = high.iloc[len(date) - 4][0]
    high5 = high.iloc[len(date) - 5][0]
    tomorrow = dt.date.today() + dt.timedelta(days=1)
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
            tomorrowValueList.append(high.iloc[location][0])
        else:
            tomorrowValueList.append(0)
    test_data = list()
    test_data.append(high1)
    test_data.append(high2)
    test_data.append(high3)
    test_data.append(high4)
    test_data.append(high5)
    test_data = test_data + tomorrowValueList
    test_df = pd.DataFrame(test_data, columns=["Test"])
    #print(np.reshape(test_data, (1, len(test_data))))
    tomorrow_high = regr.predict(np.reshape(test_data, (1, len(test_data))))
    #print(tomorrow_close)
    return  np.reshape(tomorrow_high,1)[0]
def predict_low(df):
    dateList = list()
    lowList = list()
    low1List = list()
    low2List = list()
    low3List = list()
    low4List = list()
    low5List = list()
    totalList = list()
    totalvalueList = list()
    date = df.reset_index()['Date']
    low = df[['Low']]
    for i in range(len(date)):
        lowList.append(low.iloc[i][0])
        dateList.append(date[i])
        if i == 0:
            low1List.append(0)
        else:
            low1List.append(low.iloc[i - 1][0])

        if i <= 1:
            low2List.append(0)
        else:
            low2List.append(low.iloc[i - 2][0])

        if i <= 2:
            low3List.append(0)
        else:
            low3List.append(low.iloc[i - 3][0])

        if i <= 3:
            low4List.append(0)
        else:
            low4List.append(low.iloc[i - 4][0])

        if i <= 4:
            low5List.append(0)
        else:
            low5List.append(low.iloc[i - 5][0])

    df1 = pd.DataFrame(list(zip(lowList, low1List, low2List, low3List, low4List, low5List)),
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
                # print(location)
                totalvalueList[j].append(low.iloc[location][0])
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
    #print(df2)
    X = df2
    y = df1[["Low"]]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #print(regr.coef_)
    ####get predict data##############
    low1 = low.iloc[len(date) - 1][0]
    low2 = low.iloc[len(date) - 2][0]
    low3 = low.iloc[len(date) - 3][0]
    low4 = low.iloc[len(date) - 4][0]
    low5 = low.iloc[len(date) - 5][0]
    tomorrow = dt.date.today() + dt.timedelta(days=1)
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
            tomorrowValueList.append(low.iloc[location][0])
        else:
            tomorrowValueList.append(0)
    test_data = list()
    test_data.append(low1)
    test_data.append(low2)
    test_data.append(low3)
    test_data.append(low4)
    test_data.append(low5)
    test_data = test_data + tomorrowValueList
    test_df = pd.DataFrame(test_data, columns=["Test"])
    #print(np.reshape(test_data, (1, len(test_data))))
    tomorrow_low = regr.predict(np.reshape(test_data, (1, len(test_data))))
    #print(tomorrow_close)
    return np.reshape(tomorrow_low,1)[0]

OPTIONS = pairs
master = Tk()
#master.geometry("220x150")
variable = StringVar(master)
variable.set(OPTIONS[14])
w = OptionMenu(master, variable, *OPTIONS)
w.pack()
button = Button(master, text="OK", command=(lambda: Analysis(variable.get())))
#button.place(x= 50,y = 50)
button.pack()
mainloop()