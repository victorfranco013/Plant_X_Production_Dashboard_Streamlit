import streamlit as st
import pandas as pd
import numpy as np
import csv
import os
import psycopg2
import psycopg2.extras as extras
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import time
from pandas.tseries.offsets import DateOffset
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.graph_objs as go
import scipy
from statsmodels.graphics.gofplots import qqplot
#import timestring

def connect_db_sql(database="Morganton", user= "postgres", password= "United2018", host="localhost", port="5432"):
    conn = psycopg2.connect(database= database, user=user, password=password, host=host, port=port)
    return conn

def get_db_sql(conn, query='select * from "all_op200"'):
    df = pd.read_sql_query(query,con=conn, parse_dates=['date_time'])
    df.index = pd.to_datetime(df.date_time.dt.date)
    return df

@st.cache(allow_output_mutation=True)
def long_running_function_to_load_data(query = 'select * from "all_op200"', db='all_op200'):
    hours = range(0,24)
    shits = ['3rd']*7 + ['1st']*8 + ['2nd']*8 + ['3rd']
    shifts_dict = dict(zip(hours,shits))

    conn = connect_db_sql(database="Morganton", user= "postgres", password= "United2018", host="localhost", port="5432")
    df = get_db_sql(conn, query=query)

    df['time'] = df['date_time']
    df['date'] = pd.to_datetime(df.date_time.dt.date)
    df['year_month'] = df['date'].apply(lambda date: str(date.year) + '-' + str(date.month))
    df['year_week'] = df['date'].apply(lambda date: str(date.year) + '-' + str(date.week))
    df['week'] = df['date'].dt.week
    df['week'] = df['week'].apply(lambda x: '0' + str(x) if len(str(x))==1 else x).astype('str')
    df['year'] = df['date'].dt.year.astype('str')
    df['year_week'] = df['year'] + '-' + df['week']
    df['year_week_date'] = pd.to_datetime(df.week.astype(str)+ df.year.astype(str).add('-1') ,format='%V%G-%u')
    df['hour'] = df['date_time'].dt.hour
    df['shift'] = df['hour'].map(shifts_dict)
    df = df.rename(columns={'platform':'platform_2', 'platform_consolidated':'platform' })

    if db == 'all_op200' or db == 'all_op210':
        df['platform'] = df['platform'].replace('Jeep WK FA ', 'Jeep WK FA')
        df['platform'] = df['platform'].replace('98765432.1', 'Unknown')

    if db == 'all_op200':
        df['station#'] = df['station#'].replace(['Unspecified','Undefined NOK Station','1','998','Processed OK in OP050','7','6'], 'NOK Station: Unspecified')
        df['station'] = df['station#'].apply(lambda x: str(x).split(': ')[1])
        df['station_consolidated'] =  df['station'].replace(['OP185','OP180'], 'OP180_185')
        df['station_consolidated'] =  df['station'].replace(['Unspecified'], 'Unspecified_strut_line')
        df = df.rename(columns={'station':'station_2', 'station_consolidated':'station' })
        df['line'] = 'Strut Line'
        df['failed'] = True

    if db == 'all_op210':
        df['station'] = 'ALL_strut_line'
        df['line'] = 'Strut Line'
        df['failed'] = False

    #line = {'OP45A': 'Line 1 (OP45)', 'OP45B': 'Line 1 (OP45)', 'OP46A': 'Line 2 (OP46)', 'OP46B': 'Line 2 (OP46)'}
    line = {'OP45A': 'AS1', 'OP45B': 'AS1', 'OP46A': 'AS2', 'OP46B': 'AS2'}
    if db == 'hlt_op45_op46':
        df['line'] = df['station'].map(line)
        df['failed'] = df['hlt_failed_part']

    df['station_consolidated'] = df['station'].replace(['OP180','OP185'], 'OP180_185')
    df['station_consolidated'] = df['station_consolidated'].replace(['OP46A','OP46B'], 'OP46')
    df['station_consolidated'] = df['station_consolidated'].replace(['OP45A','OP45B'], 'OP45')
    df = df.rename(columns={'station':'station_II', 'station_consolidated':'station' })

    #platforms_to_analyze =['WL FA', 'Tesla FA Service', 'Tesla FA', 'Tesla RA', 'Ram DT FA', 'Ford FA', 'Jeep WK FA','Ram DS FA', 'HONDA FA']
    #df = df[df['platform'].isin(platforms_to_analyze)]
    #df = df[df['station'].isin(['OP180','OP185'])]
    return df

@st.cache(allow_output_mutation=True)
def Cpk_upper_only(mylist, usl):
    arr = np.array(mylist)
    arr = arr.ravel()
    sigma = np.std(arr)
    m = np.median(arr)
    Cpu = float(usl - m) / (3*sigma)
    return Cpu

@st.cache(allow_output_mutation=True)
def Cpk(mylist, lsl, usl):
    arr = np.array(mylist)
    arr = arr.ravel()
    sigma = np.std(arr)
    m = np.median(arr)
    Cpu = float(usl - m) / (3*sigma)
    Cpl = float(m - lsl) / (3*sigma)
    Cp = float(usl - lsl) / (6*sigma)
    return Cp, Cpu, Cpl

@st.cache(allow_output_mutation=True)
def plot_quantiles(x, low_bound, upp_bound):
    qqplot_data = qqplot(x, line='s').gca().lines
    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 700,
        'height': 600,
    })

    fig.add_hline(y=low_bound, line_width=1.0, line_dash="dash", line_color="red")
    fig.add_hline(y=upp_bound, line_width=1.0, line_dash="dash", line_color="red")

    return fig

@st.cache(allow_output_mutation=True)
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def plot_control_charts(df_filtered, values_dict, station_1_anal, low_bound, upp_bound, std):
    df_filtered = df_filtered.sort_values(by='time')
    df_filtered['observations'] = range(1, len(df_filtered)+1)
    df_filtered['OK/NOK'] = df_filtered[values_dict[station_1_anal]].apply(lambda x: 'OK' if (x > low_bound) & (x < upp_bound) else 'NOK')
    if station_1_anal in ['OP45-46 HLT','OP180-185 HLT']:
        fig = px.scatter(df_filtered, x='observations', y=values_dict[station_1_anal], color='OK/NOK', range_y=[low_bound - 0.1*std,upp_bound + 0.1*std],
                      opacity=0.5)
    else:
        fig = px.scatter(df_filtered, x='observations', y=values_dict[station_1_anal], color='OK/NOK', range_y=[low_bound - 0.5*std,upp_bound + 0.5*std],
                      opacity=0.5)

    fig.add_hline(y=low_bound, line_width=1.0, line_dash="dash", line_color="red")
    fig.add_hline(y=upp_bound, line_width=1.0, line_dash="dash", line_color="red")
    return fig

@st.cache(allow_output_mutation=True)
def cerate_displot(x, low_bound, upp_bound, bins, platform):
    hist_data= [x]
    group_labels = [platform]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=bins, show_hist = True, histnorm ='')

    fig.add_vline(x=low_bound, line_width=1.0, line_dash="dash", line_color="red")
    fig.add_vline(x=upp_bound, line_width=1.0, line_dash="dash", line_color="red")
    return fig

values_dict = {'OP45-46 HLT':'leak value [mbarl/s]',
               'OP180-185 HLT':'leak value [mbarl/s]',
               'OP90 torque [Nm]':'op090 torque [nm]',
               'OP190 ship height (mm)': 'op190 ship height (mm)'}

def app():
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #        Sidebar 1                                                                                                                                      /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    with st.sidebar:
        #date range
        st.write("### Station value to analyze?")
        station_1_anal = st.selectbox('',  ('OP45-46 HLT','OP180-185 HLT','OP90 torque [Nm]','OP190 ship height (mm)'))

        from_date = (datetime.date.today() - DateOffset(months=2)).date()
        st.subheader('Date range to load sql data?')
        c1, c2 = st.beta_columns(2)
        d1 = c1.date_input("From date", from_date, key='sql_d1')
        d2 = c2.date_input("Up to date", datetime.date.today(), key='sql_d2', min_value=d1+ DateOffset(days=2))

        #date range
        st.write("### Specific Date range to analyze")
        col1, col2 = st.beta_columns(2)
        day_from= col1.date_input("From date", from_date, min_value=d1)
        day_up_to= col2.date_input("Up to date", datetime.date.today(), min_value=day_from + DateOffset(days=2))

        st.markdown("""---""")


    if station_1_anal == 'OP45-46 HLT':
        # 45_46 Data-------------------
        query = "select * from hlt_op45_op46 where date_time between '%s' and '%s'" %(str(d1), str(d2) + ' 23:59:59.999')
        df= long_running_function_to_load_data(query = query ,db='hlt_op45_op46')
        df_dates = filter_dates(df, str(day_from), str(day_up_to))
    elif station_1_anal == 'OP180-185 HLT':
        # 180_185 Data-------------------
        query = "select * from hlt_op180_185_from_200_210 where date_time between '%s' and '%s'" %(str(d1), str(d2) + ' 23:59:59.999')
        df= long_running_function_to_load_data(query = query ,db='hlt_op180_185_from_200_210')
        df_dates = filter_dates(df, str(day_from), str(day_up_to))
    elif station_1_anal == 'OP90 torque [Nm]':
        # 180_185 Data-------------------
        query = "select * from hlt_op090_from_200_210 where date_time between '%s' and '%s'" %(str(d1), str(d2) + ' 23:59:59.999')
        df= long_running_function_to_load_data(query = query ,db='hlt_op090_from_200_210')
        df_dates = filter_dates(df, str(day_from), str(day_up_to))
    elif station_1_anal == 'OP190 ship height (mm)':
        # 180_185 Data-------------------
        query = "select * from all_op210 where date_time between '%s' and '%s'" %(str(d1), str(d2) + ' 23:59:59.999')
        df= long_running_function_to_load_data(query = query ,db='all_op210')
        df_dates = filter_dates(df, str(day_from), str(day_up_to))

    #Platforms and stations runn on that period
    platforms = list(df_dates['platform'].unique())


    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #        Sidebar 2                                                                                                                                      /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    with st.sidebar:
        st.subheader('Select Platform')
        platform = st.selectbox('',platforms)

    df_plat = df_dates[df_dates['platform'] == platform]
    stations  = list(df_plat['station_II'].unique())
    stations.append('ALL above')

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #        Sidebar 3                                                                                                                                      /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    with st.sidebar:
        if station_1_anal in ['OP45-46 HLT','OP180-185 HLT','OP90 torque [Nm]']:
            st.subheader('Select Specific Station')
            station = st.selectbox('', stations)
        else:
            station = 'ALL above'

    if station != 'ALL above':
        df_station = df_plat[df_plat['station_II'] == station]
    else :
        df_station = df_plat.copy()

    #removing zeroz and inserted values_dict
    df_station = df_station[df_station[values_dict[station_1_anal]] != 98765432.1]

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #        Sidebar 4                                                                                                                                      /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    with st.sidebar:
        st.subheader('Select Part Numbers')
        part_numbers_list =   list(df_station['conti part number'].unique())
        part_numbers = st.multiselect('',part_numbers_list, part_numbers_list[:], key='part # select')

    df_part_num = df_station[df_station['conti part number'].isin(part_numbers)]

    std = df_part_num[values_dict[station_1_anal]].std()
    low_percentile = np.percentile(df_part_num[values_dict[station_1_anal]], 3.0) - 2.25*std
    up_percentile = np.percentile(df_part_num[values_dict[station_1_anal]], 97.0) + 2.25*std

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #        Sidebar 5                                                                                                                                      /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    with st.sidebar:
        st.markdown("""---""")
        st.write('### Limit your data to...')
        c1, c2 = st.beta_columns(2)
        if station_1_anal in ['OP45-46 HLT','OP180-185 HLT']:
            low_filter = st.number_input('Analyze data above', value=0)
        else:
            low_filter = st.number_input('Analyze data above', value=low_percentile)
        st.write('Entered: ' + str(low_filter) )
        upp_filter = st.number_input('Analyze data below', value=up_percentile)
        st.write('Entered: ' + str(upp_filter) )
        df_filtered = df_part_num[ (df_part_num[values_dict[station_1_anal]] >= low_filter) & (df_part_num[values_dict[station_1_anal]] <= upp_filter)]
        st.markdown("""---""")

    #st st area
    st.title('Capability Study')
    st.write('## ' + platform +' - '+ values_dict[station_1_anal])


    #Boundaries to select0.
    st.markdown("""---""")
    st.write('### Enter OK/NOK boundaries')
    c1, c2 , c3= st.beta_columns(3)
    if station_1_anal in ['OP45-46 HLT','OP180-185 HLT']:
        low_bound = c1.number_input('Enter Lower Boundary', value=0)

        bins = c3.number_input('Histogram Bins',value= 10000)
    else:
        low_bound = c1.number_input('Enter Lower Boundary', value=low_percentile)
        bins = c3.number_input('Histogram Bins', value=0.05)

    upp_bound = c2.number_input('Enter Upper Boundary', value=upp_filter)

    c1.write('Entered: ' + str(low_bound) )
    c2.write('Entered: ' + str(upp_bound) )

    st.markdown("""---""")




    #Data
    x = df_filtered[values_dict[station_1_anal]]
    #CPK Calculation

    show_plots= st.checkbox('Show Capability Study', False)

    if show_plots:
        st.write("### Capability for the period")
        if station_1_anal in ['OP45-46 HLT','OP180-185 HLT']:
            cpk = Cpk_upper_only(x, upp_bound)
            st.write("Upper Limit Cpk: " + str(round(cpk,2)))

        else:
            c1,c2 = st.beta_columns(2)
            Cp, Cpu, Cpl = Cpk(x, low_bound, upp_bound)
            c1.write("Cp: " + str(round(Cp,2))   )
            c1.write("StDev: " + str(round(std,2))   )
            c2.write("Upper Limit Cpk: " + str(round(Cpu,2))   )
            c2.write("Lower Limit Cpk: " + str(round(Cpl,2))   )

        st.markdown("""---""")




        #1 plot
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader("Distribution Plot")
        fig = cerate_displot(x, low_bound, upp_bound, bins, platform)
        st.write(fig)

        #2 plot
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.markdown("""---""")
        st.subheader("Control Chart")
        fig = plot_control_charts(df_filtered, values_dict, station_1_anal, low_bound, upp_bound, std)
        st.write(fig)


        #3 plot
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        fig = plot_quantiles(x, low_bound, upp_bound)
        st.write(fig)
    else:
        st.write('Preparing Study ...')

















    #st.write(len(df_filtered))

    #st.write(df_station.head())
    #st.write(values_dict[station_1_anal])
    #st.write(df_filtered[values_dict[station_1_anal]].mean())
    #st.write(x.max())
    #st.write(df_filtered[values_dict[station_1_anal]].max())
    #st.write(df_filtered[values_dict[station_1_anal]].min())
