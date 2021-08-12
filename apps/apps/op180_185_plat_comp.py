import streamlit as st
import pandas as pd
import numpy as np
import csv
import os
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from pandas.tseries.offsets import DateOffset

# Funcitons and constants-------------------------------------------------------------------------------------------------------------------
def connect_db_sql(database="Morganton", user= "postgres", password= "United2018", host="localhost", port="5432"):
    conn = psycopg2.connect(database= database, user=user, password=password, host=host, port=port)
    return conn

def get_db_sql(conn, query='select * from "hlt_op180_185_from_200_210"'):
    df = pd.read_sql_query(query,con=conn, parse_dates=['date_time'])
    df.index = pd.to_datetime(df.date_time.dt.date)
    return df

@st.cache(allow_output_mutation=True)
def long_running_function_to_load_data(query = 'select * from "hlt_op180_185_from_200_210"'):
    hours = range(0,24)
    shits = ['3rd']*7 + ['1st']*8 + ['2nd']*8 + ['3rd']
    shifts_dict = dict(zip(hours,shits))

    conn = connect_db_sql(database="Morganton", user= "postgres", password= "United2018", host="localhost", port="5432")
    df = get_db_sql(conn, query=query)

    #eliminating duplicates
    df = df[~df.duplicated()]

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
    df['platform'] = df['platform'].replace('Jeep WK FA ', 'Jeep WK FA')
    df['platform'] = df['platform'].replace('98765432.1', 'Unknown')

    platforms_to_analyze =['WL FA', 'Tesla FA Service', 'Tesla FA', 'Tesla RA', 'Ram DT FA', 'Ford FA', 'Jeep WK FA','Ram DS FA', 'HONDA FA']
    df = df[df['platform'].isin(platforms_to_analyze)]
    df = df[df['station'].isin(['OP180','OP185'])]
    return df

@st.cache(allow_output_mutation=True)
def plot_station_distributions_stripplot(df_plat, x, y, font_size, to_plot_freq_pyplot,freq, limit_to_show, limit_068, limit_090, limit_150, show_shift):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})
    df_plat =df_plat.sort_values(by='platform', ascending=False)

    fig, ax = plt.subplots()

    if show_shift== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
    else:
        params = dict(x=x, y=y, data = df_plat,    dodge=True)

    sns.stripplot(size=2,  **params)
    plt.xlim(0,limit_to_show)

    if limit_068==True:
        plt.axvline(x=0.000068, color='red', label='6.8E-05', linestyle='--', linewidth=0.8)
    if limit_090==True:
        plt.axvline(x=0.000090, color='orange', label='9.0E-05', linestyle='--', linewidth=0.8)
    if limit_150==True:
        plt.axvline(x=0.00015 , color='green', label='1.5E-04', linestyle='--', linewidth=0.8)
    return fig

@st.cache(allow_output_mutation=True)
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def select_platforms(df, platforms):
    df_dates = df[df['platform'].isin(platforms)]
    return df_dates

to_plot_freq = {'day' : 'date', 'week': 'year_week_date', 'month': 'year_month' }
to_plot_freq_pyplot = {'day' : 'date', 'week': 'year_week', 'month': 'year_month' }
line_plot_freq = {'day' : 'D', 'week': 'W', 'month': 'M' }

def hoizontal_lines(fig, limit_068, limit_090, limit_150):
    if limit_068 == True:
        fig.add_hline(y=0.000068, line_width=1.0, line_dash="dash", line_color="red")
    if limit_090 == True:
        fig.add_hline(y=0.000090, line_width=1.0, line_dash="dash", line_color="orange")
    if limit_150 == True:
        fig.add_hline(y=0.00015, line_width=1.0, line_dash="dash", line_color="green")
    return fig

def vertical_lines(fig, limit_068, limit_090, limit_150):
    if limit_068 == True:
        fig.add_vline(x=0.000069, line_width=1.0, line_dash="dash", line_color="red")
    if limit_090 == True:
        fig.add_vline(x=0.000090, line_width=1.0, line_dash="dash", line_color="orange")
    if limit_150 == True:
        fig.add_vline(x=0.00015, line_width=1.0, line_dash="dash", line_color="green")
    return fig

#Limits
plat_limits_num = {'WL FA':0.000068, 'Tesla FA Service':0.00009, 'Tesla FA':0.00009, 'Tesla RA':0.00009, 'Ram DT FA':0.00015, 'Ford FA':0.00015, 'Jeep WK FA':0.00015,'Ram DS FA':0.00015, 'HONDA FA':0.00015}
plat_limits_str = {'WL FA':'6.8E-05', 'Tesla FA Service':'9.0E-05', 'Tesla FA':'9.0E-05', 'Tesla RA':'9.0E-05', 'Ram DT FA':'1.5E-04', 'Ford FA':'1.5E-04', 'Jeep WK FA':'1.5E-04','Ram DS FA':'1.5E-04', 'HONDA FA':'1.5E-04'}


def app():
    #Platform Selection side bar and others ------------------------------------
    with st.sidebar:
        from_date = (datetime.date.today() - DateOffset(months=2)).date()
        st.subheader('From what date to load sql data?')
        d1 = st.date_input("", from_date)

        #date range
        st.write("### Date range to analyze")
        col1, col2 = st.beta_columns(2)
        day_from= col1.date_input("From date", from_date,  min_value=d1)
        day_up_to= col2.date_input("Up to date", datetime.date.today(), min_value=day_from + DateOffset(days=1))

    # query to get data -------------------
    query = "select * from hlt_op180_185_from_200_210 where date_time > '%s'" %(str(d1))
    df= long_running_function_to_load_data(query = query )
    #Date range Leak Distribution Plot and Srop rate plot--------------
    df_dates = filter_dates(df, str(day_from), str(day_up_to))
    #Checking wha tplatform are present in date range
    platforms = list(df_dates['platform'].unique())

    with st.sidebar:
        #Selecting platforms
        st.subheader('Platforms to compare:')
        platforms = st.multiselect('',platforms, platforms[:5], key='platform_select')
        #freq
        freq = st.selectbox('Select frequency to analyze:',  ('day','week','month'))
        show_shift = st.checkbox('Show Shift Comparison?')
        #limits up to show
        limit_to_show = float(st.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','9.1E-05','1.55E-04','1.8E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04')))

        st.markdown("""---""")


        st.write('Limits to plot')
        col1, col2, col3 = st.beta_columns(3)
        limit_068 = col1.checkbox('6.8E-05')
        limit_090 = col2.checkbox('9.0E-05')
        limit_150 = col3.checkbox('1.5E-04')

        st.write(pd.DataFrame.from_dict(plat_limits_str, orient='index').rename(columns={0:'Leak Limit'}).loc[platforms])


    st.title("OP180 and OP185 Test 2")
    #Filter platform selection
    df_plat = select_platforms(df_dates, platforms)

    st.write("## Select Analysis")
    plat_station = st.selectbox('',  ("Platform",'Station'))
    st.markdown("""---""")

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        PLATFORMS                                                                                                                                      /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    if plat_station == "Platform":
        # Distribution per platform -----------------------------------------------------------------
        st.subheader('Leak Distribution per platform')

        if st.checkbox("Show single points", False):
            fig = plot_station_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y='platform', font_size=6,
                                        to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, limit_to_show=limit_to_show,
                                        limit_068=limit_068, limit_090=limit_090, limit_150=limit_150, show_shift=show_shift)
            st.pyplot(fig)
        else:
            df_to_plot = df_plat.copy()
            df_to_plot =df_to_plot.sort_values(by='platform')
            if show_shift == True:
                fig = px.box(data_frame=df_to_plot, x="leak value [mbarl/s]", y='platform', range_x=[0,limit_to_show],
                                color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'], height=650)
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            else:
                fig = px.box(data_frame=df_to_plot, x="leak value [mbarl/s]", y='platform', range_x=[0,limit_to_show], height=650)
                fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

            # Lines ----------
            vertical_lines(fig, limit_068, limit_090, limit_150)

            st.write(fig)


        st.markdown("""---""")
        #Drop Rate Comparison per platform ------------------------------------------------------------------------------------------------------------------
        df_drop = df_plat.groupby(['platform','hlt_failed_part'])['platform_2'].count().reset_index().rename(columns={'platform_2':'count'})
        df_drop = pd.pivot_table(df_drop, values='count', index='platform', columns='hlt_failed_part').fillna(0)
        if len(df_drop.columns) < 2:
            df_drop[True]= 0
        df_drop['Fail_%'] = df_drop.iloc[:,1] / (df_drop.iloc[:,1] + df_drop.iloc[:,0])*100
        df_drop['Total'] = df_drop.iloc[:,1] + df_drop.iloc[:,0]
        df_drop = df_drop.sort_values(by='Fail_%')
        df_drop['NOK_%_of_platforms'] = (df_drop[True] / df_drop[True].sum())*100

        st.subheader('Drop Rate % on HLT per platform')
        fig = px.bar(df_drop, x="Fail_%", y=df_drop.index, orientation='h', height=300)
        fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)

        st.subheader('% of NOK with respect to all NOK of current platforms selected')
        fig = px.bar(df_drop, x='NOK_%_of_platforms', y=df_drop.index, orientation='h', height=300)
        fig.update_traces(marker_color='#71978C', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)

        #NOK cummulatives over time
        st.subheader('Platforms NOK cummulative over time per ' + freq)
        df_nok_total = pd.DataFrame()
        for platform in platforms:
            df_nok = pd.DataFrame()
            df_platform = df_plat[(df_plat['platform'] == platform) & (df_plat['hlt_failed_part'] == True) ]
            df_platform['ocurr'] = 1
            df_nok['NOK_cummulative'] = df_platform['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
            df_nok['platform'] = platform

            df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

        fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='platform')
        st.write(fig)


        #Metrics overtime
        #Median------------------------------
        st.subheader('Platform median Leak Rate per ' + freq)
        df_leak_total = pd.DataFrame()
        for platform in platforms:
            df_leak = pd.DataFrame()
            leak_rates = df_plat[df_plat['platform'] == platform]['leak value [mbarl/s]']
            df_leak['leak value [mbarl/s]'] = leak_rates.resample(line_plot_freq[freq]).median().dropna()
            df_leak['platform'] = platform

            df_leak_total = pd.concat([df_leak_total,df_leak], axis=0)
        fig = px.line(df_leak_total, x=df_leak_total.index, y="leak value [mbarl/s]", color='platform', range_y=[0,limit_to_show], height=400)
        # Lines ----------
        fig = hoizontal_lines(fig, limit_068, limit_090, limit_150)
        st.write(fig)


    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        STATIONS                                                                                                                                       /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    elif plat_station == "Station":
        #Per Station -----------------------------------------------------------------------------------------------------
        st.write('# Station Comparison')
        st.subheader('Leak Distribution per station')

        if st.checkbox("Show single points", False ,key='dist_2'):
            fig = plot_station_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y='station', font_size=6,
                                        to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, limit_to_show=limit_to_show,
                                        limit_068=limit_068, limit_090=limit_090, limit_150=limit_150, show_shift=show_shift)
            st.pyplot(fig)
        else:
            df_to_plot = df_plat.sort_values(by='station', ascending=True)
            if show_shift == True:
                fig = px.box(data_frame=df_to_plot, x="leak value [mbarl/s]", y='station', range_x=[0,limit_to_show],
                                color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'], height=450)
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            else:
                fig = px.box(data_frame=df_to_plot, x="leak value [mbarl/s]", y='station', range_x=[0,limit_to_show], height=450)
                fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

                # Lines ----------
            fig = hoizontal_lines(fig, limit_068, limit_090, limit_150)

            st.plotly_chart(fig)


        #Drop Rate Comparison per station------------------------------------------------------------------------------------------------------------------
        st.subheader('Drop Rate % on HLT per station')
        df_drop = df_plat.groupby(['station','hlt_failed_part'])['platform_2'].count().reset_index().rename(columns={'platform_2':'count'})
        df_drop = pd.pivot_table(df_drop, values='count', index='station', columns='hlt_failed_part').fillna(0)
        if len(df_drop.columns) < 2:
            df_drop[True]= 0
        df_drop['Fail_%'] = df_drop.iloc[:,1] / (df_drop.iloc[:,1] + df_drop.iloc[:,0])*100
        df_drop['Total'] = df_drop.iloc[:,1] + df_drop.iloc[:,0]
        df_drop = df_drop.sort_values(by='Fail_%')
        df_drop['NOK_%_of_stations'] = (df_drop[True] / df_drop[True].sum())*100

        fig = px.bar(df_drop, x="Fail_%", y=df_drop.index, orientation='h', height=300)
        fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)

        st.subheader('Station % of NOK with respect to all NOK  ')
        fig = px.bar(df_drop, x='NOK_%_of_stations', y=df_drop.index, orientation='h', height=300)
        fig.update_traces(marker_color='#71978C', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)

        #NOK cummulatives over time
        st.subheader('Stations NOK cummulative over time per ' + freq)
        stations = list(df_plat.station.unique())
        df_nok_total = pd.DataFrame()
        for station in stations:
            df_nok = pd.DataFrame()
            df_station = df_plat[(df_plat['station'] == station) & (df_plat['hlt_failed_part'] == True) ]
            df_station['ocurr'] = 1
            df_nok['NOK_cummulative'] = df_station['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
            df_nok['station'] = station
            df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

        fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='station')
        st.write(fig)


        #Metrics overtime
        #Median------------------------------
        st.subheader('Station median Leak Rate per ' + freq)
        df_leak_total = pd.DataFrame()
        for station in stations:
            df_leak = pd.DataFrame()
            leak_rates = df_plat[df_plat['station'] == station]['leak value [mbarl/s]']
            df_leak['leak value [mbarl/s]'] = leak_rates.resample(line_plot_freq[freq]).median().dropna()
            df_leak['station'] = station
            df_leak_total = pd.concat([df_leak_total,df_leak], axis=0)

        fig = px.line(df_leak_total, x=df_leak_total.index, y="leak value [mbarl/s]", color='station', range_y=[0,limit_to_show], height=400)

        # Lines ----------
        fig = hoizontal_lines(fig, limit_068, limit_090, limit_150)

        st.write(fig)
