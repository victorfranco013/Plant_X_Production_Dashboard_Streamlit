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
def long_running_function_to_load_data(query = 'select * from "hlt_op45_op46"'):
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
def plot_distributions_stripplot(df_plat, x, y, font_size, to_plot_freq_pyplot,freq, limit_to_show, limit_068, limit_090, limit_150, show_shift):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()

    df_plat = df_plat.sort_values(by=to_plot_freq_pyplot[freq], ascending=False)

    if show_shift== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
    else:
        params = dict(x=x, y=y, data = df_plat,    dodge=True)

    sns.stripplot(size=2,  **params)
    plt.xlim(0,limit_to_show)

    if to_plot_freq_pyplot[freq] == 'date':
        x_dates = df_plat[to_plot_freq_pyplot[freq]].dt.strftime('%Y-%m-%d').sort_values(ascending=False).unique()
        ax.set_yticklabels(labels=x_dates, ha='right')
    else:
        x_dates = df_plat[to_plot_freq_pyplot[freq]].sort_values(ascending=False).unique()
        ax.set_yticklabels(labels=x_dates, ha='right')

    if limit_068==True:
        plt.axvline(x=0.000068, color='red', label='6.8E-05', linestyle='--', linewidth=0.8)
    if limit_090==True:
        plt.axvline(x=0.000090, color='orange', label='9.0E-05', linestyle='--', linewidth=0.8)
    if limit_150==True:
        plt.axvline(x=0.00015 , color='green', label='1.5E-04', linestyle='--', linewidth=0.8)
    return fig

@st.cache(allow_output_mutation=True)
def plot_station_distributions_stripplot(df_plat, x, y, font_size, to_plot_freq_pyplot,freq, limit_to_show, limit_068, limit_090, limit_150, show_shift):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()

    df_plat = df_plat.sort_values(by=to_plot_freq_pyplot[freq], ascending=False)

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
def plot_frop_rates(df_plot, freq):
    drop_rate = df_plot.groupby([to_plot_freq[freq],'hlt_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index=to_plot_freq[freq], columns='hlt_failed_part')
    drop_rate = drop_rate.fillna(0)
    if len(drop_rate.columns) < 2:
        drop_rate[True]= 0
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    drop_rate = drop_rate.rename(columns={True: 'HLT NOK Amount'})
    avg_drop_rate = df_plot['hlt_failed_part'].mean()*100

    fig_drop = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , text='fail_%')
    fig_drop.update_traces(marker_color='#798D98', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6 ,texttemplate='%{text:.2s}', textposition='outside')
    fig_drop.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

    fig_drop.add_hline(y=avg_drop_rate, line_width=1.0, line_dash="solid", line_color="red")
    annotations = [dict(xref='paper', x=1.13, y=avg_drop_rate, xanchor='right', yanchor='middle', text='o.a' + ' {}%'.format(round(avg_drop_rate,2)),
                            font=dict(family='Arial',size=13), showarrow=False)]
    fig_drop.update_layout(annotations=annotations)

    fig_prod =  px.bar(drop_rate, x=drop_rate.index, y='HLT NOK Amount', text='HLT NOK Amount')
    fig_prod.update_traces(marker_color='#F56531', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6,texttemplate='%{text:.3s}', textposition='outside')
    fig_prod.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

    return fig_drop, fig_prod

@st.cache(allow_output_mutation=True)
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def select_platform(df, platform):
    df_plat = df[df['platform'] == platform]
    return df_plat

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

    #Platform Selection side bar and others ---------------------------------------------------------------------------------
    with st.sidebar:

        from_date = (datetime.date.today() - DateOffset(months=2)).date()
        st.subheader('From what date to load sql data?')
        d1 = st.date_input("", from_date)


        #date range
        st.write("### Date range to plot")
        col1, col2 = st.beta_columns(2)
        day_from= col1.date_input("From date", from_date,  min_value=d1)
        day_up_to= col2.date_input("Up to date", datetime.date.today(), min_value=day_from + DateOffset(days=2))
        #Platform

    # query to get data ----------------------------------------------------------------------------------------------------
    #raw data
    query = "select * from hlt_op180_185_from_200_210 where date_time > '%s'" %(str(d1))
    df= long_running_function_to_load_data(query = query)
    st.title("OP180 and OP185 Test (single platform analysis)")

    df_dates = filter_dates(df, str(day_from), str(day_up_to))

    platforms = list(df_dates['platform'].unique())

    with st.sidebar:
        st.subheader('Platform')
        platform = st.selectbox('',  platforms)

        #freq
        freq = st.selectbox('Frequency to plot',  ('day','week','month'))
        show_shift = st.checkbox('Show Shift Comparison?')

        limit_to_show = float(st.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','9.1E-05','1.55E-04','1.8E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04')))

        st.markdown("""---""")
        st.write('Limits to plot')
        col1, col2, col3 = st.beta_columns(3)
        limit_068 = col1.checkbox('6.8E-05')
        limit_090 = col2.checkbox('9.0E-05')
        limit_150 = col3.checkbox('1.5E-04')

        st.write('Platform Limit')
        st.write(pd.DataFrame.from_dict(plat_limits_str, orient='index').rename(columns={0:'Leak Limit'}).loc[platform])


    df_plat = select_platform(df_dates, platform)

    st.write("## Select Analysis")
    analysis = st.selectbox('',  ("Platform",'Station',"Single Day Analysis"))
    st.markdown("""---""")

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        PLATFORMS                                                                                                                                      /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if analysis == "Platform":
        #Date range Leak Distribution Plot and Srop rate plot--------------------------------------------------------------------------------------
        st.subheader(platform+ ' - Leak Distribution over time')
        # Distribution------------------------------------------------------------------
        if st.checkbox("Show single points", False):
            fig = plot_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y=to_plot_freq_pyplot[freq], font_size=6,
                                        to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, limit_to_show=limit_to_show,
                                        limit_068=limit_068, limit_090=limit_090, limit_150=limit_150, show_shift=show_shift)
            st.pyplot(fig)
        else:
            if show_shift == True:
                fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y=to_plot_freq[freq], range_x=[0,limit_to_show],
                                color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
                st.write(fig)
            else:
                fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y=to_plot_freq[freq], range_x=[0,limit_to_show])
                fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
                # Lines ----------
                fig = vertical_lines(fig, limit_068, limit_090, limit_150)

                st.write(fig)

        #Drop Rate --------------------------------------------------------------------------------------

        st.subheader(platform+ ' - Production & HLT Drop Rate')

        c1,c2,c3 = st.beta_columns(3)
        c1.markdown('1st Shift: ' + str(round(df_plat[df_plat['shift'] == '1st']['hlt_failed_part'].mean()*100,2) ) + '% Fail')
        c2.markdown('2nd Shift: ' + str(round(df_plat[df_plat['shift'] == '2nd']['hlt_failed_part'].mean()*100,2) ) + '% Fail')
        c3.markdown('3rd Shift: ' + str(round(df_plat[df_plat['shift'] == '3rd']['hlt_failed_part'].mean()*100,2) ) + '% Fail')

        fig_prod, fig_drop = plot_frop_rates(df_plat, freq)

        st.write(fig_prod)
        st.write(fig_drop)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Single Day                                                                                                                                     /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == "Single Day Analysis":


            st.subheader(platform+ ' - Single day Leak rates')

            #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            col1, _ , _, _= st.beta_columns(4)
            day = col1.date_input("Select day to plot", datetime.date.today())

            if str(day) in df_plat.index:
                df_day_plot = df_plat.loc[str(day)]

                if show_shift == True:
                    fig = px.scatter(df_day_plot, x = 'hour', y = 'leak value [mbarl/s]', range_y=[0,limit_to_show],
                                    color='shift',  color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
                    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
                else:
                    fig = px.scatter(df_day_plot, x = 'hour', y = 'leak value [mbarl/s]', range_y=[0,limit_to_show], color='station')
                    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

                hoizontal_lines(fig, limit_068, limit_090, limit_150)
                st.write(fig)

            else:
                st.write('Platform not run on ' + str(day))


            #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            st.subheader(platform+ ' - Single day drop rates')
            fig_prod, fig_drop = plot_frop_rates(df_plat, 'day')

            st.write(fig_prod)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        STATIONS                                                                                                                                       /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == "Station":
        st.subheader('Station Analysis')
        st.subheader(platform + ' - Leak Distribution per station')
        # Distribution------------------------------------------------------------------

        c1,c2,c3 = st.beta_columns(3)
        stations= list(df_plat.station.unique())
        c1.markdown(stations[0]+': ' + str(round(df_plat[df_plat['station'] == stations[0]]['hlt_failed_part'].mean()*100,2) ) + '% Fail')
        if len(stations) > 0:
            c2.markdown(stations[1]+': ' + str(round(df_plat[df_plat['station'] == stations[1]]['hlt_failed_part'].mean()*100,2) ) + '% Fail')

        if st.checkbox("Show single points", False, key='station'):
            fig = plot_station_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y='station', font_size=6,
                                        to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, limit_to_show=limit_to_show,
                                        limit_068=limit_068, limit_090=limit_090, limit_150=limit_150, show_shift=show_shift)
            st.pyplot(fig)

        else:
            if show_shift == True:
                fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y='station', range_x=[0,limit_to_show],
                                color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            else:
                fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y='station', range_x=[0,limit_to_show])
                fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            # Lines ----------
            vertical_lines(fig, limit_068, limit_090, limit_150)

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
        hoizontal_lines(fig, limit_068, limit_090, limit_150)

        st.write(fig)

        #NOK cummulatives over time
        st.subheader('Station NOK cummulative over time per ' + freq)
        df_nok_total = pd.DataFrame()
        for station in stations:
            df_nok = pd.DataFrame()
            df_platform = df_plat[(df_plat['station'] == station) & (df_plat['hlt_failed_part'] == True) ]
            df_platform['ocurr'] = 1
            df_nok['NOK_cummulative'] = df_platform['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
            df_nok['station'] = station
            df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

        fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='station')
        st.write(fig)
