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

def get_op45_46_db_sql(conn, query='select * from "hlt_op45_op46"'):
    df = pd.read_sql_query(query,con=conn, parse_dates=['date_time'])
    df.index = pd.to_datetime(df.date_time.dt.date)
    return df

@st.cache(allow_output_mutation=True)
def long_running_function_to_load_data(query = 'select * from "hlt_op45_op46"'):
    hours = range(0,24)
    shits = ['3rd']*7 + ['1st']*8 + ['2nd']*8 + ['3rd']
    shifts_dict = dict(zip(hours,shits))

    conn = connect_db_sql(database="Morganton", user= "postgres", password= "United2018", host="localhost", port="5432")
    df = get_op45_46_db_sql(conn, query=query)

    #eliminating duplicates
    df = df[~df.duplicated()]

    line = {'OP45A': 'Line 1 (OP45)', 'OP45B': 'Line 1 (OP45)', 'OP46A': 'Line 2 (OP46)', 'OP46B': 'Line 2 (OP46)'}
    df['line'] = df['station'].map(line)
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

    #df['runs'] = df.groupby('continental barcode' )['continental barcode'].cumcount() + 1
    #df['runs'] = df['runs'].apply(lambda x: 1 if x > 4 else x)

    return df

@st.cache(allow_output_mutation=True)
def plot_station_distributions_stripplot(df_plat, x, y, font_size, to_plot_freq_pyplot,freq, limit_to_show, limit_1, limit_2, limit_3, show_shift):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()
    df_plat = df_plat.sort_values(by=y, ascending=False)

    if show_shift== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
    else:
        params = dict(x=x, y=y, data = df_plat,    dodge=True)

    sns.stripplot(size=2,  **params)
    plt.xlim(0,limit_to_show)

    if limit_1==True:
        plt.axvline(x=0.000022, color='red', label='6.8E-05', linestyle='--', linewidth=0.8)
    if limit_2==True:
        plt.axvline(x=0.000068, color='orange', label='9.0E-05', linestyle='--', linewidth=0.8)
    if limit_3==True:
        plt.axvline(x=0.00025 , color='green', label='1.5E-04', linestyle='--', linewidth=0.8)
    return fig

@st.cache(allow_output_mutation=True)
def plot_frop_rates(df_dates, freq):
    drop_rate = df_dates.groupby([to_plot_freq[freq],'hlt_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index=to_plot_freq[freq], columns='hlt_failed_part')
    drop_rate = drop_rate.fillna(0)
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    drop_rate = drop_rate.rename(columns={True: 'HLT NOK Amount'})
    avg_drop_rate = df_dates['hlt_failed_part'].mean()*100

    fig_drop = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , text='fail_%', height=350)
    fig_drop.update_traces(marker_color='#798D98', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6 ,texttemplate='%{text:.2s}', textposition='outside')
    fig_drop.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

    fig_drop.add_hline(y=avg_drop_rate, line_width=1.0, line_dash="solid", line_color="red")
    annotations = [dict(xref='paper', x=1.13, y=avg_drop_rate, xanchor='right', yanchor='middle', text='o.a' + ' {}%'.format(round(avg_drop_rate,2)),
                            font=dict(family='Arial',size=13), showarrow=False)]
    fig_drop.update_layout(annotations=annotations)

    fig_prod =  px.bar(drop_rate, x=drop_rate.index, y='HLT NOK Amount', text='HLT NOK Amount', height=350)
    fig_prod.update_traces(marker_color='#F56531', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6,texttemplate='%{text:.3s}', textposition='outside')
    fig_prod.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    return fig_drop, fig_prod

@st.cache(allow_output_mutation=True)
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def select_platforms(df, platforms):
    df_dates = df[df['platform'].isin(platforms)]
    return df_dates

to_plot_freq = {'day' : 'date', 'week': 'year_week_date', 'month': 'year_month' }
line_plot_freq = {'day' : 'D', 'week': 'W', 'month': 'M' }
to_plot_freq_pyplot = {'day' : 'date', 'week': 'year_week', 'month': 'year_month' }

@st.cache(allow_output_mutation=True)
def hoizontal_lines(fig, limit_022, limit_068, limit_25):
    if limit_022 == True:
        fig.add_hline(y=0.000022, line_width=1.0, line_dash="dash", line_color="orange")
    if limit_068 == True:
        fig.add_hline(y=0.000068, line_width=1.0, line_dash="dash", line_color="red")
    if limit_25 == True:
        fig.add_hline(y=0.00025, line_width=1.0, line_dash="dash", line_color="green")
    return fig

@st.cache(allow_output_mutation=True)
def vertical_lines(fig, limit_022, limit_068, limit_25):
    if limit_022 == True:
        fig.add_vline(x=0.000022, line_width=1.0, line_dash="dash", line_color="red")
    if limit_068 == True:
        fig.add_vline(x=0.000068, line_width=1.0, line_dash="dash", line_color="orange")
    if limit_25 == True:
        fig.add_vline(x=0.00025, line_width=1.0, line_dash="dash", line_color="green")
    return fig

@st.cache(allow_output_mutation=True)
def plot_platform_leak_distribution(show_single_points, df_plat, to_plot_freq_pyplot, freq, limit_to_show, limit_022, limit_068, limit_25, show_shift):
    if show_single_points == True:
        fig = plot_station_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y='platform', font_size=6,
                                    to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, limit_to_show=limit_to_show,
                                    limit_1=limit_022, limit_2=limit_068, limit_3=limit_25, show_shift=show_shift)
    else:
        df_plat_ordered = df_plat.sort_values(by='platform', ascending=True)
        if show_shift == True:
            fig = px.box(data_frame=df_plat_ordered, x="leak value [mbarl/s]", y='platform', range_x=[0,limit_to_show],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'], height=650)
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        else:
            fig = px.box(data_frame=df_plat_ordered, x="leak value [mbarl/s]", y='platform', range_x=[0,limit_to_show], height=650)
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

        vertical_lines(fig, limit_022, limit_068, limit_25)
    return fig

@st.cache(allow_output_mutation=True)
def plot_drop_rates(df_plat, y):
    df_drop = df_plat.groupby([y,'hlt_failed_part'])['platform_2'].count().reset_index().rename(columns={'platform_2':'count'})
    df_drop = pd.pivot_table(df_drop, values='count', index=y, columns='hlt_failed_part').fillna(0)
    if len(df_drop.columns) < 2:
        df_drop[True]= 0
    df_drop['Fail_%'] = df_drop.iloc[:,1] / (df_drop.iloc[:,1] + df_drop.iloc[:,0])*100
    df_drop['Total'] = df_drop.iloc[:,1] + df_drop.iloc[:,0]
    df_drop = df_drop.sort_values(by='Fail_%')
    df_drop['NOK_%_of_total'] = (df_drop[True] / df_drop[True].sum())*100

    fig_drop_rate = px.bar(df_drop, x="Fail_%", y=df_drop.index, orientation='h', height=300)
    fig_drop_rate.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    fig_drop_percent = px.bar(df_drop, x='NOK_%_of_total', y=df_drop.index, orientation='h', height=300)
    fig_drop_percent.update_traces(marker_color='#71978C', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig_drop_rate, fig_drop_percent

@st.cache(allow_output_mutation=True)
def cumsum_nok(df_plat, platforms, line_plot_freq, freq):
    df_nok_total = pd.DataFrame()
    for platform in platforms:
        df_nok = pd.DataFrame()
        df_platform = df_plat[(df_plat['platform'] == platform) & (df_plat['hlt_failed_part'] == True) ]
        df_platform['ocurr'] = 1
        df_nok['NOK_cummulative'] = df_platform['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
        df_nok['platform'] = platform

        df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

    fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='platform')

    return fig

@st.cache(allow_output_mutation=True)
def plot_median_lean_per_platform(df_plat,platforms,line_plot_freq, limit_to_show, freq,limit_022, limit_068, limit_25):
    df_leak_total = pd.DataFrame()
    for platform in platforms:
        df_leak = pd.DataFrame()
        leak_rates = df_plat[df_plat['platform'] == platform]['leak value [mbarl/s]']
        df_leak['leak value [mbarl/s]'] = leak_rates.resample(line_plot_freq[freq]).median().dropna()
        df_leak['platform'] = platform

        df_leak_total = pd.concat([df_leak_total,df_leak], axis=0)

    fig = px.line(df_leak_total, x=df_leak_total.index, y="leak value [mbarl/s]", color='platform', range_y=[0,limit_to_show], height=400)

    fig = hoizontal_lines(fig, limit_022, limit_068, limit_25)

    return fig

@st.cache(allow_output_mutation=True)
def plot_station_leak_distribution(show_single_points, df_plat, to_plot_freq_pyplot, freq, limit_to_show, limit_022, limit_068, limit_25, show_shift):
    if show_single_points:
        fig = plot_station_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y='station', font_size=6,
                                    to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, limit_to_show=limit_to_show,
                                    limit_1=limit_022, limit_2=limit_068, limit_3=limit_25, show_shift=show_shift)

    else:
        df_plat_ordered = df_plat.sort_values(by='station', ascending=True)
        if show_shift == True:
            fig = px.box(data_frame=df_plat_ordered, x="leak value [mbarl/s]", y='station', range_x=[0,limit_to_show],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'], height=650)
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        else:
            fig = px.box(data_frame=df_plat_ordered, x="leak value [mbarl/s]", y='station', range_x=[0,limit_to_show], height=650)
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

        vertical_lines(fig, limit_022, limit_068, limit_25)
    return fig

@st.cache(allow_output_mutation=True)
def plot_vacuum_distribution(show_single_points ,df_plat, vac1, vac2, show_ok_nok,show_shift,x ,y):
    if show_single_points == True:
        sns.set_style('darkgrid')
        plt.rcParams.update({'font.size': 6})
        fig, ax = plt.subplots()
        df_plat = df_plat.sort_values(by=y, ascending=True)
        if show_ok_nok== True:
            params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='hlt_failed_part')
        elif show_shift== True:
            params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
        else:
            params = dict(x=x, y=y, data = df_plat,    dodge=True)
        sns.stripplot(size=2,  **params)
        plt.xlim(vac1,vac2)
    else:
        df_plat = df_plat.sort_values(by=y, ascending=False)
        if show_ok_nok == True:
            fig = px.box(data_frame=df_plat, x=x, y=y,
                            color="hlt_failed_part", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'],range_x=[vac1,vac2])
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        elif show_shift== True:
            fig = px.box(data_frame=df_plat, x=x, y=y,
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'],range_x=[vac1,vac2])
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        else:
            fig = px.box(data_frame=df_plat, x=x, y=y, range_x=[vac1,vac2])
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig

@st.cache(allow_output_mutation=True)
def cumsum_nok_station(df_plat, stations, line_plot_freq, freq):
    df_nok_total = pd.DataFrame()
    for station in stations:
        df_nok = pd.DataFrame()
        df_station = df_plat[(df_plat['station'] == station) & (df_plat['hlt_failed_part'] == True) ]
        df_station['ocurr'] = 1
        df_nok['NOK_cummulative'] = df_station['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
        df_nok['station'] = station
        df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

    fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='station')
    return fig

@st.cache(allow_output_mutation=True)
def station_median(df_plat, variable, line_plot_freq, freq, stations,  limit_022, limit_068, limit_25, limit_to_show):
    df_leak_total = pd.DataFrame()
    for station in stations:
        df_leak = pd.DataFrame()
        leak_rates = df_plat[df_plat['station'] == station][variable]
        df_leak[variable] = leak_rates.resample(line_plot_freq[freq]).median().dropna()
        df_leak['station'] = station
        df_leak_total = pd.concat([df_leak_total,df_leak], axis=0)


    # Lines ---------
    if variable == 'leak value [mbarl/s]':
        fig = px.line(df_leak_total, x=df_leak_total.index, y=variable, color='station', range_y=[0,limit_to_show], height=400)
        if limit_022 == True:
            fig.add_hline(y=0.000022, line_width=1.0, line_dash="dash", line_color="orange")
        if limit_068 == True:
            fig.add_hline(y=0.000068, line_width=1.0, line_dash="dash", line_color="red")
        if limit_25 == True:
            fig.add_hline(y=0.00025, line_width=1.0, line_dash="dash", line_color="green")
    else:
        fig = px.line(df_leak_total, x=df_leak_total.index, y=variable, color='station', height=400)
    return fig

@st.cache(allow_output_mutation=True)
def bubble_stations(df_plat, include_op45, x_axis, x_axis1, x_axis2, vac1, vac2, machine1, machine2, limit_leak):
    if include_op45 == False:
        df_op46 = df_plat[df_plat['station'].isin(['OP46A','OP46B'])]
        df_limited = df_op46[df_op46['leak value [mbarl/s]'] < limit_leak]
        fig = px.scatter_3d(df_limited, x=x_axis, y="vacuum_time final [s]", z='leak value [mbarl/s]', color="station", range_x=[x_axis1, x_axis2], range_y=[vac1,vac2], #size='leak value [mbarl/s]' ,
                      opacity=0.2)
    else:
        df_limited = df_plat[df_plat['leak value [mbarl/s]'] < limit_leak]
        fig = px.scatter_3d(df_limited, y='machine factor', x=x_axis, z='leak value [mbarl/s]', color="station", range_x=[x_axis1, x_axis2], range_y=[machine1, machine2], #size='leak value [mbarl/s]' ,
                      opacity=0.2)
    return fig

@st.cache(allow_output_mutation=True)
def bubble_platforms(df_plat, x_axis, x_axis1, x_axis2, vac1, vac2, limit_leak):
    df_limited = df_plat[df_plat['leak value [mbarl/s]'] < limit_leak]
    fig = px.scatter_3d(df_limited, x=x_axis, y="vacuum_time final [s]", z='leak value [mbarl/s]', color="platform", range_x=[x_axis1, x_axis2], range_y=[vac1,vac2], #size='leak value [mbarl/s]' ,
                  opacity=0.2)
    return fig

plat_limits_str = {'WL RA':'2.7E-04',
                    'WK RA':'2.7E-05',
                    'MB FA BR164':'2.7E-05',
                    'Ram DT RA':'2.7E-05',
                    'Ford RA':'2.7E-05',
                    'Ram DS RA':'2.7E-05',
                    'HON RA':'2.7E-04',
                    'MB FA BR251':'2.2E-05',
                    'Ford FA':'2.7E-04',
                    'Unknown':'2.5E-04',
                    'MB RA BR251':'2.7E-04'}

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
    query = "select * from hlt_op45_op46 where date_time > '%s'" %(str(d1))
    df= long_running_function_to_load_data(query = query )
    #Date range Leak Distribution Plot and Srop rate plot--------------
    df_dates = filter_dates(df, str(day_from), str(day_up_to))
    #Checking wha tplatform are present in date range
    platforms = list(df_dates['platform'].unique())
    #checking for runs
    #runs=list(df_dates['runs'].unique())

    with st.sidebar:
        #Selecting platforms
        st.subheader('Platforms to compare:')
        platforms = st.multiselect('',platforms, platforms[:5], key='platform_select')

        #st.subheader('Runs to compare:')
        #runs = st.multiselect('',runs, runs[:1], key='runs_select')

        #freq
        freq = st.selectbox('Select frequency to analyze:',  ('day','week','month'))
        #Limits to show
        limit_to_show = float(st.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04')))

        #limit lines to show
        st.write('Limits to plot')
        col1, col2, col3 = st.beta_columns(3)
        limit_022 = col1.checkbox('2.2E-05')
        limit_068 = col2.checkbox('6.8E-05')
        limit_25 = col3.checkbox('2.5E-04')
        show_shift = st.checkbox('Show Shift Comparison?')

        #st.write(pd.DataFrame.from_dict(plat_limits_str, orient='index').rename(columns={0:'Leak Limit'}).loc[platforms])

    #Filtering Platforms    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    df_plat = select_platforms(df_dates, platforms)

    st.title("OP45 & OP46 Test")
    st.write("## Select Analysis")
    plat_station = st.selectbox('',  ("Platform",'Station',"Overall Drop Rate"))

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        PLATFORMS                                                                                                                                      /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if plat_station == "Platform":
        st.write('# Platform Comparison')
        #1stPLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        st.subheader('Leak Distribution per platform')
        show_single_points = st.checkbox("Show single points", False, key='01')
        fig = plot_platform_leak_distribution(show_single_points, df_plat, to_plot_freq_pyplot, freq, limit_to_show, limit_022, limit_068, limit_25, show_shift)
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


        #2nd PLot vacumm time
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Vacuum time Distribution per platform')
        col1, col2 = st.beta_columns(2)
        show_single_points= col1.checkbox("Show single points", False, key='vacuum time')
        show_ok_nok= col2.checkbox("Show ok/NOK", False, key='vacuum time final 2')

        col1, col2 = st.beta_columns(2)
        vacuum = col1.selectbox('Select Vacuum to Plot:',  ("vacuum_time final [s]",'vacuum_time 3 mbar [s]'))
        if vacuum == "vacuum_time final [s]":
            vac1, vac2 = col2.slider("Select vacuum time range:",  0.0, 80.0, (19.0, 60.0), 1.0, key=['vac2_slider#1'])
        else:
            vac1, vac2 = col2.slider("Select vacuum time range:",  0.0, 24.0, (9.0, 14.0), 1.0, key=['vac2_slider#2'])

        fig = plot_vacuum_distribution(show_single_points ,df_plat, vac1, vac2, show_ok_nok,show_shift, vacuum , 'platform' )
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


        #3 & 4 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.markdown("""---""")
        fig_drop_rate, fig_drop_percent = plot_drop_rates(df_plat, 'platform')
        st.subheader('Drop Rate % on HLT per platform')
        st.write(fig_drop_rate)
        st.subheader('% of NOK with respect to all NOK of current platforms selected')
        st.write(fig_drop_percent)



        #5 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Platforms NOK cummulative over time per ' + freq)
        fig = cumsum_nok(df_plat, platforms, line_plot_freq, freq)
        st.write(fig)


        #6 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Platform median Leak Rate per ' + freq)
        fig = plot_median_lean_per_platform(df_plat,platforms,line_plot_freq, limit_to_show, freq,limit_022, limit_068, limit_25)
        st.write(fig)

        #7 PLOT
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.markdown("""---""")
        st.write("### 3D plot per PLatform ")
        col1, col2= st.beta_columns(2)
        x_axis = col1.selectbox('Select X-Axis:',  ('machine factor','vacuum_time 3 mbar [s]'), key='num_2')
        limit_leak = float(col2.select_slider('Select max leak to filter:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','9.0E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider8'))
        col1, col2= st.beta_columns(2)
        x_axis1, x_axis2  = col1.slider("Select X_axis range:",  0.0, 25.0, (0.0, 12.0), 0.5, key=['vac2_slider8'])
        vac1, vac2 = col2.slider("Select vacuum_time Final [s] range:",  0.0, 80.0, (15.0, 55.0), 2.0, key=['vac2_slider9'])

        fig= bubble_platforms(df_plat, x_axis, x_axis1, x_axis2, vac1, vac2, limit_leak)

        st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        STATIONS                                                                                                                                       /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    elif plat_station == "Station":
        st.write('# Station Comparison')
        st.markdown("""---""")
        #1stPLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Leak Distribution per station')
        show_single_points = st.checkbox("Show single points", False, key='02')
        fig= plot_station_leak_distribution(show_single_points, df_plat, to_plot_freq_pyplot, freq, limit_to_show, limit_022, limit_068, limit_25, show_shift)
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


        #2nd PLot vacumm time
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Vacuum time Distribution per station')
        col1, col2 = st.beta_columns(2)
        show_single_points= col1.checkbox("Show single points", False, key='vacuum time2')
        show_ok_nok= col2.checkbox("Show ok/NOK", False, key='vacuum time final 3')

        col1, col2 = st.beta_columns(2)
        vacuum = col1.selectbox('Select Vacuum to Plot:',  ("vacuum_time final [s]",'vacuum_time 3 mbar [s]'))
        if vacuum == "vacuum_time final [s]":
            vac1, vac2 = col2.slider("Select vacuum time range:",  0.0, 80.0, (19.0, 60.0), 1.0, key=['vac2_slider#3'])
        else:
            vac1, vac2 = col2.slider("Select vacuum time range:",  0.0, 24.0, (9.0, 14.0), 1.0, key=['vac2_slider#4'])

        fig = plot_vacuum_distribution(show_single_points ,df_plat, vac1, vac2, show_ok_nok,show_shift, vacuum , 'station' )
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


        #3 & 4 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.markdown("""---""")
        fig_drop_rate, fig_drop_percent = plot_drop_rates(df_plat, 'station')
        st.subheader('Drop Rate % on HLT per station')
        st.write(fig_drop_rate)
        st.subheader('% of NOK with respect to all NOK of current station')
        st.write(fig_drop_percent)

        #5 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Stations NOK cummulative over time per ' + freq)
        stations = list(df_plat.station.unique())
        fig = cumsum_nok_station(df_plat, stations, line_plot_freq, freq)
        st.write(fig)


        #5 PLOT Metrics
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Station median variable per ' + freq)
        col1 , _ = st.beta_columns(2)
        variable = col1.selectbox('Select variable to plot',  ('leak value [mbarl/s]',"vacuum_time final [s]",'vacuum_time 3 mbar [s]'))
        fig = station_median(df_plat, variable, line_plot_freq, freq, stations,  limit_022, limit_068, limit_25, limit_to_show)
        st.write(fig)

        #6 Bubble Metrics
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.markdown("""---""")
        st.write("### 3D plot Vacuum per Station ")
        include_op45 = st.checkbox('Include OP45')
        if include_op45 == False:
            col1, col2= st.beta_columns(2)
            x_axis = col1.selectbox('Select X-Axis:',  ('machine factor','vacuum_time 3 mbar [s]'), key='num_2')
            limit_leak = float(col2.select_slider('Select max leak to filter:', ['5.0E-07','5.0E-06','2.23E-05',
                                    '6.9E-05','9.0E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider8'))
            col1, col2= st.beta_columns(2)
            x_axis1, x_axis2  = col1.slider("Select X_axis range:",  0.0, 25.0, (0.0, 12.0), 0.5, key=['vac2_slider8'])
            vac1, vac2 = col2.slider("Select vacuum_time Final [s] range:",  0.0, 80.0, (15.0, 55.0), 2.0, key=['vac2_slider9'])
            machine1, machine2 = 0, 0
            fig= bubble_stations(df_plat, include_op45, x_axis, x_axis1, x_axis2, vac1, vac2, machine1, machine2, limit_leak)
        else:
            x_axis = 'vacuum_time 3 mbar [s]'
            col1, col2 , col3= st.beta_columns(3)
            machine1, machine2  = col1.slider("Select Machine factor range:",  0.0, 20.0, (0.0, 10.0), 0.5, key=['vac2_slider10'])
            x_axis1, x_axis2 = col2.slider("Select vacuum_time 3 mbar [s] range:",  0.0, 25.0, (5.0, 16.0), 1.0, key=['vac2_slider11'])
            limit_leak = float(col3.select_slider('Select max leak to filter:', ['5.0E-07','5.0E-06','2.23E-05',
                                    '6.9E-05','9.0E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider11'))
            fig= bubble_stations(df_plat, include_op45, x_axis, x_axis1, x_axis2, vac1, vac2, machine1, machine2, limit_leak)

        st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        DROP RATES                                                                                                                                     /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif plat_station == "Overall Drop Rate":
        st.markdown("""---""")
        #drop rate over time per platform
        st.subheader('Overall Drop Rate over time & HLT Drop Rate')
        fig_prod, fig_drop = plot_frop_rates(df_plat, freq)

        st.write(fig_prod)
        st.write(fig_drop)
