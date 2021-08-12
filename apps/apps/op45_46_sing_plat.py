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

st.set_option('deprecation.showPyplotGlobalUse', False)

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

    #df['runs'] = df.groupby('continental barcode' )['continental barcode'].cumcount() + 1
    #df['runs'] = df['runs'].apply(lambda x: 1 if x > 4 else x)

    return df

@st.cache(allow_output_mutation=True)
def plot_frop_rates(df_plot, freq):
    drop_rate = df_plot.groupby([to_plot_freq[freq],'hlt_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index=to_plot_freq[freq], columns='hlt_failed_part')
    drop_rate = drop_rate.fillna(0)
    if len(drop_rate.columns) < 2:
        drop_rate[True]= 0
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    drop_rate = drop_rate.rename(columns={True: 'HLT NOK Amount'})
    drop_rate = drop_rate.rename(columns={False: 'HLT OK Amount'})
    avg_drop_rate = df_plot['hlt_failed_part'].mean()*100

    fig_drop = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , text='fail_%')
    fig_drop.update_traces(marker_color='#798D98', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6 ,texttemplate='%{text:.2s}', textposition='outside')
    fig_drop.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

    fig_drop.add_hline(y=avg_drop_rate, line_width=1.0, line_dash="solid", line_color="red")
    annotations = [dict(xref='paper', x=1.13, y=avg_drop_rate, xanchor='right', yanchor='middle', text='o.a' + ' {}%'.format(round(avg_drop_rate,2)),
                            font=dict(family='Arial',size=13), showarrow=False)]
    fig_drop.update_layout(annotations=annotations)

    fig_prod =  px.bar(drop_rate, x=drop_rate.index, y=['HLT OK Amount','HLT NOK Amount'])
    fig_prod.update_traces( marker_line_width=1.0, opacity=0.6)
    fig_prod.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

    return fig_drop, fig_prod

@st.cache(allow_output_mutation=True)
def plot_distributions_stripplot(df_plat, x, y, font_size, to_plot_freq_pyplot,freq,low_limit_to_show, up_limit_to_show, limit_1, limit_2, limit_3, show_shift, show_ok_nok, show_limits):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()

    df_plat = df_plat.sort_values(by=to_plot_freq_pyplot[freq], ascending=False)

    if show_ok_nok== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='hlt_failed_part')
    elif show_shift== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
    else:
        params = dict(x=x, y=y, data = df_plat,    dodge=True)



    sns.stripplot(size=2,  **params)
    plt.xlim(low_limit_to_show,up_limit_to_show)

    if to_plot_freq_pyplot[freq] == 'date':
        x_dates = df_plat[to_plot_freq_pyplot[freq]].dt.strftime('%Y-%m-%d').sort_values(ascending=False).unique()
        ax.set_yticklabels(labels=x_dates, ha='right')
    else:
        x_dates = df_plat[to_plot_freq_pyplot[freq]].sort_values(ascending=False).unique()
        ax.set_yticklabels(labels=x_dates, ha='right')
    if show_limits:
        if limit_1==True:
            plt.axvline(x=0.000022, color='red', label='6.8E-05', linestyle='--', linewidth=0.8)
        if limit_2==True:
            plt.axvline(x=0.000068, color='orange', label='9.0E-05', linestyle='--', linewidth=0.8)
        if limit_3==True:
            plt.axvline(x=0.00025 , color='green', label='1.5E-04', linestyle='--', linewidth=0.8)
    return fig


@st.cache(allow_output_mutation=True)
def plot_vacuum_distributions_stripplot(df_plat, x, y, font_size, to_plot_freq_pyplot,freq,low_limit_to_show, up_limit_to_show, show_shift, show_ok_nok):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()

    df_plat = df_plat.sort_values(by=to_plot_freq_pyplot[freq], ascending=False)

    if show_ok_nok== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='hlt_failed_part')
    elif show_shift== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
    else:
        params = dict(x=x, y=y, data = df_plat,    dodge=True)



    sns.stripplot(size=2,  **params)
    plt.xlim(low_limit_to_show,up_limit_to_show)

    if to_plot_freq_pyplot[freq] == 'date':
        x_dates = df_plat[to_plot_freq_pyplot[freq]].dt.strftime('%Y-%m-%d').sort_values(ascending=False).unique()
        ax.set_yticklabels(labels=x_dates, ha='right')
    else:
        x_dates = df_plat[to_plot_freq_pyplot[freq]].sort_values(ascending=False).unique()
        ax.set_yticklabels(labels=x_dates, ha='right')

    return fig

@st.cache(allow_output_mutation=True)
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def select_platform(df, platform):
    df_plat = df[df['platform_consolidated'] == platform]
    return df_plat

@st.cache(allow_output_mutation=True)
def select_runs(df, runs):
    df_runs = df[df['runs'].isin(runs)]
    return df_runs

@st.cache(allow_output_mutation=True)
def get_df_dates_fitlered_and_platforms_avalailable(df, day_from, day_up_to):
    df_dates = filter_dates(df, str(day_from), str(day_up_to))
    platforms = list(df_dates['platform_consolidated'].unique())
    return df_dates, platforms

@st.cache(allow_output_mutation=True)
def leak_distribution(show_single_points, df_plat, to_plot_freq_pyplot, freq, limit_to_show, limit_022, limit_068, limit_25, show_shift, show_ok_nok):
    if show_single_points == True:
        fig = plot_distributions_stripplot(df_plat=df_plat, x="leak value [mbarl/s]", y=to_plot_freq_pyplot[freq], font_size=6,
                                    to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, low_limit_to_show=0, up_limit_to_show=limit_to_show,
                                    limit_1=limit_022, limit_2=limit_068, limit_3=limit_25, show_shift=show_shift, show_ok_nok=show_ok_nok, show_limits=True)
    else:
        if show_ok_nok== True:
            fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y=to_plot_freq[freq], range_x=[0,limit_to_show],
                            color="hlt_failed_part", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            vertical_lines(fig, limit_022, limit_068, limit_25)
        elif show_shift == True:
            fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y=to_plot_freq[freq], range_x=[0,limit_to_show],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            vertical_lines(fig, limit_022, limit_068, limit_25)
        else:
            fig = px.box(data_frame=df_plat, x="leak value [mbarl/s]", y=to_plot_freq[freq], range_x=[0,limit_to_show])
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            vertical_lines(fig, limit_022, limit_068, limit_25)
    return fig

@st.cache(allow_output_mutation=True)
def plot_vacuum_distributions(df_plat, vacuum, vac1, vac2, show_single_points, show_ok_nok, show_shift, to_plot_freq_pyplot, freq, limit_022, limit_068,limit_25 ):
    if show_single_points:
        fig = plot_vacuum_distributions_stripplot(df_plat=df_plat, x=vacuum, y=to_plot_freq_pyplot[freq], font_size=6,
                                    to_plot_freq_pyplot= to_plot_freq_pyplot, freq=freq, low_limit_to_show=vac1, up_limit_to_show=vac2,
                                    show_shift=show_shift, show_ok_nok=show_ok_nok)
    else:
        if show_shift == True:
            fig = px.box(data_frame=df_plat, x=vacuum, y=to_plot_freq[freq],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'],range_x=[vac1,vac2])
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            vertical_lines(fig, limit_022, limit_068, limit_25)
        else:
            fig = px.box(data_frame=df_plat, x=vacuum, y=to_plot_freq[freq], range_x=[vac1,vac2])
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig

@st.cache(allow_output_mutation=True)
def plot_station_distribution(df_plat, x, show_limits ,show_single_points, to_plot_freq_pyplot, freq, lower_limit_to_show ,limit_to_show, limit_022, limit_068, limit_25, show_shift):
    if show_single_points:
        sns.set_style('darkgrid')
        plt.rcParams.update({'font.size': 6})
        fig, ax = plt.subplots()
        df_plat = df_plat.sort_values(by='station', ascending=True)
        if show_shift== True:
            params = dict(x=x, y='station', data = df_plat,    dodge=True, hue='shift')
        else:
            params = dict(x=x, y='station', data = df_plat,    dodge=True)
        sns.stripplot(size=2,  **params)
        plt.xlim(lower_limit_to_show,limit_to_show)

        if show_limits:
            if limit_022==True:
                plt.axvline(x=0.000022, color='red', label='2.2E-05', linestyle='--', linewidth=0.8)
            if limit_068==True:
                plt.axvline(x=0.000068, color='orange', label='6.8E-05', linestyle='--', linewidth=0.8)
            if limit_25==True:
                plt.axvline(x=0.00025 , color='green', label='2.5E-04', linestyle='--', linewidth=0.8)

    else:
        df_to_plot = df_plat.sort_values(by='station', ascending=False)
        if show_shift == True:
            fig = px.box(data_frame=df_to_plot, x=x, y='station', range_x=[lower_limit_to_show,limit_to_show],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        else:
            fig = px.box(data_frame=df_to_plot, x=x, y='station', range_x=[lower_limit_to_show,limit_to_show])
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

        if show_limits:
            fig = vertical_lines(fig, limit_022, limit_068, limit_25)

    return fig

@st.cache(allow_output_mutation=True)
def plot_station_median(df_plat,stations,line_plot_freq,freq, limit_to_show, limit_022, limit_068, limit_25):
    df_leak_total = pd.DataFrame()
    for station in stations:
        df_leak = pd.DataFrame()
        leak_rates = df_plat[df_plat['station'] == station]['leak value [mbarl/s]']
        df_leak['leak value [mbarl/s]'] = leak_rates.resample(line_plot_freq[freq]).median().dropna()
        df_leak['station'] = station
        df_leak_total = pd.concat([df_leak_total,df_leak], axis=0)

    fig = px.line(df_leak_total, x=df_leak_total.index, y="leak value [mbarl/s]", color='station', range_y=[0,limit_to_show], height=400)
    fig = hoizontal_lines(fig, limit_022, limit_068, limit_25)
    return fig

@st.cache(allow_output_mutation=True)
def plot_scatter_vacuum_leak_rates(df_plat, limit_leak,vac1, vac2, x, limit_022, limit_068, limit_25):
    fig = px.scatter(data_frame=df_plat, y="leak value [mbarl/s]", x=x,
                    range_y=[0,limit_leak], range_x=[vac1,vac2], color='hlt_failed_part', opacity=0.4)
    fig = hoizontal_lines(fig, limit_022, limit_068, limit_25)
    return fig

@st.cache(allow_output_mutation=True)
def plot_vacuum_steps_drop_rate(df_plat, vacuum, start, end, step):
    bins = vacuum + ' bins'
    df_plat[bins] = pd.cut( df_plat[vacuum], bins= list(np.arange(start, end, step))).astype('str')
    drop_rate = df_plat.groupby([bins,'hlt_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index=bins, columns='hlt_failed_part')
    drop_rate = drop_rate.fillna(0)
    if len(drop_rate.columns) < 2:
        drop_rate[True]= 0
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    drop_rate = drop_rate.rename(columns={True: 'HLT NOK Amount'})
    drop_rate =drop_rate.round(2)
    fig = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , text='fail_%')
    fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
    return fig

@st.cache(allow_output_mutation=True)
def median_leak_value_per_vacuum_time(df_plat,vac1, vac2, limit_leak, x , limit_022, limit_068, limit_25):
    data = df_plat.copy()
    data[x] = data[x].round(1)
    to_plot_median = data.groupby(x)['leak value [mbarl/s]'].median()
    fig = px.scatter(to_plot_median, x=to_plot_median.index, y="leak value [mbarl/s]", range_x=[vac1, vac2], range_y=[0, limit_leak], height=400  )
    fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
    fig = hoizontal_lines(fig, limit_022, limit_068, limit_25)
    return fig

to_plot_freq = {'day' : 'date', 'week': 'year_week_date', 'month': 'year_month' }
line_plot_freq = {'day' : 'D', 'week': 'W', 'month': 'M' }
to_plot_freq_pyplot = {'day' : 'date', 'week': 'year_week', 'month': 'year_month' }

@st.cache(allow_output_mutation=True)
def three_d_plot_1(df_plat, limit_leak, vac1, vac2, vac3, vac4):

    df_limited = df_plat[df_plat['leak value [mbarl/s]'] < limit_leak]
    fig = px.scatter_3d(df_limited, x='vacuum_time final [s]', y='vacuum_time 3 mbar [s]', z='leak value [mbarl/s]',
              color='hlt_failed_part', size='machine factor', range_y=[vac1, vac2], range_x=[vac3, vac4], opacity=0.5)
    return fig

@st.cache(allow_output_mutation=True)
def three_d_plot_2(df_plat, limit_leak, vac1, vac2, machine1, machine2):

    df_limited = df_plat[df_plat['leak value [mbarl/s]'] < limit_leak]
    fig = px.scatter_3d(df_limited, x='machine factor', y="vacuum_time 3 mbar [s]", z='leak value [mbarl/s]', color="hlt_failed_part", range_x=[machine1, machine2], range_y=[vac1,vac2],
                 opacity=0.5)
    return fig

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

#Limits
plat_limits_num = {'WL RA':0.000068,
                    'WK RA':0.000022,
                    'MB FA BR164':0.000022,
                    'Ram DT RA':0.00025,
                    'Ford RA':0.00025,
                    'Ram DS RA':0.00022,
                    'HON RA':0.00025,
                    'MB FA BR251':0.000022,
                    'Ford FA':0.00025,
                    'Unknown':0.00025,
                    'MB RA BR251':0.000022}

plat_limits_str = {'WL RA':'6.8E-05',
                    'WK RA':'2.2E-05',
                    'MB FA BR164':'2.2E-05',
                    'Ram DT RA':'2.5E-04',
                    'Ford RA':'2.5E-04',
                    'Ram DS RA':'2.2E-05',
                    'HON RA':'2.5E-04',
                    'MB FA BR251':'2.2E-05',
                    'Ford FA':'2.5E-04',
                    'Unknown':'2.5E-04',
                    'MB RA BR251':'2.2E-05'}


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
        day_up_to= col2.date_input("Up to date", datetime.date.today(), min_value=day_from + DateOffset(days=1))
        #Platform

    # query to get data ----------------------------------------------------------------------------------------------------
    #raw data
    query = "select * from hlt_op45_op46 where date_time > '%s'" %(str(d1))
    df= long_running_function_to_load_data(query = query)
    st.title("OP45 and OP46 Test (single platform analysis)")

    df_dates, platforms = get_df_dates_fitlered_and_platforms_avalailable(df, day_from, day_up_to)

    with st.sidebar:
        st.subheader('Platform')
        platform = st.selectbox('',  platforms)
        #freq
        freq = st.selectbox('Frequency to plot',  ('day','week','month'))

        limit_to_show = float(st.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04')))

        st.write('Limits to plot')
        col1, col2, col3 = st.beta_columns(3)
        limit_022 = col1.checkbox('2.2E-05')
        limit_068 = col2.checkbox('6.8E-05')
        limit_25 = col3.checkbox('2.5E-04')
        show_shift = st.checkbox('Show Shift Comparison?')

        st.write('Platform Limit')
        st.write(pd.DataFrame.from_dict(plat_limits_str, orient='index').rename(columns={0:'Leak Limit'}).loc[platform])

    #filter runs
    #df_runs = select_runs(df_dates, runs)
    #filter platform
    df_plat = select_platform(df_dates, platform)
    st.write('## Select Analysis')
    analysis = st.selectbox('',  ("Test1 & Vacuum time Distributions",'Drop Rate',"Station Analysis","Single Day", 'Test 1 vs Vacuum times', 'Machine Factor and 3D PLots'))
    st.markdown("""---""")

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        DISTRIBUTIONS                                                                                                                                  /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if analysis == "Test1 & Vacuum time Distributions":

        #1 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader(platform+ ' - Test Distribution over time')
        show_single_points = st.checkbox("Show single points", False, key='#1')
        show_ok_nok= st.checkbox("Show ok/NOK", False, key='vacuum time final 1')
        fig = leak_distribution(show_single_points, df_plat, to_plot_freq_pyplot, freq, limit_to_show, limit_022, limit_068, limit_25, show_shift, show_ok_nok)
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)

        #2 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader(platform+ ' - Vacuum time Distribution over time')
        col1, col2= st.beta_columns(2)
        vacuum = col1.selectbox('Select Vacuum to Plot:',  ("vacuum_time final [s]",'vacuum_time 3 mbar [s]'))
        if vacuum == "vacuum_time final [s]":
            vac1, vac2 = col2.slider("Select vacuum time range:",  0, 80, (18, 60), 3, key=['vac2_slider#1'])
        elif vacuum == 'vacuum_time 3 mbar [s]':
            vac1, vac2 = col2.slider("Select vacuum time range:",  0, 30, (10, 15), 1, key=['vac2_slider#1'])

        col1, col2, _ = st.beta_columns(3)
        show_single_points = col1.checkbox("Show single points", False, key='#2')
        show_ok_nok= col2.checkbox("Show ok/NOK", False, key='vacuum #1')

        fig = plot_vacuum_distributions(df_plat, vacuum, vac1, vac2, show_single_points, show_ok_nok, show_shift, to_plot_freq_pyplot, freq, limit_022, limit_068,limit_25 )
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Drop Rate                                                                                                                                      /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == 'Drop Rate':

        #1 & 2 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    elif analysis == "Single Day":

        #1 PLOT
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader(platform+ ' - Single day Leak rates')

        col1, col2 , _, _= st.beta_columns(4)
        day = col1.date_input("Select day to plot", datetime.date(2021, 6, 4))
        color_to_show = col2.selectbox('Comparison to show',  ('None','Station','Shift','OK/NOK'))


        def plot_single_day(df_plat,  day, limit_to_show, limit_022,  limit_068, limit_25, color_to_show):
            df_day_plot = df_plat.loc[str(day)]

            if color_to_show == 'Station':
                fig = px.scatter(df_day_plot, x = 'hour', y = 'leak value [mbarl/s]', range_y=[0,limit_to_show],
                                color='station')
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            elif color_to_show == 'Shift':
                fig = px.scatter(df_day_plot, x = 'hour', y = 'leak value [mbarl/s]', range_y=[0,limit_to_show],
                                color='shift')
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            elif color_to_show == 'OK/NOK':
                fig = px.scatter(df_day_plot, x = 'hour', y = 'leak value [mbarl/s]', range_y=[0,limit_to_show],
                                color='hlt_failed_part')
                fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
            else:
                fig = px.scatter(df_day_plot, x = 'hour', y = 'leak value [mbarl/s]', range_y=[0,limit_to_show])
                fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

            # Lines ----------
            fig = hoizontal_lines(fig, limit_022, limit_068, limit_25)
            return fig

        if str(day) in df_plat.index:
            fig =plot_single_day(df_plat,  day, limit_to_show, limit_022,  limit_068, limit_25, color_to_show)
            st.write(fig)
        else:
            st.write('Platform not run on ' + str(day))

        st.subheader(platform+ ' - Single day drop rates')
        fig_prod, fig_drop = plot_frop_rates(df_plat, 'day')

        st.write(fig_prod)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Station Analysis                                                                                                                               /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == "Station Analysis" :

        st.subheader('Station Analysis')

        #1 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader(platform + ' - Test Distribution per station')
        # station drop rate ------------------------------------------------------------------
        c1,c2,c3 = st.beta_columns(3)
        stations= list(df_plat.station.unique())
        c1.markdown(stations[0]+': ' + str(round(df_plat[df_plat['station'] == stations[0]]['hlt_failed_part'].mean()*100,2) ) + '% Fail')
        if len(stations) > 0:
            c2.markdown(stations[1]+': ' + str(round(df_plat[df_plat['station'] == stations[1]]['hlt_failed_part'].mean()*100,2) ) + '% Fail')

        # distribution ---------------------------------------------------------
        show_single_points = st.checkbox("Show single points", False, key='station_dist_1')
        fig = plot_station_distribution(df_plat, "leak value [mbarl/s]", True,show_single_points, to_plot_freq_pyplot, freq,0 , limit_to_show, limit_022, limit_068, limit_25, show_shift)
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


        #2 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader(platform + ' - Vacuum Distribution per station')
        col1,col2 = st.beta_columns(2)
        vacuum = col1.selectbox('Select Vacuum to Plot:',  ("vacuum_time final [s]",'vacuum_time 3 mbar [s]'), key='vacuum#2')

        # Distribution------------------------------------------------------------------
        if vacuum=="vacuum_time final [s]":
            vac1, vac2 = col2.slider("Select vacuum time range:",  0, 80, (15, 50), 3, key=['vac2_slider1'])
        elif vacuum=="vacuum_time 3 mbar [s]":
            vac1, vac2 = col2.slider("Select vacuum time range:",  0, 80, (9, 15), 3, key=['vac2_slider1'])

        show_single_points = st.checkbox("Show single points", False, key='station_dist_2')

        fig = plot_station_distribution(df_plat, vacuum, False,show_single_points, to_plot_freq_pyplot, freq,vac1 , vac2, limit_022, limit_068, limit_25, show_shift)
        if show_single_points == True:
            st.pyplot(fig)
        else:
            st.write(fig)


        #3 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Station median Test Rate per ' + freq)
        fig = plot_station_median(df_plat,stations,line_plot_freq,freq, limit_to_show, limit_022, limit_068, limit_25)
        st.write(fig)

        #4 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Metric over Time per Station' )
        col1,_ = st.beta_columns(2)
        metric = col1.selectbox('Select Metric to Plot:',  ("vacuum_time final [s]",'vacuum_time 3 mbar [s]', 'machine factor'), key='metric#2')
        ranges= {"vacuum_time final [s]":(15,70), 'vacuum_time 3 mbar [s]':(8,16), 'machine factor':(0,15)}
        fig = px.scatter(df_plat, x = 'date', y = metric, color='station', range_y=ranges[metric])
        fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Test Rate vs Vaccum time                                                                                                                       /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == 'Test 1 vs Vacuum times':

        #1 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader(platform+ ' - Test 1 vs Vacuum time')
        # 1st vacuum step ------------------------------------------------------
        st.write('### 1st vacuum step')
        col1, col2 = st.beta_columns(2)

        limit_leak = float(col1.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider'))

        vac1, vac2 = col2.slider("Select vacuum time range:",  0.0, 30.0, (9.0, 18.0), 0.5, key=['vac2_slider'])
        fig = plot_scatter_vacuum_leak_rates(df_plat, limit_leak,vac1, vac2, 'vacuum_time 3 mbar [s]' , limit_022, limit_068, limit_25)
        st.write(fig)


        #2 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Median line -----------------------------------------------------------
        st.write('Leak value median vs 1st step vacuum time')
        fig = median_leak_value_per_vacuum_time(df_plat,vac1, vac2, limit_leak, 'vacuum_time 3 mbar [s]' , limit_022, limit_068, limit_25)
        st.write(fig)


        #3 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.write('Drop Rate on vacuum ranges')
        vac = st.slider("Select vacuum time delimiter:",  0.0, 30.0,  10.0, 0.25, key=['vac2_sliders'])

        lower= df_plat[df_plat['vacuum_time 3 mbar [s]'] <= vac]['hlt_failed_part'].mean()
        upper= df_plat[df_plat['vacuum_time 3 mbar [s]'] > vac]['hlt_failed_part'].mean()

        c1,c2 = st.beta_columns(2)
        stations= list(df_plat.station.unique())
        c1.markdown('Vacuum time bellow ' + str(vac) +'(s): ' +str(round(lower*100,2) ) + '% Fail')
        c2.markdown('Vacuum time above  ' + str(vac) +'(s): ' +str(round(upper*100,2) ) + '% Fail')
        st.markdown('HLT Drop rate % vs vacuum_time 3 mbar bins')
        fig= plot_vacuum_steps_drop_rate(df_plat, "vacuum_time 3 mbar [s]", 5.0,20.0,0.25)
        st.write(fig)


        if df_plat['vacuum_time final [s]'].median() > 1:

            #4 PLOT
            #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            st.markdown("""---""")
            st.write('### 2nd vacuum step')
            col1, col2 = st.beta_columns(2)
            limit_leak = float(col1.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                    '6.9E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider2'))
            vac1, vac2 = col2.slider("Select vacuum time range:",  0, 60, (18, 50), 3, key=['vac2_slider2'])
            fig = plot_scatter_vacuum_leak_rates(df_plat, limit_leak,vac1, vac2, 'vacuum_time final [s]' , limit_022, limit_068, limit_25)
            st.write(fig)


            #5 PLOT
            #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            st.write('Leak value median vs final vacuum time')
            fig = median_leak_value_per_vacuum_time(df_plat,vac1, vac2, limit_leak, 'vacuum_time final [s]' , limit_022, limit_068, limit_25)
            st.write(fig)

            #6 PLOT
            #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            st.write('Drop Rate on vacuum ranges')
            vac = st.slider("Select vacuum time delimiter:",  0.0, 60.0,  22.0, 0.25, key=['vac2_sliders2'])
            lower= df_plat[df_plat['vacuum_time final [s]'] <= vac]['hlt_failed_part'].mean()
            upper= df_plat[df_plat['vacuum_time final [s]'] > vac]['hlt_failed_part'].mean()
            c1,c2 = st.beta_columns(2)
            stations= list(df_plat.station.unique())
            c1.markdown('Vacuum time bellow ' + str(vac) +'(s): ' +str(round(lower*100,2) ) + '% Fail')
            c2.markdown('Vacuum time above  ' + str(vac) +'(s): ' +str(round(upper*100,2) ) + '% Fail')
            fig= plot_vacuum_steps_drop_rate(df_plat, "vacuum_time final [s]", 5.0,40.0,0.25)
            st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #       Machine Factor and 3D PLots                                                                                                                     /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == 'Machine Factor and 3D PLots':
        #1 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Machine factor --------------------------------------------------------
        st.write('### Machine Factor')
        col1, col2 = st.beta_columns(2)
        limit_leak = float(col1.select_slider('Select max leak to plot:', ['5.0E-07','5.0E-06','2.23E-05',
                                '6.9E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider3'))
        vac1, vac2 = col2.slider("Select machine factor range:",  0.0, 20.0, (2.0, 15.0), 0.5, key=['vac2_slider3'])
        fig = px.scatter(data_frame=df_plat, y="leak value [mbarl/s]", x='machine factor',
                        range_y=[0,limit_leak], range_x=[vac1,vac2], color='hlt_failed_part', opacity=0.5)
        fig = plot_scatter_vacuum_leak_rates(df_plat, limit_leak,vac1, vac2, 'machine factor' , limit_022, limit_068, limit_25)
        st.write(fig)

        #2 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.write('Drop Rate on machine factor')
        vac = st.slider("Select machine factor delimiter:",  0.0, 20.0,  6.0, 0.25, key=['vac2_sliders3'])
        lower= df_plat[df_plat['machine factor'] <= vac]['hlt_failed_part'].mean()
        upper= df_plat[df_plat['machine factor'] > vac]['hlt_failed_part'].mean()
        c1,c2 = st.beta_columns(2)
        stations= list(df_plat.station.unique())
        c1.markdown('Machine factor bellow ' + str(vac) +'(s): ' +str(round(lower*100,2) ) + '% Fail')
        c2.markdown('Machine factor above' + str(vac) +'(s): ' +str(round(upper*100,2) ) + '% Fail')
        st.write('Drop Rate on machine factor')
        fig= plot_vacuum_steps_drop_rate(df_plat, 'machine factor', 0,12,0.25)
        st.write(fig)


        #3 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #buble plot --------------------------------------------------------------------------------------
        st.markdown("""---""")
        st.write("### 3D Bubble plot")
        st.write("Bubbles size: Machine factor")
        if df_plat['vacuum_time final [s]'].median() > 1:
            col1, col2, col3 = st.beta_columns(3)
            vac1, vac2 = col1.slider("Select vacuum_time 3 mbar [s] range:",  0, 25, (5, 16), 1, key=['vac2_slider5'])
            vac3, vac4 = col2.slider("Select vacuum_time final [s] range:",  0.0, 65.0, (15.0, 50.0), 2.0, key=['vac2_slider6'])
            limit_leak = float(col3.select_slider('Select max leak to filter:', ['5.0E-07','5.0E-06','2.23E-05',
                                    '6.9E-05','9.0E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider5'))
            fig = three_d_plot_1(df_plat, limit_leak, vac1, vac2, vac3, vac4)
            st.write(fig)
        else:
            col1, col2 , col3= st.beta_columns(3)
            machine1, machine2  = col1.slider("Select Machine factor range:",  0, 20, (0, 2), 1, key=['vac2_slider5'])
            vac1, vac2 = col2.slider("Select vacuum_time 3 mbar [s] range:",  0.0, 25.0, (5.0, 16.0), 1.0, key=['vac2_slider6'])
            limit_leak = float(col3.select_slider('Select max leak to filter:', ['5.0E-07','5.0E-06','2.23E-05',
                                    '6.9E-05','9.0E-05','1.5E-04','2.55E-04','5.0E-04','1.0E-03','1.0E-02'], value=('2.55E-04'), key='vacc_slider6'))
            fig = three_d_plot_2(df_plat, limit_leak, vac1, vac2, machine1, machine2)
            st.write(fig)
