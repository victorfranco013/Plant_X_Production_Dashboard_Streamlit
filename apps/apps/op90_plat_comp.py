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

def get_db_sql(conn, query='select * from "hlt_op090_from_200_210"'):
    df = pd.read_sql_query(query,con=conn, parse_dates=['date_time'])
    df.index = pd.to_datetime(df.date_time.dt.date)
    return df

@st.cache(allow_output_mutation=True)
def long_running_function_to_load_data(query = 'select * from "hlt_op090_from_200_210"'):
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


    return df

@st.cache(allow_output_mutation=True)
def plot_frop_rates(df_dates, freq):
    drop_rate = df_dates.groupby([to_plot_freq[freq],'op090_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index=to_plot_freq[freq], columns='op090_failed_part')
    drop_rate = drop_rate.fillna(0)
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    drop_rate['fail_%'] = drop_rate['fail_%'].round(2)
    drop_rate = drop_rate.rename(columns={True: 'OP90 NOK Amount'})
    drop_rate = drop_rate.rename(columns={False: 'OP90 OK Amount'})
    avg_drop_rate = df_dates['op090_failed_part'].mean()*100

    fig_drop = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , text='fail_%')
    fig_drop.update_traces(marker_color='#798D98', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6 )
    fig_drop.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

    fig_drop.add_hline(y=avg_drop_rate, line_width=1.0, line_dash="solid", line_color="red")
    annotations = [dict(xref='paper', x=1.13, y=avg_drop_rate, xanchor='right', yanchor='middle', text='o.a' + ' {}%'.format(round(avg_drop_rate,2)),
                            font=dict(family='Arial',size=13), showarrow=False)]
    fig_drop.update_layout(annotations=annotations)

    fig_prod =  px.bar(drop_rate, x=drop_rate.index, y=['OP90 OK Amount','OP90 NOK Amount'])
    fig_prod.update_traces(marker_line_width=1.0, opacity=0.6)
    fig_prod.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    return fig_drop, fig_prod

@st.cache(allow_output_mutation=True)
def plot_station_distributions_stripplot(df_plat, x, y, font_size, lower_L_show, upper_L_show, line_lower_l, line_upper_l , show_shift, show_limits):
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()
    df_plat = df_plat.sort_values(by='platform', ascending=False)

    if show_shift== True:
        params = dict(x=x, y=y, data = df_plat,    dodge=True, hue='shift')
    else:
        params = dict(x=x, y=y, data = df_plat,    dodge=True)

    sns.stripplot(size=2,  **params)
    plt.xlim(lower_L_show,upper_L_show)

    if show_limits == True:
        plt.axvline(x=line_lower_l, color='red', label='6.8E-05', linestyle='--', linewidth=0.8)
        plt.axvline(x=line_upper_l, color='red', label='6.8E-05', linestyle='--', linewidth=0.8)

    return fig

@st.cache(allow_output_mutation=True)
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def select_platforms(df, platforms):
    df_dates = df[df['platform'].isin(platforms)]
    return df_dates

@st.cache(allow_output_mutation=True)
def drop_rates_per_platforms(df_plat):
    df_drop = df_plat.groupby(['platform','op090_failed_part'])['platform_2'].count().reset_index().rename(columns={'platform_2':'count'})
    df_drop = pd.pivot_table(df_drop, values='count', index='platform', columns='op090_failed_part').fillna(0)
    if len(df_drop.columns) < 2:
        df_drop[True]= 0
    df_drop['Fail_%'] = df_drop.iloc[:,1] / (df_drop.iloc[:,1] + df_drop.iloc[:,0])*100
    df_drop['Total'] = df_drop.iloc[:,1] + df_drop.iloc[:,0]
    df_drop = df_drop.sort_values(by='Fail_%')
    df_drop['NOK_%_of_platforms'] = (df_drop[True] / df_drop[True].sum())*100

    fig_drop_rate = px.bar(df_drop, x="Fail_%", y=df_drop.index, orientation='h', height=300)
    fig_drop_rate.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    fig_oa_drop_percent = px.bar(df_drop, x='NOK_%_of_platforms', y=df_drop.index, orientation='h', height=300)
    fig_oa_drop_percent.update_traces(marker_color='#71978C', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig_drop_rate, fig_oa_drop_percent

@st.cache(allow_output_mutation=True)
def nok_cumsum_overtime(df_plat, platforms, line_plot_freq, freq):
    df_nok_total = pd.DataFrame()
    for platform in platforms:
        df_nok = pd.DataFrame()
        df_platform = df_plat[(df_plat['platform'] == platform) & (df_plat['op090_failed_part'] == True) ]
        df_platform['ocurr'] = 1
        df_nok['NOK_cummulative'] = df_platform['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
        df_nok['platform'] = platform

        df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

    fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='platform')
    return fig

@st.cache(allow_output_mutation=True)
def torque_dist_per_platform(df_plat, single_points ,lower_lim_show, upper_lim_show, lower_l, upper_l, show_shift, show_limits):
    if single_points == True:
        fig= plot_station_distributions_stripplot(df_plat, x='op090 torque [nm]', y='platform', font_size=6, lower_L_show=lower_lim_show,
                                                    upper_L_show=upper_lim_show, line_lower_l=lower_l, line_upper_l=upper_l ,
                                                    show_shift=show_shift, show_limits= show_limits)
    else:
        df_plat_single = df_plat.copy().sort_values(by='platform', ascending=True)
        if show_shift == True:
            fig = px.box(data_frame=df_plat_single, x="op090 torque [nm]", y='platform', range_x=[lower_lim_show,upper_lim_show],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'], height=650)
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        else:
            fig = px.box(data_frame=df_plat_single, x="op090 torque [nm]", y='platform', range_x=[lower_lim_show,upper_lim_show], height=650)
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

        # Lines ----------
        if show_limits == True:
            fig.add_vline(x=lower_l, line_width=1.0, line_dash="dash", line_color="red")
            fig.add_vline(x=upper_l, line_width=1.0, line_dash="dash", line_color="red")
    return fig

@st.cache(allow_output_mutation=True)
def angle_dist_per_platform(df_plat, single_points ,angle_lower_lim_show, angle_upper_lim_show, angle_lower_l, angle_upper_l, show_shift, show_angle_limits):
    if single_points:
        fig= plot_station_distributions_stripplot(df_plat, x='op090 angle [°]', y='platform', font_size=6, lower_L_show=angle_lower_lim_show,
                                                    upper_L_show=angle_upper_lim_show, line_lower_l=angle_lower_l, line_upper_l=angle_upper_l ,
                                                    show_shift=show_shift,  show_limits= show_angle_limits)
    else:
        df_plat_single = df_plat.copy().sort_values(by='platform', ascending=True)
        if show_shift == True:
            fig = px.box(data_frame=df_plat_single, x="op090 angle [°]", y='platform', range_x=[angle_lower_lim_show,angle_upper_lim_show],
                            color="shift", color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'], height=650)
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        else:
            fig = px.box(data_frame=df_plat_single, x="op090 angle [°]", y='platform', range_x=[angle_lower_lim_show,angle_upper_lim_show], height=650)
            fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

        # Lines ----------
        if show_angle_limits == True:
            fig.add_vline(x=angle_lower_l, line_width=1.0, line_dash="dash", line_color="red")
            fig.add_vline(x=angle_upper_l, line_width=1.0, line_dash="dash", line_color="red")
    return fig


@st.cache(allow_output_mutation=True)
def plot_torque_vs_angle(df_plat, show_limits ,lower_l, upper_l, angle_lower_lim_show,angle_upper_lim_show, lower_lim_show,upper_lim_show):
    fig = px.scatter(df_plat, x='op090 angle [°]', y="op090 torque [nm]", color="platform", range_x=[angle_lower_lim_show,angle_upper_lim_show], range_y=[lower_lim_show,upper_lim_show],
                  opacity=0.5, title='Torque vs Angle')
    if show_limits == True:
        fig.add_hline(y=lower_l, line_width=1.0, line_dash="dash", line_color="red")
        fig.add_hline(y=upper_l, line_width=1.0, line_dash="dash", line_color="red")
    return fig

@st.cache(allow_output_mutation=True)
def plot_torque_vs_angle_singleplat(df_plat, show_limits ,lower_l, upper_l, angle_lower_lim_show,angle_upper_lim_show, lower_lim_show,upper_lim_show):
    fig = px.scatter(df_plat, x='op090 angle [°]', y="op090 torque [nm]", color="op090_failed_part", range_x=[angle_lower_lim_show,angle_upper_lim_show], range_y=[lower_lim_show,upper_lim_show],
                  opacity=0.5)
    if show_limits == True:
        fig.add_hline(y=lower_l, line_width=1.0, line_dash="dash", line_color="red")
        fig.add_hline(y=upper_l, line_width=1.0, line_dash="dash", line_color="red")
    return fig

@st.cache(allow_output_mutation=True)
def plot_torque_angle_overtime(df_platform,platform_to_show, to_plot_freq ,freq, lower_lim_show,upper_lim_show,show_limits, lower_l, upper_l, angle_lower_lim_show,angle_upper_lim_show,show_angle_limits ,angle_lower_l, angle_upper_l):
    fig_torque = px.scatter(df_platform, y='op090 torque [nm]', x=to_plot_freq[freq], color="op090_failed_part", range_y=[lower_lim_show,upper_lim_show],
                  opacity=0.5, title=platform_to_show + ' Torque over time', height=400)
    if show_limits == True:
        fig_torque.add_hline(y=lower_l, line_width=1.0, line_dash="dash", line_color="red")
        fig_torque.add_hline(y=upper_l, line_width=1.0, line_dash="dash", line_color="red")

    #-----------------------------------------------------------------------

    fig_angle = px.scatter(df_platform, y='op090 angle [°]', x=to_plot_freq[freq], color="op090_failed_part", range_y=[angle_lower_lim_show,angle_upper_lim_show],
                  opacity=0.5, title=platform_to_show + ' Angle over time')
    if show_angle_limits == True:
        fig_angle.add_hline(y=angle_lower_l, line_width=1.0, line_dash="dash", line_color="red")
        fig_angle.add_hline(y=angle_upper_l, line_width=1.0, line_dash="dash", line_color="red")

    return fig_torque, fig_angle

@st.cache(allow_output_mutation=True)
def drop_rate_per_angle_bins(df_platform):
    df_platform['angle bins'] = pd.cut( df_platform["op090 angle [°]"], bins= list(np.arange(1.0,180.0,5.0))).astype('str')
    drop_rate = df_platform.groupby(['angle bins','op090_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index='angle bins', columns='op090_failed_part')
    drop_rate = drop_rate.fillna(0)
    if len(drop_rate.columns) < 2:
        drop_rate[True]= 0
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    drop_rate = drop_rate.rename(columns={True: 'OP90 NOK Amount'})
    drop_rate =drop_rate.round(2)
    fig = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , text='fail_%', title='OP90 Drop rate % vs angle bins')
    fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig

@st.cache(allow_output_mutation=True)
def plot_single_day_torque(day, df_plat,show_shift ,lower_lim_show , show_limits, upper_lim_show, lower_l, upper_l):
    df_day_plot = df_plat.loc[str(day)]
    if show_shift == True:
        fig = px.scatter(df_day_plot, x = 'hour', y = 'op090 torque [nm]', range_y=[lower_lim_show, upper_lim_show],
                        color='shift',  color_discrete_sequence =['#308196','#FFC300 ', '#34BA55'])
        fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
    else:
        fig = px.scatter(df_day_plot, x = 'hour', y = 'op090 torque [nm]', range_y=[lower_lim_show, upper_lim_show])
        fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
    # Lines ----------
    if show_limits == True:
        fig.add_hline(y=lower_l, line_width=1.0, line_dash="dash", line_color="red")
        fig.add_hline(y=upper_l, line_width=1.0, line_dash="dash", line_color="red")

    return fig

to_plot_freq = {'day' : 'date', 'week': 'year_week_date', 'month': 'year_month' }
line_plot_freq = {'day' : 'D', 'week': 'W', 'month': 'M' }
torque_limits = {'WL FA': [52.0,56.0], 'Ram DT FA': [52.0, 56.0], 'Ford FA': [46.7, 63.3], 'Jeep WK FA': [48.0, 52.0], 'HONDA FA': [55,60]}
angle_limits = {'WL FA': [1.0,90.0], 'Ram DT FA': [1.0, 90.0], 'Jeep WK FA': [1.0, 90.0], 'HONDA FA': [0,6000]}

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
        day_up_to= col2.date_input("Up to date", datetime.date.today(), min_value=day_from + DateOffset(days=2))

    # query to get data -------------------
    query = "select * from hlt_op090_from_200_210 where date_time > '%s'" %(str(d1))
    df= long_running_function_to_load_data(query = query)
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

        st.markdown("""---""")
        st.subheader('Torque Limits')
        #Limits to show
        lower_lim_show, upper_lim_show = st.slider('Select torque range to plot:', 0.0, 80.0, (46.0,61.0) ,1.0)

        #limit lines to show
        show_limits = st.checkbox('Show torque limits?')
        lower_l = 52.0
        upper_l = 56.0
        if show_limits == True:
            limit_to_show = st.selectbox('Select limits to plot:',  ('Manually','WL FA','Jeep WK FA','Ram DT FA','Ford FA','HONDA FA'))
            if limit_to_show == 'Manually':
                col1, col2 = st.beta_columns(2)
                lower_l  = col1.slider("Lower limit:",  0.0, 80.0, 52.0, 0.5, key='lower_limit_0')
                upper_l  = col2.slider("Upper limit:",  0.0, 80.0, 56.0, 0.5, key='upper_limit_0')
            else:
                lower_l =  torque_limits[limit_to_show][0]
                upper_l =  torque_limits[limit_to_show][1]
        st.markdown("""---""")

        #limit lines to show
        st.subheader('Angle Limits')
        angle_lower_lim_show, angle_upper_lim_show = st.slider('Select angle range to plot:', 0.0, 360.0, (5.0,180.0) ,5.0)
        show_angle_limits = st.checkbox('Show angle limits?')
        angle_lower_l = 0
        angle_upper_l = 90
        if show_angle_limits == True:
            angle_limit_to_show = st.selectbox('Select angle limits to plot:',  ('Manually','WL FA','Jeep WK FA','Ram DT FA'))
            if angle_limit_to_show == 'Manually':
                col1, col2 = st.beta_columns(2)
                angle_lower_l  = col1.slider("Lower limit:",  0.0, 360.0, 1.0, 0.5, key='lower_limit_1')
                angle_upper_l  = col2.slider("Upper limit:",  0.0, 360.0, 90.0, 0.5, key='upper_limit_1')
            else:
                angle_lower_l =  angle_limits[angle_limit_to_show][0]
                angle_upper_l =  angle_limits[angle_limit_to_show][1]
        st.markdown("""---""")


    #Filtering Platforms    ////////////////////////////////////////////////////
    df_plat = select_platforms(df_dates, platforms)

    st.title("OP090 Test 3 Station ")
    st.write('# Platform Comparison')


    st.write("## Select Analysis")
    analysis = st.selectbox('',  ("ALL Platforms",'Single Platform',"Single Day Analysis"))
    st.markdown("""---""")

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        PLATFORMS                                                                                                                                      /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if analysis == "ALL Platforms":

        #1st and 2nd PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Drop Rate % on OP90 per platform')
        fig_drop_rate, fig_oa_drop_percent = drop_rates_per_platforms(df_plat)
        st.write(fig_drop_rate)
        st.subheader('% of NOK with respect to all NOK of current platforms selected')
        st.write(fig_oa_drop_percent)


        #3rd PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #NOK cummulatives over time
        st.subheader('Platforms NOK cummulative over time per ' + freq)
        fig = nok_cumsum_overtime(df_plat, platforms, line_plot_freq, freq)
        st.write(fig)


        #4th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # torque Distribution per platform -----------------------------------------------------------------
        st.subheader('Torque per platform')
        single_points = st.checkbox("Show single points (slower)", False)
        fig = torque_dist_per_platform(df_plat, single_points ,lower_lim_show, upper_lim_show, lower_l, upper_l, show_shift, show_limits)
        if single_points == True:
            st.pyplot(fig)
        else:
            st.plotly_chart(fig)


        #5th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Angle per platform')
        single_points = st.checkbox("Show single points (slower)", False, key='angle_single_points')
        fig = angle_dist_per_platform(df_plat, single_points ,angle_lower_lim_show, angle_upper_lim_show, angle_lower_l, angle_upper_l, show_shift, show_angle_limits)
        if single_points == True:
            st.pyplot(fig)
        else:
            st.plotly_chart(fig)


        #6th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Torque vs angle
        fig =  plot_torque_vs_angle(df_plat, show_limits ,lower_l, upper_l, angle_lower_lim_show,angle_upper_lim_show, lower_lim_show,upper_lim_show)
        st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        SINGLE PLATFORM                                                                                                                                /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == 'Single Platform':

        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Single PLatform Analysis')
        platform_to_show = st.selectbox('Platform to show:',  platforms)
        df_platform = df_plat[df_plat['platform']== platform_to_show]

        #7th and 8th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        fig_torque,fig_angle =  plot_torque_angle_overtime(df_platform,platform_to_show, to_plot_freq ,freq, lower_lim_show,upper_lim_show,show_limits, lower_l, upper_l, angle_lower_lim_show,angle_upper_lim_show,show_angle_limits ,angle_lower_l, angle_upper_l)
        st.write(fig_torque)
        st.write(fig_angle)


        #9th and 10th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.write(platform_to_show+ ' - OP90 Drop Rate')

        c1,c2,c3 = st.beta_columns(3)
        c1.markdown('1st Shift: ' + str(round(df_platform[df_platform['shift'] == '1st']['op090_failed_part'].mean()*100,2) ) + '% Fail')
        c2.markdown('2nd Shift: ' + str(round(df_platform[df_platform['shift'] == '2nd']['op090_failed_part'].mean()*100,2) ) + '% Fail')
        c3.markdown('3rd Shift: ' + str(round(df_platform[df_platform['shift'] == '3rd']['op090_failed_part'].mean()*100,2) ) + '% Fail')

        fig_prod, fig_drop = plot_frop_rates(df_platform, freq)

        st.write(fig_prod)
        st.write(fig_drop)


        #11th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.write(platform_to_show + ' - Torque vs Angle')
        fig = plot_torque_vs_angle_singleplat(df_platform, show_limits ,lower_l, upper_l, angle_lower_lim_show,angle_upper_lim_show, lower_lim_show,upper_lim_show)
        st.write(fig)


        #12th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        fig = drop_rate_per_angle_bins(df_platform)
        st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        SINGLE DAY                                                                                                                                     /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == "Single Day Analysis":
        #13th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        platform_to_show = st.selectbox('Platform to show:',  platforms)
        df_platform = df_plat[df_plat['platform']== platform_to_show]
        st.subheader(platform_to_show+ ' - Single day torque rates')
        col1, _ , _, _= st.beta_columns(4)
        day = col1.date_input("Select day to plot", datetime.date.today())

        if str(day) in df_plat.index:
            fig = plot_single_day_torque(day,df_platform,show_shift, lower_lim_show , show_limits, upper_lim_show, lower_l, upper_l)
            st.write(fig)

        else:
            st.write('Platform not run on ' + str(day))
