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
def filter_dates(df, date1, date2):
    df_filtered = df.loc[date1 : date2]
    return df_filtered

@st.cache(allow_output_mutation=True)
def concatenate(df200_plot, df210_plot, df45_46_plot,customer_map):
    features_to_concat= ['line','station','station_II','platform','date_time','date_only','year_month','year_week','week','year','hour','shift','failed']
    df_all = pd.concat([df200_plot[features_to_concat],
                        df210_plot[features_to_concat],
                        df45_46_plot[features_to_concat]], axis=0)
    df_all['customer'] = df_all['platform'].map(customer_map)
    df_all.index = df_all['date_only']
    df_all = df_all.sort_values(by='date_time')
    return df_all

@st.cache(allow_output_mutation=True)
def get_all_data(df_45_46,df_200,df_210 ,customer_map):
    features_to_concat = ['station','station_II','platform','line','shift','failed','date_only','date_time','date','year_month','year_week','year_week_date']
    all_data = pd.concat([df_45_46[features_to_concat], df_200[features_to_concat], df_210[features_to_concat]], axis=0)
    all_data['customer'] = all_data['platform'].map(customer_map)
    all_data = all_data.sort_values(by='date')
    return all_data


@st.cache(allow_output_mutation=True)
def plot_1_station_rejects(df_all_fails):
    failed_df = df_all_fails.groupby('station')['line'].count().reset_index().rename(columns={'line':'count'})
    failed_df = failed_df.sort_values(by='count')
    total_rejects = failed_df['count'].sum()
    failed_df['[%] of total NOK'] = failed_df['count']/total_rejects*100
    failed_df['[%] of total NOK'] = failed_df['[%] of total NOK'].round(2)
    return failed_df

@st.cache(allow_output_mutation=True)
def plot_2_pie_chart(df_all_fails):
    pie = df_all_fails.groupby(['station_II','customer', 'platform'])['failed'].count().reset_index()
    return pie

@st.cache(allow_output_mutation=True)
def plot_line_pie_chart(df_all_fails):
    pie = df_all_fails.groupby(['line','shift','customer'])['failed'].count().reset_index()
    return pie

@st.cache(allow_output_mutation=True)
def plot_3_cumsum_stations_overtime(failed_df, df_all_fails, line_plot_freq ,freq):
    stations = failed_df['station'].unique()
    df_nok_total = pd.DataFrame()
    for station in stations:
        df_nok = pd.DataFrame()
        df_station = df_all_fails[(df_all_fails['station'] == station)]
        df_station['ocurr'] = 1
        df_station.index = pd.to_datetime(df_station.index)
        df_nok['NOK_cummulative'] = df_station['ocurr'].resample(line_plot_freq[freq]).sum().cumsum()
        df_nok['station'] = station
        df_nok_total = pd.concat([df_nok_total,df_nok], axis=0)

    return df_nok_total

@st.cache(allow_output_mutation=True)
def plot_4_percent_nok_station(df_45_46, df_200, df_210):
    op45_46_drop = df_45_46.groupby(['station_II','hlt_failed_part'])['platform'].count().reset_index().rename(columns={'platform':'count'})
    op45_46_drop = pd.pivot_table(op45_46_drop, index= 'station_II', columns='hlt_failed_part', values='count')
    op45_46_drop['Station Fail %'] = op45_46_drop[True] / (op45_46_drop[True]+op45_46_drop[False])*100
    op45_46_drop = op45_46_drop.rename(columns={True: 'count'})

    op200_drop = df_200.groupby(['station_II'])['platform'].count().reset_index().rename(columns={'platform':'count'}).sort_values(by='station_II')
    total_nok = op200_drop['count'].sum();    total_ok = len(df_210);
    op200_drop['count_cumsum'] = op200_drop['count'].cumsum().shift(1).fillna(0)
    op200_drop.index = op200_drop['station_II']
    op200_drop.loc['Unspecified_strut_line','count_cumsum'] = 0
    op200_drop['total_run_in_station'] = total_nok+ total_ok
    op200_drop['total_run'] = op200_drop['total_run_in_station'] - op200_drop['count_cumsum']
    op200_drop['Station Fail %'] = op200_drop['count'] / op200_drop['total_run'] *100

    features_drop_concat = ['count','Station Fail %']
    df_nok_perc_station = pd.concat([op45_46_drop[features_drop_concat]  , op200_drop[features_drop_concat]  ], axis=0)
    df_nok_perc_station['Station Fail %']  = df_nok_perc_station['Station Fail %'].round(2)
    df_nok_perc_station['Station'] = df_nok_perc_station.index

    return df_nok_perc_station

@st.cache(allow_output_mutation=True)
def plot_frop_rates(df_dates, freq, to_plot_freq, limit_percent):
    drop_rate = df_dates.groupby([to_plot_freq[freq],'failed'])['customer'].count().reset_index().rename(columns={'customer':'count'})
    drop_rate = pd.pivot_table(drop_rate, values='count', index=to_plot_freq[freq], columns='failed')
    drop_rate = drop_rate.fillna(0)
    drop_rate['fail_%'] = drop_rate.iloc[:,1] / (drop_rate.iloc[:,0] + drop_rate.iloc[:,1])*100
    avg_drop_rate = df_dates['failed'].mean()*100

    fig_drop = px.bar(drop_rate, x=drop_rate.index, y='fail_%' , range_y= [0,limit_percent])
    fig_drop.update_traces(marker_color='#798D98', marker_line_color='rgb(8,48,107)', marker_line_width=1.0, opacity=0.6)
    fig_drop.add_hline(y=avg_drop_rate, line_width=1.0, line_dash="solid", line_color="red")
    annotations = [dict(xref='paper', x=1.13, y=avg_drop_rate, xanchor='right', yanchor='middle', text='o.a' + ' {}%'.format(round(avg_drop_rate,2)),
                            font=dict(family='Arial',size=13), showarrow=False)]
    fig_drop.update_layout(annotations=annotations)
    return fig_drop

@st.cache(allow_output_mutation=True)
def get_customer_percent_rejects(all_data):
    rejects_cust = all_data.groupby(['customer','failed'])['station'].count().reset_index().rename(columns= {'station':'count'})
    rejects_cust = pd.pivot_table(rejects_cust, values='count', index='customer', columns='failed')
    rejects_cust = rejects_cust.fillna(1)
    rejects_cust['fail_%'] = rejects_cust.iloc[:,1] / (rejects_cust.iloc[:,0] + rejects_cust.iloc[:,1])*100
    rejects_cust['fail_%'] = rejects_cust['fail_%'].round(2)
    rejects_cust = rejects_cust[rejects_cust[False]>20]
    rejects_cust = rejects_cust.sort_values(by='fail_%', ascending=True)
    sum_nok = rejects_cust[True].sum()
    rejects_cust['%_Of_Overall_NOK'] = rejects_cust[True] / sum_nok * 100
    rejects_cust['%_Of_Overall_NOK'] = rejects_cust['%_Of_Overall_NOK'].round(2)



    fig_perct_individual = px.bar(rejects_cust, x="fail_%", y=rejects_cust.index, orientation='h', height=300)
    fig_perct_individual.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)


    rejects_cust = rejects_cust.sort_values(by='%_Of_Overall_NOK', ascending=True)
    fig_percent_nok = px.bar(rejects_cust, x="%_Of_Overall_NOK", y=rejects_cust.index, orientation='h', height=300)
    fig_percent_nok.update_traces(marker_color='#ff6666', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    rejects_cust = rejects_cust.rename(columns={False:'OK Production'})
    rejects_cust = rejects_cust.sort_values(by='OK Production', ascending=True)
    fig_ok = px.bar(rejects_cust, x='OK Production', y=rejects_cust.index, orientation='h', height=300)
    fig_ok.update_traces(marker_color='#00cc66', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig_perct_individual,fig_percent_nok, fig_ok

@st.cache(allow_output_mutation=True)
def get_percent_rejects(all_data, y_axis='line'):
    rejects_cust = all_data.groupby([y_axis,'failed'])['station'].count().reset_index().rename(columns= {'station':'count'})
    rejects_cust = pd.pivot_table(rejects_cust, values='count', index=y_axis, columns='failed')
    rejects_cust = rejects_cust.fillna(1)
    rejects_cust['fail_%'] = rejects_cust.iloc[:,1] / (rejects_cust.iloc[:,0] + rejects_cust.iloc[:,1])*100
    rejects_cust['fail_%'] = rejects_cust['fail_%'].round(2)
    rejects_cust = rejects_cust[rejects_cust[False]>20]
    rejects_cust = rejects_cust.sort_values(by='fail_%', ascending=True)
    sum_nok = rejects_cust[True].sum()

    fig_perct_individual = px.bar(rejects_cust, x="fail_%", y=rejects_cust.index, orientation='h', height=300)
    fig_perct_individual.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig_perct_individual

@st.cache(allow_output_mutation=True)
def get_group_bar_chart(all_data, var1='line', var2='shift', marker_color='#308196', limit_percent=30.0):
    rejects_cust = all_data.groupby([var1,var2,'failed'])['station'].count().reset_index().rename(columns= {'station':'count'})
    rejects_cust = pd.pivot_table(rejects_cust, values='count', index=[var1,var2], columns='failed')
    rejects_cust = rejects_cust.fillna(1)
    rejects_cust['fail_%'] = rejects_cust.iloc[:,1] / (rejects_cust.iloc[:,0] + rejects_cust.iloc[:,1])*100
    rejects_cust['fail_%'] = rejects_cust['fail_%'].round(2)
    rejects_cust['shift_2'] = rejects_cust.index
    rejects_cust[var1] = rejects_cust['shift_2'].str[0]
    rejects_cust[var2] = rejects_cust['shift_2'].str[1]

    fig = px.bar(rejects_cust, x=var2, y="fail_%", barmode="group", facet_col=var1, range_y=[0,limit_percent], height=450)
    fig.update_traces(marker_color=marker_color, marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)

    return fig

#@st.cache(allow_output_mutation=True)
#def plot_4_percent_nok_station(df_45_46, df_200, df_210):


to_plot_freq = {'day' : 'date_only', 'week': 'year_week_date', 'month': 'year_month' }
line_plot_freq = {'day' : 'D', 'week': 'W', 'month': 'M' }
customer_map = {'WK RA': 'Stellantis',
                'Ram DT RA': 'Stellantis',
                'Ram DS RA' : 'Stellantis',
                'WL RA' :'Stellantis',
                'WL FA' :'Stellantis',
                'Jeep WK FA' :'Stellantis',
                'Ram DT RA' :'Stellantis',
                'Ram DS FA' :'Stellantis',
                'Ford RA': 'Ford' ,
                'Ford FA': 'Ford',
                'MB FA BR164': 'M. Benz',
                'MB RA BR251': 'M. Benz',
                'MB FA BR251': 'M. Benz',
                'HON RA': 'HONDA',
                'HONDA FA': 'HONDA',
                'Tesla FA Service' : 'Tesla',
                'Tesla FA' : 'Tesla',
                'Tesla RA' : 'Tesla',
                'Dummy': 'Unknown',
                'Unknown' :'Unknown'}

def app():
    with st.sidebar:
        from_date = (datetime.date.today() - DateOffset(months=2)).date()
        st.subheader('From what date to load sql data?')
        d1 = st.date_input("", from_date)

        #date range
        st.write("### Date range to analyze")
        col1, col2 = st.beta_columns(2)
        day_from= col1.date_input("From date", from_date,  min_value=d1)
        day_up_to= col2.date_input("Up to date", datetime.date.today(), min_value=day_from + DateOffset(days=2))

        #freq
        freq = st.selectbox('Select frequency to analyze:',  ('day','week','month'))

    # 200 Data-------------------
    query = "select * from all_op200 where date_time > '%s'" %(str(d1))
    df_200= long_running_function_to_load_data(query = query , db='all_op200')
    df_200 = filter_dates(df_200, str(day_from), str(day_up_to))

    # 210 Data-------------------
    query = "select * from all_op210 where date_time > '%s'" %(str(d1))
    df_210= long_running_function_to_load_data(query = query ,db='all_op210')
    df_210 = filter_dates(df_210, str(day_from), str(day_up_to))

    # 45_46 Data-------------------
    query = "select * from hlt_op45_op46 where date_time > '%s'" %(str(d1))
    df_45_46= long_running_function_to_load_data(query = query ,db='hlt_op45_op46')
    df_45_46 = filter_dates(df_45_46, str(day_from), str(day_up_to))

    #Concatenate--------------
    df_all = concatenate(df_200, df_210, df_45_46,customer_map)
    df_all_fails = df_all[df_all['failed'] == True]
    all_data = get_all_data(df_45_46,df_200,df_210 ,customer_map)

    st.title('Plant Production ')

    st.write("## Select Analysis Level")
    analysis = st.selectbox('',  ("Site 2 Overall",'Line Level',"Station Level"))
    st.markdown("""---""")

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Site 2 Overall                                                                                                                                 /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if analysis == "Site 2 Overall":
        #1th PLot - Plant level -> Total % of rejected parts
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        st.subheader("Overall " +  str(round(all_data['failed'].mean()*100,2)) + '% of production was rejected')
        col1, _ = st.beta_columns(2)
        limit_percent =  col1.slider("Select y_axis (%) range:",  0.0, 100.0, 20.0, 3.0, key=['perct01'])
        fig = plot_frop_rates(all_data, freq, to_plot_freq, limit_percent)
        st.write(fig)

        #2,3 and 4 PLot - Plant level -> rejects per customer
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('Individual Customer % of NOK')
        fig_perct_individual, fig_percent_nok, fig_ok=get_customer_percent_rejects(all_data)
        st.write(fig_perct_individual)

        st.subheader('Customer % of Total NOK')
        st.write(fig_percent_nok)

        st.subheader('OK Production')
        st.write(fig_ok)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Line Level                                                                                                                                     /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    elif analysis == "Line Level":
        #1 d PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.title('Line Level')
        st.subheader('Individual Line % of NOK')
        fig_perct_individual=get_percent_rejects(all_data, 'line')
        st.write(fig_perct_individual)

        #2 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        #Line and Stations percentage
        st.subheader('NOK amount distribution over Line / Shift / Customer')
        pie = plot_line_pie_chart(all_data[all_data['failed']==True])
        fig = px.sunburst(pie, path=['line', 'shift', 'customer'], values='failed', branchvalues ='total',
                          color='line',color_discrete_sequence=px.colors.sequential.Aggrnyl)
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig.update_traces(textinfo="label+percent root", insidetextorientation='horizontal')
        st.write(fig)

        #3 PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.subheader('% NOK per Line/Shift')
        col1, _ = st.beta_columns(2)
        limit_percent =  col1.slider("Select y_axis (%) range:",  0.0, 100.0, 20.0, 3.0, key=['perct02'])
        fig = get_group_bar_chart(all_data, var1='line', var2='shift', marker_color='#308196', limit_percent=limit_percent)
        st.write(fig)

        st.subheader('% NOK per Line/Customer')
        col1, _ = st.beta_columns(2)
        limit_percent =  col1.slider("Select y_axis (%) range:",  0.0, 100.0, 20.0, 3.0, key=['perct03'])
        fig = get_group_bar_chart(all_data, var1='line', var2='customer', marker_color='#008080', limit_percent=limit_percent)
        st.write(fig)

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                                                                                                                                                       /
    #        Station Level                                                                                                                                  /
    #                                                                                                                                                       /
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    elif analysis == "Station Level":
        #7th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        st.title('Station Level')
        #Line and Stations percentage
        st.subheader('NOK amount distribution over Stations')
        pie = plot_2_pie_chart(df_all_fails)
        fig = px.sunburst(pie, path=['station_II','customer', 'platform'], values='failed',
                          color='station_II',color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig.update_traces(textinfo="label+percent root", insidetextorientation='horizontal')
        st.write(fig)

        #8th PLOT
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        st.subheader('Amount of Rejects per Station')
        failed_df = plot_1_station_rejects(df_all_fails)
        fig = px.bar(failed_df, x="count", y='station', orientation='h', height=450)
        fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)


        #9th PLot
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        st.subheader('Cummulative NOK per Station')
        df_nok_total = plot_3_cumsum_stations_overtime(failed_df, df_all_fails, line_plot_freq ,freq)
        fig = px.line(df_nok_total, x=df_nok_total.index, y="NOK_cummulative", color='station')
        fig.update_layout(legend=dict( yanchor="top",  y=0.99,   xanchor="left",   x=0.01  ))
        st.write(fig)


        #10th PLot - Stations percentage of NOK of production
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        st.subheader('Percent of NOK per Stations')
        st.write('[ NOK amount ] / [ Total run in station ] * 100')
        df_nok_perc_station= plot_4_percent_nok_station(df_45_46, df_200, df_210)
        fig = px.bar(df_nok_perc_station, x='Station', y='Station Fail %', orientation='v', height=450)
        fig.update_traces(marker_color='#308196', marker_line_color='rgb(8,48,107)', marker_line_width=0.5,  opacity=0.6)
        st.write(fig)
