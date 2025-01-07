'''
Messing around with the hand timing inputs

'''

import pandas as pd
import streamlit as st
import numpy as np
import rpy2.robjects as ro
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from rpy2.robjects import globalenv
from rpy2.robjects import r, pandas2ri, StrVector
from rpy2.robjects import r,pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects


#Streamlit App building

st.title('RCA Hand Time Monitoring')
user_name = st.text_input('Enter your Username')
password = st.text_input("Enter your Password", type="password")
col1, col2 = st.columns(2)
with col1: 
    start = st.date_input('Start Date', datetime.now() - relativedelta(months=1))
    start_str = start.strftime('%d/%m/%Y')

with col2: 
    end = st.date_input('End Date', datetime.now())
    end_str = end.strftime('%d/%m/%Y')


login = st.button('login')

if login: 

    pandas2ri.activate()

    r_script = """
    library(smartabaseR)
    start_date <- "{start}"
    end_date <- "{end}"

    cat("Start Date as Date object:", start_date, "\\n")
    cat("End Date as Date object:", end_date, "\\n")

    df_sb <- sb_get_event(
    form = "RCA Hand Time Monitoring",
    date_range = c(start_date,end_date),
    url = "https://canadiansport.smartabase.com/csip",
    username = "{username}",
    password = "{password}",
    option = sb_get_event_option(
        cache = TRUE
    )
    )
    df_sb <- as.data.frame(df_sb)


    """
    # Execute the R script
    # Execute the R script
    r_script = r_script.format(start=start_str, end=end_str, username = user_name, password = password) 
    #r(r_script)
    robjects.r(r_script) 

    # Retrieve the R object explicitly in a conversion context
    with localconverter(pandas2ri.converter):
        r_df = r("df_sb")


    _='''
    This section is really just testing the time calculations. Things we want
    - Long term tracking of splits and boat classes
    - Profiling of metrics based on this (not entirely sure what that looks like yet but this is biomech in action for sure)

    '''
    def int_string(value):
        # Extract hours, minutes, seconds, and milliseconds
        #hours = int((value % 100000000) // 1000000)
        minutes = int((value % 1000000) // 10000)
        seconds = int((value % 10000) // 100)
        milliseconds = int(value % 100)
        # Combine into a time-like format (hh:mm:ss.ms)
        time_string = f"{minutes:02}:{seconds:02}.{milliseconds:02}"
        return time_string

    def convert_to_seconds(value):
        # Convert the value to an integer to avoid decimal issues
        value = int(value)
        
        # Extract the components
        minutes = value // 10000  # First digit(s) for minutes
        seconds = (value % 10000) // 100  # Next two digits for seconds
        milliseconds = value % 100  # Last two digits for milliseconds
        
        # Convert to total seconds
        total_seconds = minutes * 60 + seconds + milliseconds / 1000
        return total_seconds

    def seconds_to_mmssmm(seconds):
        # Extract the minutes
        minutes = int(seconds // 60)
        # Extract the remaining seconds
        remaining_seconds = seconds % 60
        # Format as mm:ss.mm
        formatted_time = f"{minutes:02}:{remaining_seconds:05.2f}"  # .2f keeps two decimal places
        return formatted_time

    def time_to_seconds(time_str):
        # Split the string into minutes and seconds
        minutes, seconds = time_str.split(":")
        # Convert minutes and seconds to float and calculate total seconds
        total_seconds = int(minutes) * 60 + float(seconds)
        return total_seconds


    data = r_df
    progs = pd.read_csv('progs.csv')

    cleaned_df = data[data['Start'].notna() & (data['Start'] != "")]
    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df['Start'] = pd.to_numeric(cleaned_df['Start'], errors='coerce')
    cleaned_df['Finish'] = pd.to_numeric(cleaned_df['Finish'], errors='coerce')
    cleaned_df['Duration'] = cleaned_df['Finish']- cleaned_df['Start']
    cleaned_df['Start (c)'] = cleaned_df['Start'].apply(int_string)
    cleaned_df['Finish (c)'] = cleaned_df['Finish'].apply(int_string)
    cleaned_df['Duration (c)'] = cleaned_df['Duration'].apply(int_string)
    cleaned_df['Duration (s)'] = cleaned_df['Duration'].apply(convert_to_seconds)
    cleaned_df['Average Speed'] = cleaned_df['Distance (m)']/cleaned_df['Duration (s)']
    cleaned_df['500 Split'] = (500/cleaned_df['Average Speed']).apply(seconds_to_mmssmm)

    prog_times = []
    for bclass in cleaned_df['Boat Class']: 
        
        prog = progs[progs['Class']==bclass]
        prog_time = prog['Prog'].iloc[0]
        prog_times.append(prog_time)

    cleaned_df['Prog'] = prog_times
    cleaned_df['Prog (s)'] = cleaned_df['Prog'].apply(time_to_seconds)
    cleaned_df['Prog (m/s)'] = 2000/cleaned_df['Prog (s)']
    
    cleaned_df['Distance (m)'] = cleaned_df['Distance (m)'].astype(int)
    cleaned_df['Percentage Prog'] = (cleaned_df['Average Speed']/cleaned_df['Prog (m/s)'])*100
    cleaned_df = cleaned_df.drop(columns = ['user_id', 'form', 'about', 'end_date', 'entered_by_user_id', 'event_id', 'start_time', 'end_time', 'Duration', 'Duration (s)', 'Prog (s)', 'Start', 'Finish'])
        

    styled_df = cleaned_df.style.background_gradient(
        cmap='coolwarm',          # Color scale (blue to red)
        subset=['Percentage Prog'],         # Apply to the 'Score' column
        vmin=75,                  # Minimum value of the color scale
        vmax=100                  # Maximum value of the color scale
    )

    st.dataframe(styled_df)
else: 
    st.header('Please enter User Credentials')