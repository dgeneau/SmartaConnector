'''
Messing around with the hand timing inputs

'''

import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import requests
import json
from collections import defaultdict
from typing import List, Dict, Union


st.set_page_config(layout="wide")

#Streamlit App building
image, title = st.columns([1,9])
with image: 
    st.image('https://images.squarespace-cdn.com/content/v1/5f63780dbc8d16716cca706a/1604523297465-6BAIW9AOVGRI7PARBCH3/rowing-canada-new-logo.jpg')
with title:
    st.title('RCA Hand Time Monitoring')

_='''
Lots of fancy code to get into the API

'''

def expand_dataframe_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands a DataFrame by creating new rows from lists in columns, maintaining alignment
    between corresponding list elements.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame where some columns may contain lists
        
    Returns:
    --------
    pd.DataFrame
        Expanded DataFrame where lists are converted to individual rows
    """
    
    def is_list_like(x):
        return isinstance(x, (list, tuple, pd.Series))
    
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Identify columns containing lists
    list_columns = []
    for col in df.columns:
        if df[col].apply(is_list_like).any():
            list_columns.append(col)
    
    if not list_columns:
        return result_df
    
    # Function to standardize non-list values into lists
    def standardize_to_list(val: Union[List, str, int, float], length: int) -> List:
        if is_list_like(val):
            return list(val)
        return [val] * length
    
    # Process each row
    expanded_rows = []
    
    for idx, row in df.iterrows():
        # Find the maximum list length in this row
        max_length = 1
        for col in list_columns:
            if is_list_like(row[col]):
                max_length = max(max_length, len(row[col]))
        
        # Create expanded rows
        for i in range(max_length):
            new_row = {}
            for col in df.columns:
                if col in list_columns:
                    standardized_list = standardize_to_list(row[col], max_length)
                    new_row[col] = standardized_list[i] if i < len(standardized_list) else None
                else:
                    new_row[col] = row[col]
            expanded_rows.append(new_row)
    
    # Create new DataFrame from expanded rows
    return pd.DataFrame(expanded_rows).reset_index(drop=True)

class AMSClient:
    def __init__(self, base_url, app_name, app_id=None):
        """
        Initialize the AMS API client
        
        Parameters:
        base_url (str): Base URL for the API (replace {{amsURL}})
        app_name (str): Your AMS application name
        app_id (str, optional): Your X-APP-ID
        """
        self.base_url = base_url.rstrip('/')
        self.app_name = app_name
        self.session = None
        
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        if app_id:
            self.headers['X-APP-ID'] = app_id

    def login(self, username, password):
        """
        Authenticate with the AMS API using username and password
        """
        url = f"{self.base_url}/api/v2/user/loginUser"
        
        payload = {
            "username": username,
            "password": password,
            "loginProperties": {
                "appName": self.app_name,
                "clientTime": datetime.now().strftime("%Y-%m-%dT%H:%M")
            }
        }

        try:
            response = requests.post(
                url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            
            session_id = response.cookies.get('JSESSIONID')
            if not session_id:
                raise ValueError("No session ID received in response")
            
            self.headers.update({
                'session-header': session_id,
                'Cookie': f'JSESSIONID={session_id}'
            })
            
            self.session = session_id
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Login failed: {e}")
            return False

    def fetch_event_forms(self, form_names, start_date=None, end_date=None, user_ids = None):
        """
        Fetch event forms from the API
        
        Parameters:
        form_names (list): List of form names to fetch
        start_date (str): Start date in dd/mm/yyyy format
        end_date (str): End date in dd/mm/yyyy format
        """
        if not self.session:
            raise ValueError("Not authenticated. Call login() first.")

        url = f"{self.base_url}/api/v1/eventsearch"
        
        payload = {
            "formNames": form_names
        }
        
        # Add dates if provided
        if start_date:
            payload["startDate"] = start_date
        if end_date:
            payload["finishDate"] = end_date
        
        payload["userIds"] =  user_ids
        try:
            print(f"\nMaking request to: {url}")
            print(f"Payload: {payload}")
            
            response = requests.post(
                url,
                headers=self.headers,
                data=json.dumps(payload),
                params={'informat': 'json', 'format': 'json'}
            )
            
            print(f"Response Status Code: {response.status_code}")
            print(f"Raw Response Text: {response.text}")
            
            response.raise_for_status()
            
            if response.text:
                return response.json()
            else:
                print("Received empty response from server")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching event forms: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None

    def fetch_group_athletes(self):
        """
        Fetch athlete groups and their IDs
        
        Returns:
        dict: JSON response containing athlete group information
        """
        if not self.session:
            raise ValueError("Not authenticated. Call login() first.")

        url = f"{self.base_url}/api/v3/group-athletes"
        
        try:
            print(f"\nFetching athlete groups from: {url}")
            
            response = requests.get(
                url,
                headers=self.headers
            )
            
            print(f"Response Status Code: {response.status_code}")
            print(f"Raw Response Text: {response.text}")
            
            response.raise_for_status()
            
            if response.text:
                return response.json()
            else:
                print("Received empty response from server")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching athlete groups: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None


# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.start_str = ""
    st.session_state.end_str = ""
    st.session_state.groups_data = None
    st.session_state.r_df = pd.DataFrame()

if not st.session_state.logged_in:
    # Login form
    user_name = st.text_input('Enter your Username')
    password = st.text_input("Enter your Password", type="password")
    login = st.button('Login')
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input('Start Date', datetime.now() - relativedelta(months=1))
        st.session_state.start_str = start.strftime('%d/%m/%Y')

    with col2:
        end = st.date_input('End Date', datetime.now())
        st.session_state.end_str = end.strftime('%d/%m/%Y')

    if login:
        st.session_state.username = user_name
        st.session_state.password = password
        st.session_state.logged_in = True
        st.success("Logged in successfully!")
else:
    st.write(f"Logged in as: {st.session_state.username}")

    fetch_data = True
    if fetch_data:
        
        CONFIG = {
            "base_url": "https://canadiansport.smartabase.com/csip",    # Replace {{amsURL}}
            "app_name": "csip",    # Replace {{amsAppName}}
            "username": st.session_state.username,     # Replace {{username}}
            "password": st.session_state.password      # Replace {{password}}
        }
        
        # Initialize client
        client = AMSClient(
            base_url=CONFIG["base_url"],
            app_name=CONFIG["app_name"]
        )
        
        # Login
        if client.login(CONFIG["username"], CONFIG["password"]):
            st.write("Yes Successfully logged in")
            
            if st.session_state.groups_data is None:
                st.write('Pulling Athletes')
                groups_data = client.fetch_group_athletes()
                if groups_data:
                    st.write("Athlete groups fetched successfully")
                    names = pd.DataFrame(groups_data['athleteMap']).T

                    groups_data = pd.DataFrame(groups_data['groups'])
                    men = groups_data[groups_data['name']== 'Rowing Men NTC']
                    women = groups_data[groups_data['name']== 'Rowing Women NTC']
                    all = pd.concat([men, women], ignore_index = True, axis =0)
                    all_id = all['athletes'].explode().tolist()

                    names = names[names['id'].isin(all_id)]
                    names = names.drop(columns = ['avatarId'])
                    names = names.reset_index(drop = True)
                    st.session_state.groups_data = names
            
            
            id_data = st.session_state.groups_data 
            # Get today's date and 30 days ago
            today = datetime.now()
            thirty_days_ago = today - timedelta(days=30)
            
            # Format dates as dd/mm/yyyy

            start_date = st.session_state.start_str
            end_date = st.session_state.end_str

            # Example event form request with date range
            response_data = client.fetch_event_forms(
                form_names=['RCA Hand Time Monitoring'],
                start_date=start_date,
                end_date=end_date, 
                user_ids=id_data['id'].tolist() 
            )
            
            if response_data:
                
                json = json.dumps(response_data, indent=2)
                pulled_data = pd.DataFrame(response_data['events'])
                
                def transform_rowing_data(pulled_data: Union[str, Dict]) -> pd.DataFrame:
                    _="""
                    Transform nested JSON rowing data into a flat DataFrame.
                    
                    Parameters:
                    -----------
                    pulled_data : Union[str, Dict]
                        Input data either as a JSON string or dictionary containing rowing race information
                    
                    Returns:
                    --------
                    pd.DataFrame
                        Cleaned and structured DataFrame with race information
                    """
                    # Convert string to dictionary if necessary
                    if isinstance(pulled_data, str):
                        json_data = json.loads(pulled_data)
                    else:
                        json_data = pulled_data
                    
                    # Initialize list to store flattened records
                    records = []
                    
                    # Process each event
                    for event in json_data['events']:
                        # Base information common to all rows
                        base_info = {
                            'formName': event['formName'],
                            'startDate': datetime.strptime(event['startDate'], '%d/%m/%Y').strftime('%Y-%m-%d'),
                            'startTime': event['startTime'],
                            'finishDate': datetime.strptime(event['finishDate'], '%d/%m/%Y').strftime('%Y-%m-%d'),
                            'finishTime': event['finishTime'],
                            'userId': event['userId'],
                            'enteredByUserId': event['enteredByUserId'],
                            'event_id': event['id']
                        }
                        
                        # Process each row in the event
                        for row in event['rows']:
                            record = base_info.copy()
                            
                            # Convert pairs to dictionary for easier access
                            pairs_dict = {pair['key']: pair['value'] for pair in row['pairs']}
                            
                            # Add all available fields
                            record.update({
                                'Race': pairs_dict.get('Race', '1'),
                                'Boat Number': pairs_dict.get('Boat Number', '1.0'),
                                'Boat Class': pairs_dict.get('Boat Class', None),
                                'Distance (m)': float(pairs_dict.get('Distance (m)', 0)),
                                'Start': float(pairs_dict.get('Start', 0)),
                                'Finish': float(pairs_dict.get('Finish', 0)),
                                'row_number': row['row']
                            })
                            
                            
                            records.append(record)
                    
                    # Create DataFrame
                    df = pd.DataFrame(records)
                    return df
                
                test = transform_rowing_data(response_data)
                
                st.session_state.r_df = test
            else:
                st.write("\nNo data returned from API")
        else:
            st.write("Login failed")


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

    def convert_to_time_object(hhmmss):
        """
        Converts an integer in the format HHMMSS to a timedelta object.
        """
        hours = hhmmss // 10000
        minutes = (hhmmss // 100) % 100
        seconds = hhmmss % 100

        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    def subtract_time(row):
        """
        Subtracts two time columns in HHMMSS format from a DataFrame row.
        """
        time1 = convert_to_time_object(row['Start'])
        time2 = convert_to_time_object(row['Finish'])
        
        difference = time1 - time2

        # Handle negative differences
        if difference.total_seconds() < 0:
            difference = -difference

        total_seconds = int(difference.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return hours * 10000 + minutes * 100 + seconds


    

    data = st.session_state.r_df
    progs = pd.read_csv('progs.csv')
    headshots = pd.read_csv('headshots.csv')

    cleaned_df = data[(data['Finish'] != 0)]
    #cleaned_df = data[(data['Finish'] != 0) & (data['Start'] != 0)]
    cleaned_df['Distance (m)'] = pd.to_numeric(cleaned_df['Distance (m)'])
    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df['Start'] = pd.to_numeric(cleaned_df['Start'], errors='coerce')
    cleaned_df['Finish'] = pd.to_numeric(cleaned_df['Finish'], errors='coerce')
    cleaned_df['Duration'] = cleaned_df.apply(subtract_time, axis=1)
    cleaned_df['Start (c)'] = cleaned_df['Start'].apply(int_string)
    cleaned_df['Finish (c)'] = cleaned_df['Finish'].apply(int_string)
    cleaned_df['Duration (c)'] = cleaned_df['Duration'].apply(int_string)
    cleaned_df['Duration (s)'] = cleaned_df['Duration'].apply(convert_to_seconds)
    cleaned_df['Average Speed'] = cleaned_df['Distance (m)']/cleaned_df['Duration (s)']
    cleaned_df['500 Split'] = (500/cleaned_df['Average Speed']).apply(seconds_to_mmssmm)
 

    ID_df = st.session_state.groups_data

    merged_data = pd.merge(cleaned_df, ID_df, how='left', left_on ='userId', right_on = 'id')
    cols_to_drop = [col for col in merged_data.columns if col.startswith("id")]
    merged_data.drop(columns=cols_to_drop, inplace=True) 
    merged_data['about'] = merged_data['firstname'] + " " + merged_data['lastname'] 
    cleaned_df = merged_data
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
    

    appended_dfs = []
    for date in cleaned_df['startDate'].unique():
        date_df = cleaned_df[cleaned_df['startDate']==date]
        for race in date_df['Race'].unique():
            race_df = date_df[date_df['Race'] == race]
            max_prog_day = np.max(race_df['Percentage Prog'])
            race_df['win_diff'] = max_prog_day - race_df['Percentage Prog']
            appended_dfs.append(race_df)

    # Concatenate all the dataframes in the list into a single dataframe
    cleaned_df = pd.concat(appended_dfs, ignore_index=True) 
    cleaned_df = cleaned_df.drop(columns = ['userId', 'formName', 'finishDate', 'finishTime', 'enteredByUserId', 'startTime', 'Duration', 'Duration (s)', 'Prog (s)', 'Start', 'Finish'])

    analysis  = st.sidebar.selectbox('Select Analysis', ['Athlete', 'Day Report'])

    if analysis == 'Athlete':     

        styled_df = cleaned_df.style.background_gradient(
            cmap='coolwarm',          # Color scale (blue to red)
            subset=['Percentage Prog'],  # Apply to the 'Score' column
            vmin=75,                  # Minimum value of the color scale
            vmax=100                  # Maximum value of the color scale
        )


        athlete = st.selectbox('Select Athlete to Analyze',
                    cleaned_df['about'].unique())
        
        athlete_data = cleaned_df[cleaned_df['about']==athlete]
        
        fig = go.Figure()
        _='''
        for date in athlete_data['startDate'].unique():
            #if len(athlete_data['Race'][athlete_data['startDate']==date])>1:
            for race in athlete_data['Race'][athlete_data['startDate']==date]:
                subset = athlete_data[athlete_data['Race'] == race]
                fig.add_trace(go.Bar(
                    x=subset['startDate'],
                    y=subset['win_diff'],
                    #name=f'Win Diff: {race}', 
                    yaxis = 'y2',
                    showlegend= False,
                    marker=dict(
                        color=subset['win_diff'],
                        colorscale='RdBu',
                        cmin=0,
                        cmax=10),
                        opacity=0.6
                ))

        

        fig.add_trace(go.Bar(
            x=athlete_data['startDate'],
            y=athlete_data['win_diff'],
            name='Win Difference',
            opacity=0.4,
            marker=dict(
                color=athlete_data['win_diff'],
                colorscale='RdBu',
                cmin=0,
                cmax=10,
                # colorbar=dict(title='Win Diff')  # Uncomment if you want a color bar
            ),
            # Remove mode='markers' since it's not applicable to Bar
            yaxis='y2' 
        ))
        '''
        # Scatter plot for Average Speed
        fig.add_trace(go.Scatter(
            y=athlete_data['Average Speed'],
            x=athlete_data['startDate'],
            mode='markers',
            name='Session Speed',
            line=dict(color='red'),
            hovertemplate='Split: %{customdata[0]}<extra></extra>',
        ))
        fig.data[-1].customdata = athlete_data[['500 Split']].to_numpy()
        
        dates = []
        speed_avgs = []
        for date in athlete_data['startDate'].unique():
            dates.append(date)
            speed = athlete_data['Average Speed'][athlete_data['startDate'] == date].mean()
            speed_avgs.append(speed)

        fig.add_trace(go.Scatter(
            y=speed_avgs,
            x=dates,
            mode='lines',
            name='Speed Trend',
            line=dict(color='red'),
        ))
        # Scatter plot for Prog Speed
        fig.add_trace(go.Scatter(
            y=athlete_data['Prog (m/s)'],
            x=athlete_data['startDate'],
            mode='markers',
            name='Prog Speed',
            marker=dict(color='gold',
                        size = 10),
        ))

        # Bar chart for win_diff on secondary y-axis
    

        # Update layout to include secondary y-axis
        fig.update_layout(
            barmode = 'group',
            yaxis=dict(
                title="Speed (m/s)",
                range=[np.min(athlete_data['Average Speed'])-1, np.max(athlete_data['Prog (m/s)']) + 1.5],
            ),
            yaxis2=dict(
                title="Win Difference (%)",
                overlaying='y',
                side='right',  # Secondary axis on the right side
                showgrid = False, 
                range = [0,20]
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            )
        )


        col3, col4 = st.columns([7.5,2.5])
        with col3:
            st.plotly_chart(fig)
        with col4: 
            st.image(headshots[headshots['Name']==athlete]['Link'].iloc[0])

        col5,col6, col7 = st.columns(3)
        with col5: 
            st.metric('Average Prog', round(np.mean(athlete_data['Percentage Prog']),2), delta = round(np.mean(athlete_data['Percentage Prog'])-100,2))
        with col6: 
            st.metric('Average Win % Diff', round(np.mean(athlete_data['win_diff']),1))



        athlete_data = athlete_data.drop(columns = ['Prog (m/s)','event_id', 'row_number', 'Prog (m/s)', 'firstname', 'lastname', 'about'])
        athlete_df = athlete_data.style.background_gradient(
                cmap='coolwarm',          # Color scale (blue to red)
                subset=['Percentage Prog'],         # Apply to the 'Score' column
                vmin=70,                  # Minimum value of the color scale
                vmax=100                  # Maximum value of the color scale
            )
        

        st.dataframe(athlete_df)
    elif analysis == 'Day Report':
        
        day = st.selectbox('Select Date for Analyis', cleaned_df['startDate'].unique() , placeholder="Choose a Date") 
        day_data = cleaned_df[cleaned_df['startDate']==day]

        day_data['crew'] = day_data.groupby('win_diff')['about'].transform(lambda x: ', '.join(x))
        
        day_data = day_data.drop(columns = ['event_id', 'row_number', 'firstname', 'lastname', 'about'])
        day_data.drop_duplicates(keep='first', inplace=True)

        cols = day_data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        day_data = day_data[cols]
        day_data = day_data.reset_index(drop=True)

        display_df = day_data.style.background_gradient(
                cmap='coolwarm',          # Color scale (blue to red)
                subset=['Percentage Prog'],    # Apply to the 'Score' column
                vmin=70,                  # Minimum value of the color scale
                vmax=100                  # Maximum value of the color scale
            )

        st.dataframe(display_df, height=35*len(day_data)+38)

        
        
 
