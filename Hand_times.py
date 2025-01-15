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


#Streamlit App building
image, title = st.columns([1,9])
with image: 
    st.image('https://images.squarespace-cdn.com/content/v1/5f63780dbc8d16716cca706a/1604523297465-6BAIW9AOVGRI7PARBCH3/rowing-canada-new-logo.jpg')
with title:
    st.title('RCA Hand Time Monitoring')

_='''
Lots of fancy code to get into the API

'''

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
            #start_date = thirty_days_ago.strftime("%d/%m/%Y")
            #end_date = today.strftime("%d/%m/%Y")
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

                def process_rows_column(row):
                    flattened_rows = {}
                    for row_entry in row:  
                        for pair in row_entry.get("pairs", []):  
                            key = pair.get("key", "Unknown Key")
                            value = pair.get("value", "Unknown Value")
                            flattened_rows[key] = value
                    return flattened_rows
                expanded_rows = pulled_data['rows'].apply(process_rows_column)
                expanded_rows_df = pd.DataFrame(expanded_rows.tolist())
                data_cleaned = pd.concat([pulled_data.drop(columns=['rows']), expanded_rows_df], axis=1)
                st.session_state.r_df = data_cleaned
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

    cleaned_df = data[data['Start'].notna() & (data['Start'] != "")]
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
        max_prog_day = np.max(date_df['Percentage Prog'])
        date_df['win_diff'] = max_prog_day - date_df['Percentage Prog']
        appended_dfs.append(date_df)

    # Concatenate all the dataframes in the list into a single dataframe
    final_df = pd.concat(appended_dfs, ignore_index=True) 
    cleaned_df = final_df

    
    cleaned_df = cleaned_df.drop(columns = ['userId', 'formName', 'finishDate', 'finishTime', 'enteredByUserId', 'startTime', 'Duration', 'Duration (s)', 'Prog (s)', 'Start', 'Finish'])
        

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

    # Scatter plot for Average Speed
    fig.add_trace(go.Scatter(
        y=athlete_data['Average Speed'],
        x=athlete_data['startDate'],
        mode='markers+lines',
        name='Speed Over Time',
        line=dict(color='green'),
        hovertemplate='Split: %{customdata[0]}<extra></extra>',
    ))
    fig.data[-1].customdata = athlete_data[['500 Split']].to_numpy()

    # Scatter plot for Prog Speed
    fig.add_trace(go.Scatter(
        y=athlete_data['Prog (m/s)'],
        x=athlete_data['startDate'],
        mode='markers+lines',
        name='Prog Speed',
        line=dict(color='gold'),
    ))

    # Bar chart for win_diff on secondary y-axis
    fig.add_trace(go.Bar(
        y=athlete_data['win_diff'],
        x=athlete_data['startDate'],
        name='Win Difference',
        yaxis='y2',  # Assign to secondary y-axis
        opacity = 0.4, 
            marker=dict(
        color=athlete_data['win_diff'],  # Set color based on the data values
        colorscale='RdBu',  # Red-Blue color scale
        cmin=0,  # Minimum value for the color scale
        cmax=10,  # Maximum value for the color scale
        #colorbar=dict(title='Win Diff')  # Add a colorbar for reference
    )
    ))

    # Update layout to include secondary y-axis
    fig.update_layout(
        yaxis=dict(
            title="Speed (m/s)",
            range=[0, np.max(athlete_data['Prog (m/s)']) + 1.5],
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



    col3, col4 = st.columns([7,3])
    with col3:
        st.plotly_chart(fig)
    with col4: 
        st.image(headshots[headshots['Name']==athlete]['Link'].iloc[0])

    col5,col6, col7 = st.columns(3)
    with col5: 
        st.metric('Average Prog', round(np.mean(athlete_data['Percentage Prog']),2), delta = round(np.mean(athlete_data['Percentage Prog'])-100,2))
    with col6: 
        st.metric('Average Win % Diff', round(np.mean(athlete_data['win_diff']),1))

    #athlete_data = athlete_data.drop(columns = ['first_name', 'last_name', 'Prog (m/s)'])
    athlete_df = athlete_data.style.background_gradient(
            cmap='coolwarm',          # Color scale (blue to red)
            subset=['Percentage Prog'],         # Apply to the 'Score' column
            vmin=75,                  # Minimum value of the color scale
            vmax=100                  # Maximum value of the color scale
        )
    
    st.dataframe(athlete_df)
    
 
