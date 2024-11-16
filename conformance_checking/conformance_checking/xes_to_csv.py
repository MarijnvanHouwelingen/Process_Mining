import pandas as pd
import pm4py 


def throughput_time_to_xes(xes_file_path: str, output_csv_path:str ,output_xes_path : str) -> None:
    """
    This function converts a xes file to xes and calculates the throughput time for each trace.

    Parameters
    ----------
    :xes_file_path (str): The filepath of the xes file (.xes)
    :output_csv_path (str): The filepath of the csv file (.csv)
    :output_xes_path (str): The filepath of the cleaned xes file (.xes)
    
    Returns
    ----------
    None
    """
    # Load the XES file
    raw_log = pm4py.read_xes(xes_file_path)

    # Create the DataFrame
    df = pd.DataFrame(raw_log)

    # Convert the timestamp to a pandas datetime object for time manipulation.
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    
    # Group the dataframe by 'trace_id' and calculate the throughput time
    throughput_times = df.groupby('case:concept:name').agg(
        first_event=('time:timestamp', 'min'),
        last_event=('time:timestamp', 'max')
    )
    # Calculate the throughput time as the difference between last and first event
    throughput_times['case:throughput_time'] = throughput_times['last_event'] - throughput_times['first_event']

    # Merge the throughput times back into the original dataframe and name it as a trace attribute for pm4py
    df = df.merge(throughput_times[['case:throughput_time']], on='case:concept:name', how='left')
    
    # Convert timestamp to ISO without localization.
    df['time:timestamp'] = df['time:timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z')
    
    # Write it to csv
    df.to_csv(output_csv_path)

    # Read in the new CSV
    df_from_csv = pd.read_csv(output_csv_path)

    # Convert the pandas DataFrame to a pm4py Event Log
    event_log = pm4py.format_dataframe(df_from_csv, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

    # Export the event log to XES format
    pm4py.write_xes(event_log, output_xes_path)

    print("Event log has been successfully exported to XES format!")

    
if __name__ == "__main__": #### Dit gaat weg MORGEN ####
    throughput_time_to_xes("data/BPI2017Denied(3).xes","data/BPI2017Denied(3).csv","data/BPI2017Denied(3)_Throughput.xes")
    