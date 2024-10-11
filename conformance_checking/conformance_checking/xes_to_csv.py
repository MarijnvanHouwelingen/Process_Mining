import pandas as pd
import pm4py 


def xes_to_csv(xes_file_path: str, output_file_path : str) -> None:
    """
    This function converts a xes file to csv and calculates the throughput time for each event.

    Parameters
    ----------
    :xes_file_path (str): The filepath of the xes file (.xes)
    :output_file_path (str): The filepath of the csv file (.csv)
    
    Returns
    ----------
    None
    """
    # Load the XES file
    log = pm4py.read_xes(xes_file_path)

    # Create the DataFrame
    df = pd.DataFrame(log)

    print(df)

    # Step 1: Convert timestamp column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['time:timestamp']):
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    
    # Group the dataframe by 'trace_id' and calculate the throughput time
    throughput_times = df.groupby('case:concept:name').agg(
        first_event=('time:timestamp', 'min'),
        last_event=('time:timestamp', 'max')
    )
    # Calculate the throughput time as the difference between last and first event
    throughput_times['throughput_time'] = throughput_times['last_event'] - throughput_times['first_event']

    # Merge the throughput times back into the original dataframe
    df = df.merge(throughput_times[['throughput_time']], on='case:concept:name', how='left')
    # Display the DataFrame
    df.to_csv(output_file_path)

if __name__ == "__main__":
    xes_to_csv("data/BPI2017Denied(3).xes","data/BPI2017Denied(3).csv")

