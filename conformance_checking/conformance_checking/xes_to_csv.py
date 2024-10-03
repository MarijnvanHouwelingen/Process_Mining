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
    
    # Step 2: Group by 'case_id' and calculate time difference between consecutive events
    df['time:throughput time'] = df.groupby('case:concept:name')['time:timestamp'].diff()
    
    # Display the DataFrame
    df.to_csv(output_file_path)

if __name__ == "__main__":
    xes_to_csv("data/BPI2017Denied(3).xes","data/BPI2017Denied(3).csv")

