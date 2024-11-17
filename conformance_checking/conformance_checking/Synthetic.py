# import libraries
import pandas as pd
import random
from datetime import timedelta
from pm4py.objects.petri_net.importer import importer as pnml_importer
import pm4py
import numpy as np


def generate_synthetic_log(petri_net, initial_marking, final_marking):
    """
    Generate a synthetic event log from a Petri net and convert it to a Pandas DataFrame.

    Parameters
    ----------
    petri_net : PetriNet
        The Petri net model.
    initial_marking : Marking
        The initial marking of the Petri net.
    final_marking : Marking
        The final marking of the Petri net.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the synthetic event log.
    """
    # Generate synthetic event log from the Petri net
    synth_log = pm4py.sim.play_out(petri_net, initial_marking, final_marking)

    # Convert synthetic log to Pandas DataFrame
    log_data = []
    for trace in synth_log:
        trace_id = trace.attributes['concept:name']  # Access attributes as object properties
        for event in trace:
            log_data.append({
                'trace_id': trace_id,
                'activity': event['concept:name'],  # Event attributes are accessed this way
                'timestamp': event['time:timestamp']
            })

    # Create a DataFrame for the synthetic event log
    synth_log_df = pd.DataFrame(log_data)
    return synth_log_df


def adjust_timestamps(df, trace_column='trace_id', timestamp_column='timestamp',
                      min_increment=2400, max_increment=6000):
    """
    Adjust timestamps within each trace to add realistic time differences while maintaining chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        The event log DataFrame containing traces with activities and timestamps.
    trace_column : str
        The column name representing trace identifiers.
    timestamp_column : str
        The column name representing activity timestamps.
    min_increment : int
        Minimum increment (in seconds) between consecutive activities.
    max_increment : int
        Maximum increment (in seconds) between consecutive activities.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with adjusted timestamps.
    """
    # Copy the DataFrame to avoid modifying the original data
    df = df.copy()

    # Convert timestamps to pandas datetime if not already
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)

    # Group by trace and process each trace
    for trace_id, group in df.groupby(trace_column):
        # Sort the group by timestamp to ensure order
        group = group.sort_values(by=timestamp_column)
        adjusted_timestamps = []
        current_time = group.iloc[0][timestamp_column]  # Start with the first timestamp

        # Generate adjusted timestamps for the trace
        for _ in range(len(group)):
            adjusted_timestamps.append(current_time)
            # Increment the current time by a random value within the range
            increment = timedelta(seconds=random.randint(min_increment, max_increment))
            current_time += increment

        # Update the original DataFrame with adjusted timestamps
        df.loc[group.index, timestamp_column] = adjusted_timestamps

    return df


def introduce_synthetic_deviations_W(log_df: pd.DataFrame, activities_to_skip: list,
                                     deviation_ratio: float = 0.3,
                                     time_reduction_range: tuple = (3600, 10800),
                                     seed: int = 42) -> pd.DataFrame:
    """
    Introduces synthetic deviations by skipping user-specified activities in a subset of traces.

    Parameters
    ----------
    log_df : pd.DataFrame
        The synthetic event log as a DataFrame with columns ['trace_id', 'activity', 'timestamp'].
    activities_to_skip : list
        List of activities to skip (remove) from traces.
    deviation_ratio : float, optional
        The ratio of traces containing the specified activity to modify (default is 0.3, i.e., 30%).
    time_reduction_range : tuple, optional
        Range of time (in seconds) to reduce for subsequent events (default is (60, 3600)).
    seed : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    pd.DataFrame
        A modified event log with synthetic deviations introduced.
    """
    np.random.seed(seed)
    modified_log = log_df.copy()

    for activity in activities_to_skip:
        # Identify traces containing the activity
        traces_with_activity = modified_log[modified_log['activity'] == activity]['trace_id'].unique()

        # Randomly select traces to modify based on the deviation ratio
        num_traces_to_modify = int(len(traces_with_activity) * deviation_ratio)
        traces_to_modify = np.random.choice(traces_with_activity, num_traces_to_modify, replace=False)

        for trace_id in traces_to_modify:
            # Filter the trace
            trace = modified_log[modified_log['trace_id'] == trace_id].copy()

            # Find indices of activities to be removed
            indices_to_remove = trace[trace['activity'] == activity].index

            # Determine the first affected index
            first_affected_index = indices_to_remove.min()

            # Remove all occurrences of the activity
            trace = trace.drop(indices_to_remove)

            # Generate a random time reduction in seconds
            time_reduction = pd.Timedelta(seconds=np.random.uniform(*time_reduction_range))

            # Adjust timestamps for activities after the last removed activity
            if not trace.empty and first_affected_index is not None:
                # Identify indices of events following the removed activities
                subsequent_indices = trace.index[trace.index > first_affected_index]

                # Adjust their timestamps
                trace.loc[subsequent_indices, 'timestamp'] -= time_reduction

            # Replace the trace in the modified log
            modified_log = modified_log[modified_log['trace_id'] != trace_id]
            modified_log = pd.concat([modified_log, trace])

    # Sort the modified log by trace_id and timestamp
    modified_log = modified_log.sort_values(by=['trace_id', 'timestamp']).reset_index(drop=True)
    return modified_log

def map_alignments_with_trace_ids(log_df, alignments):
    """
    Map alignments to their respective trace IDs from the event log.

    Parameters
    ----------
    log_df : pd.DataFrame
        Event log as a DataFrame containing 'case:concept:name'.
    alignments : List
        List of alignment dictionaries from the conformance checking.

    Returns
    -------
    List[Dict[str, Any]]
        A list where each element contains:
            - 'trace_id': The trace ID
            - 'alignment': The corresponding alignment
    """
    # Ensure trace IDs are unique and in order
    unique_trace_ids = log_df['case:concept:name'].unique()

    if len(unique_trace_ids) != len(alignments):
        raise ValueError("Mismatch between the number of traces in the log and alignments!")

    # Map alignments to trace IDs
    alignments_with_trace_ids = [
        {"trace_id": trace_id, "alignment": alignment}
        for trace_id, alignment in zip(unique_trace_ids, alignments)
    ]

    return alignments_with_trace_ids

def generate_trace_encoding(alignments_with_trace_ids):
    """
    Generate trace encodings and associate them with trace IDs.

    Parameters
    ----------
    alignments_with_trace_ids : List[Dict[str, Any]]
        A list where each element contains a trace ID and its corresponding alignment.

    Returns
    -------
    List[Dict[str, Any]]
        A list where each element includes a trace ID and its move counts.
    """
    trace_encodings = []

    for item in alignments_with_trace_ids:
        trace_id = item['trace_id']
        alignment = item['alignment']

        # Initialize move count for the trace
        move_count = {}
        for log_activity, model_activity in alignment['alignment']:
            if log_activity != ">>" and log_activity not in move_count:
                move_count[log_activity] = {'log_moves': 0, 'model_moves': 0}
            elif model_activity != ">>" and model_activity not in move_count:
                move_count[model_activity] = {'log_moves': 0, 'model_moves': 0}

            if log_activity == ">>":
                move_count[model_activity]['model_moves'] += 1
            elif model_activity == ">>":
                move_count[log_activity]['log_moves'] += 1

        trace_encodings.append({'trace_id': trace_id, 'move_count': move_count})

    return trace_encodings

def make_dataframe_for_decision_tree_with_throughput(log_df, trace_encodings, save_path):
    """
    Create a DataFrame for decision tree analysis from trace encodings and throughput times.

    Parameters
    ----------
    log_df : pd.DataFrame
        The synthetic event log as a DataFrame.
    trace_encodings : List[Dict[str, Any]]
        A list of trace encodings with their IDs.
    save_path : str
        The path to save the resulting CSV file.

    Returns
    -------
    None
    """
    # Calculate throughput time for each trace
    throughput_times = log_df.groupby('case:concept:name').agg(
        first_event=('time:timestamp', 'min'),
        last_event=('time:timestamp', 'max')
    )
    throughput_times['case:throughput_time'] = throughput_times['last_event'] - throughput_times['first_event']
    throughput_times = throughput_times[['case:throughput_time']].reset_index()

    # Flatten trace encodings and include trace IDs
    flattened_traces = []
    for item in trace_encodings:
        trace_id = item['trace_id']
        move_count = item['move_count']

        flat_trace = {'trace_id': trace_id}
        for activity, moves in move_count.items():
            if activity is not None:
                activity = activity.replace(' ', '_')
            flat_trace[f'{activity}_log_moves'] = moves['log_moves']
            flat_trace[f'{activity}_model_moves'] = moves['model_moves']
        flattened_traces.append(flat_trace)

    df_flattened = pd.DataFrame(flattened_traces)

    # Merge with throughput time
    df = pd.merge(df_flattened, throughput_times, left_on='trace_id', right_on='case:concept:name')
    df = df.drop('case:concept:name', axis=1)

    # Save to CSV
    df.to_csv(save_path, index=False)
