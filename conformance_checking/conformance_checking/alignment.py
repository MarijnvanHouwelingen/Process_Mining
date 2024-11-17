import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as petri_visualizer
from typing import Tuple, List, Dict, Optional
import pickle
import pandas as pd

def view_event_log_petrinet(xes_file_path: str):
    """
    This function visualizes a PetriNet from an xes event log.

    Parameters
    ----------
    :xes_file_path (str): The filepath of the xes file (.xes)
    
    Returns
    ----------
    None
    """
    # Load the XES file
    log = pm4py.read_xes(xes_file_path)

    # Discover a Petri net using the Heuristics Miner
    PetriNet, iMarking, fMarking = pm4py.discover_petri_net_heuristics(log)
    
    # Show the petri net using Graphviz
    pm4py.view_petri_net(PetriNet, iMarking, fMarking)

def load_pnml(pnml_file_path: str) -> Tuple[pm4py.PetriNet, pm4py.Marking, pm4py.Marking]:
    """
    This function loads and visualizes a PetriNet from an Petrinet (.pnml) file.

    Parameters
    ----------
    :pnml_file_path (str): The filepath of the pnml file (.pnml)
    
    Returns
    ----------
    :Tuple[pm4py.PetriNet, pm4py.Marking, pm4py.Marking]: A tuple with the PetriNet object, initial marking and final marking.
    """
    # Load the PNML file
    petri_net, initial_marking, final_marking = pnml_importer.apply(pnml_file_path)

    # Visualize the Petri net
    gviz = petri_visualizer.apply(petri_net, initial_marking, final_marking)
    petri_visualizer.view(gviz)

    return petri_net, initial_marking, final_marking

def create_alignment(xes_file_path: str, PetriNet:pm4py.PetriNet, initial_marking:pm4py.Marking, final_marking:pm4py.Marking, parameters: dict = None):
    """
    This function applies a alignment algorithm on a PetriNet and event log. Creating an alignment Dictionairy.

    Parameters
    ----------
    :xes_file_path (str): The filepath of the xes file (event log file).
    :PetriNet (pm4py.PetriNet): The PetriNet.
    :initial_marking (pm4py.Marking): The initial marking of the PetriNet.
    :final_marking (pm4py.Marking): The final marking of the PetriNet.
    :parameters (dict): A dictionary with optional parameters for the alignement algorithm.
    
    Returns
    ----------
    :List[Dict[str,List[Tuple[Optional[str],Optional[str]]]]]: The alignment is a sequence of labels of the form (a,t), (a,>>), or (>>,t) representing synchronous/log/model-moves.
    """
    # Load the XES file
    log = pm4py.read_xes(xes_file_path)

    # Apply the conformance alignment algorithm to the log and PetriNet using the initial and final markings
    alignments = pm4py.algo.conformance.alignments.petri_net.algorithm.apply_log(log=log, petri_net=PetriNet, initial_marking=initial_marking,final_marking=final_marking)
    
    return alignments

def save_alignments(alignments: List, file_path: str):
    """
    Save the given alignments to a file using pickle.

    Parameters
    ----------
    - alignments (List): The alignments to be saved.
    - file_path (str): The path where the alignments will be saved.
    
    Returns
    ----------
    None
    """
    # Save alignments to pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(alignments, f)
    print(f"Alignments successfully saved to {file_path}")
    
def load_alignments(file_path: str):
    """
    Load alignments from a file using pickle.

    Parameters
    ----------
    - file_path (str): The path of the file to load the alignments from. 
    
    Returns
    ----------
    :List[Dict[str,List[Tuple[Optional[str],Optional[str]]]]]: The alignments that was loaded from the file.
    """
    # Load alignments from pickle file
    with open(file_path, 'rb') as f:
        alignments = pickle.load(f)
    print(f"Alignments successfully opened")
    
    return alignments

def clean_alignments(alignments: List):
    """
    Cleans the alignment data by replacing None values with the tau symbol (τ).

    Parameters
    ----------
    - alignments (List): A list of alignment results where each trace containts the alignment of activities between the log and PetriNet
    
    Returns
    ----------
    :List[Dict[str,List[Tuple[Optional[str],Optional[str]]]]]: The alignments that was loaded from the file.
    """
    # Initialize tau symbol 
    tau_symbol = '\u03c4'
    
    for trace in alignments:
        for i, (log_activity, model_activity) in enumerate(trace['alignment']):
            # Check if an activity is None and replace it with the tau symbol
            if log_activity is None:
                trace['alignment'][i] = (tau_symbol, model_activity)
            elif model_activity is None:
                trace['alignment'][i] = (log_activity, tau_symbol)
    
    return alignments

def generate_trace_encoding(alignments: List):
    """
    Load alignments from a file using pickle.

    Parameters
    ----------
    - alignments (List): A list of alignment results where each trace containts the alignment of activities between the log and PetriNet
    
    Returns
    ----------
    List[Dict[str, Dict[str, int]]]: A list where each element corresponds to a trace. Each trace is represented by a dictionary where the keys are the activity names and the values are dictionaries with the counts of 'log_moves' and 'model_moves'
    """
    # Initialize a list to store the move counts for each trace
    move_count_per_trace = []
    
    for trace in alignments:
        # Initialize a dictionary to store move counts for each activity in the current trace
        move_count = {}
        for log_activity, model_activity in trace['alignment']:
            # If the activity is not the '>>' symbol and hasn't been added to the dictionary yet, add it
            if (log_activity != ">>") and (log_activity not in move_count):
                move_count[log_activity] = {'log_moves': 0, 'model_moves': 0}
            elif (model_activity != ">>") and model_activity not in move_count:
                move_count[model_activity] = {'log_moves': 0, 'model_moves': 0}
            
            # Increment the model move count if the log activity is '>>' (indicating a move only in the model)
            if log_activity == ">>":
                move_count[model_activity]['model_moves'] += 1
            # Increment the log move count if the model activity is '>>' (indicating a move only in the log)
            elif model_activity == ">>":
                move_count[log_activity]['log_moves'] += 1
        
        # Add the move count dictionary for the current trace to the list
        move_count_per_trace.append(move_count) 
        
    return move_count_per_trace

def make_alignments_table(alignments: List) -> None:
    """
    Displays the alignments between log and model for each trace in a tabular format. 

    Parameters
    ----------
    - alignments (List): A list of alignment results where each trace containts the alignment of activities between the log and PetriNet.
    
    Returns
    ----------
    None
    """
    for trace_id, trace in enumerate(alignments):
        # Initialize a list to store both log and model activities for each trace
        log_activities = []
        model_activities = []
        
        # Iterate over each (log_activity, model_activity) pair and add them to their respective lists
        for log_activity, model_activity in trace['alignment']:
            log_activities.append(log_activity)
            model_activities.append(model_activity)
        
        # Make dataframe
        data = {
            'Log': log_activities,
            'Model': model_activities}
        df = pd.DataFrame(data)
        
        # Print the table heading including the trace index
        print(f"\n\033[1mTable for Trace {trace_id + 1}:\033[0m\n")
        
        # Print the headers for the log and model columns
        print(f"\033[1m{'LOG':<50} \033[1m{'MODEL':<50}\033[0m")
        
        # Print a separator line to visually distinguish between rows
        print('-' * 100)
        
        # Print the log and model activities
        for index, row in df.iterrows():
            log_value = str(row['Log'])
            model_value = str(row['Model'])
            print(f"{log_value:<50} {model_value:<50}")
            print('-' * 100)
            
def make_dataframe_for_decision_tree(xes_file_path: str, move_count_per_trace: List, save_path: str):
    """
    This function merges throughput time with trace moves (log and model) into a DataFrame, preparing it for decision tree analysis.

    Parameters
    ----------
    - xes_file_path (str): The filepath of the xes file (event log file).
    - move_count_per_trace (List): A list where each element corresponds to a trace. Each trace is represented by a dictionary where the keys are the activity names and the values are dictionaries with the counts of 'log_moves' and 'model_moves'.
    - save_path (str): The path where the resulting DataFrame will be saved as a CSV file. 
    
    Returns
    ----------
    None
    """
    # Load the XES file
    log_df = pm4py.read_xes(xes_file_path)
    
    # Calculate throughput time for each trace, grouped by '@@case_index'
    throughput_time_df = log_df.groupby('@@case_index')['case:throughput_time'].first().reset_index()

    # Flatten the trace data and convert it to a DataFrame
    flattened_traces = []
    for trace_number, trace in enumerate(move_count_per_trace, start=0):
        # Add trace number to each flattened trace
        flat_trace = {'trace_number': trace_number} 
        for activity, moves in trace.items():
            # Replace spaces with underscores for activity names, they are not None
            if activity is not None: 
                activity = activity.replace(' ', '_')  
            # Add the model moves and log moves for each activity to the trace dictionary.
            flat_trace[f'{activity}_log_moves'] = moves['log_moves']
            flat_trace[f'{activity}_model_moves'] = moves['model_moves']
        flattened_traces.append(flat_trace)

    # Create a DataFrame for the flattened trace data
    df_flattend = pd.DataFrame(flattened_traces)
    
    # Combine the flattened trace data with the throughput time based on the trace number
    df = pd.merge(df_flattend, throughput_time_df, left_on= 'trace_number', right_on= '@@case_index')
    
    # Drop the '@@case_index' column, as it is no longer needed
    df = df.drop('@@case_index', axis=1)
    
    # Write it to csv
    df.to_csv(save_path, index=False)

if __name__ == "__main__": #### Dit gaat nog weg MORGEN ####
    # PetriNet, iMarking, fMarking = load_pnml("data/BPI2017Denied_petriNet.pnml")
    # alignments = create_alignment("data/BPI2017Denied(3)_Throughput.xes", PetriNet, iMarking, fMarking)
    # save_alignments(alignments, 'data/alignments.pkl')
    
    alignments = load_alignments('data/alignments.pkl')
    clean_alignments = clean_alignments(alignments)
    # move_count_per_trace = generate_trace_encoding(clean_alignments)
    # print(f"Move counts for the first trace (log and model): {move_count_per_trace[0:5]}")
    # make_dataframe_for_decision_tree("data/BPI2017Denied(3)_Throughput.xes", move_count_per_trace, 'data/df_with_tau_for_decision_tree.csv')
    
    # view_event_log_petrinet("data/BPI2017Denied(3)_Throughput.xes")
    make_alignments_table([clean_alignments[0]])

