import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as petri_visualizer
from typing import Tuple, List, Dict, Optional

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

def create_alignment(xes_file_path: str, PetriNet:pm4py.PetriNet, initial_marking:pm4py.Marking, final_marking:pm4py.Marking, parameters: dict = None) -> List[Dict[str,List[Tuple[Optional[str],Optional[str]]]]]:
    """
    This function applies a alignment algorithm on a PetriNet and event log. Creating an alignment Dictionairy.

    Parameters
    ----------
    :xes_file_path (str): The filepath of the xes file (.pnml)
    :PetriNet (pm4py.PetriNet): The PetriNet object.
    :initial_marking (pm4py.Marking): The initial marking of the PetriNet object.
    :final_marking (pm4py.Marking): The final marking of the PetriNet object.
    :parameters (dict): A dictionary with optional parameters for the alignement algorithm.
    Returns
    ----------
    :List[Dict[str,List[Tuple[Optional[str],Optional[str]]]]]: The alignment is a sequence of labels of the form (a,t), (a,>>), or (>>,t) representing synchronous/log/model-moves.
    """
    # Load the XES file
    log = pm4py.read_xes(xes_file_path)

    # Kan ook nog de cost eruit halen Kyra. Kijk de apply_log docstring.
    alignments = pm4py.algo.conformance.alignments.petri_net.algorithm.apply_log(log=log, petri_net=PetriNet, initial_marking=initial_marking,final_marking=final_marking)
    
    return alignments

if __name__ == "__main__":
    #view_event_log_petrinet("data/BPI2017Denied(3)_Throughput.xes")
    
    PetriNet, iMarking, fMarking = load_pnml("data/BPI2017Denied_petriNet.pnml")
    alignment = create_alignment("data/BPI2017Denied(3)_Throughput.xes", PetriNet, iMarking, fMarking)

