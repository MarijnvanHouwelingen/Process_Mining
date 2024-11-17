# Conformance Checking | Process Mining | Group 7
This is the repository for the Process mining group 7 for the subject Process Mining (JM0211-M-6). This project focusses on the reproducion of the 'Deviation analysis' step in the methodology of Dees et al. (2017).  

## Motivation
Conformance checking techniques compare a normative process with behaviors logged in an event log during the actual process execution. Such techniques try to pinpoint the discrepancies. Deviations from the normative process model can highlight a structural violation of possible measures or guards. For example in a loan application process, when the total family income is under €10000, then disapprove all loans above €50000.

As aforementioned, deviations might point out discrepancies that are detrimental to the overall normative process. However, recent studies highlight that deviations might point out workarounds that are actually linked to a better process performance.

Our project aims to specifically find the deviations, or workarounds, that improve process performance based on a common key performance indicator: Throughput time.

## Method 
To achieve this, our artifact must take the following inputs: A Petri Net (pmnl format) and an event log (XES format). 
Our artifact must return a set of rules correlating with each detected dviation with a KPI (Throughput time) and for each rule, a subset of traces that are compliant with that rule. 

The first step undertaken was the computation step of the throughput time and converting it back to a XES format. The function  *throughput_time_to_xes* in the python file *xes_to_csv.py* is responsible for this step. 

The second step is loading the normative process model (PetriNet). The PetriNet given *BPI2017Denied_PetriNet.pmnl* is loaded in and visualized with the function *load_pnml* in the python file *alignment.py*. 

The third step undertaken was the generation of the alignments for each trace in the event log. After the eventlog and the normative process model are loaded the function *create_alignment.py* will create the optimal alignments with the event log and PetriNet as parameters. The alignments can also be saved as a .pkl file with the function *save_alignment* in the *alignment.py* python file. 

Before applying a decision tree on the alignments, they have to be cleaned and encoded. The function clean_alignment in the python file *create_alignment.py* replaces the silent activity model move and log move *((none, >>) and (>>,none))* with a τ for additional readability and interpretability in the decision tree regressor model.

The encoding step is undertaken with the functions *generate_trace_encoding* and *make_dataframe_for_decision_tree* both in the *create_alignment.py* python file. The *generate_trace_encoding* counts and aggregates all alignments on a trace level with one-hot encoding. *make_dataframe_for_decision_tree* adds the troughput time per trace and converts the object into a csv file for the decision tree. 

The fourth step is the application of the set of encoded traces on a decision tree algorithm. The decision tree regressor algorithm can be found in *Ranking.py*. Hyperparameter optimalization trough gridsearch cross validation in *ranking.py* is being utilized to enhance the overall performance of the decision tree regressor algorithm. The final hyperparameters can be found in a configuration file named: **param_grid.json**. 

Finally, the classification rules derived and ranked based on the gini coeficient and throughput time from the decision tree are produced with the function **extract_and_print_rules** in python file **Ranking.py**. This function traverses through the derived decision tree and returns the classification rules and the amount of traces that apply for each classification rule in a txt file. 

## Repository overview
tree command in terminal aan het einde van het project (Morgen). Ook nog even alles 1 keer test draaien morgen. 

## Instructions
The file **main.ipynb** contains all steps in order as described in the method section. Running all functions from top to bottom should ultimately generate classification rules with the amount of traces that apply for each classification rule. There are additional chunks that show the alignment table and visualize both the normative process model (PetriNet) and a PetriNet model mined from the event log. These code chunks will be marked with **Additional** above the code chunk.

## Poetry install instructions:
This project utilizes poetry as a version control system for python instead of pip. Poetry additionally resolves dependencies not seen in regular version control systems (such as pip). The more detailed versions of all packages can be found in the requirements.txt. The python version utilized is: *python = "^3.11"*. With the main dependencies for this project being:
```toml
[tool.poetry.dependencies]
python = "^3.11"
pandas = "^2.2.3"
ipykernel = "^6.29.5"
pm4py = "^2.7.11.13"
scikit-learn = "^1.5.2"
matplotlib = "^3.9.2"
numpy = "^2.1.3"
```

This section of the README file focusses on the utilization of Poetry in for new users:
**Step 1: Install poetry**: 
Install poetry using the pip command: ```pip install poetry```. Install it on a virtual environment if you want poetry to be isolated from the main python environment.

**Step 2: Install dependencies**:
When poetry is installed, go to the root folder (where the toml file is located). Then install the dependencies and create a virtual environment (.venv) with the command ```poetry install``` 

**Step 3: Choose kernel in notebook**:
When the new virtual environment is created in the .venv folder, you can select the new environment with the option *select kernel*. 
