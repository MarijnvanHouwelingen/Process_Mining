import pandas as pd
import pm4py 

# Load the XES file
xes_file_path = 'data/BPI2017Denied(3).xes'
log = pm4py.read_xes(xes_file_path)

#print(log)
# Extract data and convert to pandas DataFrame
print(log)

# Create the DataFrame
df = pd.DataFrame(log)

# Display the DataFrame
df.to_csv("data/BPI2017Denied(3).csv")

