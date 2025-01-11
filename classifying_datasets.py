import os
import pandas as pd
# Get the list of all files and directories
path = "D://Mtech//Semester4//Sample-Datasets"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)
df = pd.DataFrame(dir_list, columns=['filename'])
df.to_csv("D://Mtech//Semester4//bi-assistant//source-classified-sets//file-mapping.csv")