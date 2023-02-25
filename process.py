from pre_process_ds import pre_process
import pandas as pd
from utils import string_similarity


data = pd.read_csv("NEW_DF.csv").values.tolist()
QUESTION = "get me the list of library services"
QUESTION = "campus service travel services contact"
QUESTION = "library services"
QUESTION = "eservices"

process_text = pre_process(QUESTION)
print("PROCESS TEXT", process_text)
ratio = []

for link in data:
    process_link = pre_process(link[3][29:].replace("_", " ").replace("/", " ").replace("test", "").replace("index", ""))
    ratio.append([link[3], process_link, string_similarity(process_text, process_link)])
    
df = pd.DataFrame(ratio, columns=['path', 'process_path', 'ratio'])
main_df = df.loc[~df['path'].str.contains("_hidden")]
main_df = df.loc[~df['path'].str.contains("_deleted_item")]
main_df = main_df.sort_values('ratio', ascending=False)
main_df = main_df.drop_duplicates(subset="path", keep="first")

print(main_df.head(10))