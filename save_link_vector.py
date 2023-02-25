from sentence_transformers import SentenceTransformer
import pandas as pd
from utils import get_proper_link
from pre_process_ds import pre_process
import numpy as np
from datetime import datetime

print("MODEL SAVE START", datetime.now())

VECTORIZE_MODEL = SentenceTransformer('bert-base-nli-mean-tokens')

data = pd.read_csv("ZU_all_files.csv")
links = data.path.values.tolist()
print("LINKS PRESENT IN CSV", len(links))

links_ = [VECTORIZE_MODEL.encode(pre_process(get_proper_link(link)).split()) for link in links]
np.save("NEW_DATA.npy", links_)

print("MODEL SAVE END", datetime.now())