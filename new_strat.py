from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_proper_link, get_vectorize_dict
from pre_process_ds import pre_process
import numpy as np
import pandas as pd
import hashlib


VECTORIZE_MODEL = SentenceTransformer('bert-base-nli-mean-tokens')
# VECTORIZE_MODEL = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# contents/alumni/docs/promanade_competition_brief.pdf
# message1 = "pull out promanade competition pdf"
# message1 = "pull out alumni cv checklist"
message1 = "Pull out the list of about the sasd services"
# message1 = "Pull out the list of about the library services"
# message1 = "Pull out the list of about the ice services"
# message1 = "Pull out the list of about the eservices"

data = pd.read_csv("NEW_DATA.csv")
links = data.path.values.tolist()

processed_message = pre_process(message1).split()
question_length = len(processed_message)
processed_message_set = set(processed_message)
process_msg_vec = VECTORIZE_MODEL.encode(processed_message)

print(processed_message)
score = []
dump_score = []

LINKS_VEC = np.load("NEW_DATA.npy", allow_pickle=True)
# print(type(LINKS_VEC), LINKS_VEC.shape, process_msg_vec.shape)
css = []

for i in range(len(LINKS_VEC)):
    # links_set.append(set(pre_process(get_proper_link(links[i])).split()))
    cs = np.transpose(cosine_similarity(process_msg_vec, LINKS_VEC[i]))
    css.append((i, np.average(cs)))
    
for i in range(len(css)):
    current_score = (i, css[i][1])
    if len(dump_score) == 0:
        dump_score.append(current_score)
    else:
        for j in range(len(dump_score)):
            if dump_score[j][1] < current_score[1]:
                dump_score.insert(j, current_score)
    
    if css[i][1] > 0.50:
        if len(score) == 0 :
            score.append(current_score)
        else: 
            prev_len = len(score)
            j=0
            while j < len(score):
                if current_score not in score:
                    if score[j][1] < current_score[1]:
                        score.insert(j, current_score)
                    if score[j][1] >= current_score[1] and j < len(score) - 1:
                        j+=1
                        continue
                    if j == len(score) - 1:
                        score.append(current_score)
                        break
                j+=1


# print(len(dump_score))
# print(len(score))
# print("==========================================")
# print(dump_score)
# print()
# print(score)

print(links[score[0][0]])
print(links[score[1][0]])
print(links[score[2][0]])
print(links[score[3][0]])
print(links[score[4][0]])

links_set = [(score[i][0], set(pre_process(get_proper_link(links[score[i][0]])).split())) for i in range(len(score))]
# print(links_set)

# print("JACCARD DISTANCE")
jaccard = [len(processed_message_set.intersection(link_set[1])) / len(processed_message_set.union(link_set[1])) for link_set in links_set]

# print(jaccard)
print(jaccard.index(max(jaccard)))
print(links[links_set[jaccard.index(max(jaccard))][0]])
