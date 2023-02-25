from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from utils import get_proper_link, get_vectorize_dict
from pre_process_ds import pre_process
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pandas as pd
import sys
import hashlib
import json


model = SentenceTransformer('bert-base-nli-mean-tokens')
kw_model = KeyBERT(model='all-mpnet-base-v2')

message1 = "Pull out the list of about the sasd services"
# message1 = "Pull out the list of about the library services"
# message1 = "Pull out the list of about the ice services"
# message1 = "Pull out the list of about the eservices"

message2 = "https://www.eservices.zu.ac.ae/"
message3 = "https://www.zu.ac.ae/main/en/library/services"
message4 = "https://www.zu.ac.ae/main/en/SASD/services"
message5 = "https://www.zu.ac.ae/main/en/_ice/services"

processed_message1 = pre_process(message1).split()
question_length = len(processed_message1)
# processed_message2 = pre_process(get_proper_link(message2))
# processed_message3 = pre_process(get_proper_link(message3))
# processed_message4 = pre_process(get_proper_link(message4))
# processed_message5 = pre_process(get_proper_link(message5))

# dataset = [message2, message3, message4, message5]

processed_message2 = pre_process(get_proper_link(message2)).split()
processed_message3 = pre_process(get_proper_link(message3)).split()
processed_message4 = pre_process(get_proper_link(message4)).split()
processed_message5 = pre_process(get_proper_link(message5)).split()

# JACCARD DISTANCE
processed_message_set1 = set(processed_message1)
processed_message_set2 = set(processed_message2)
processed_message_set3 = set(processed_message3)
processed_message_set4 = set(processed_message4)
processed_message_set5 = set(processed_message5)

print("JACCARD DISTANCE")
print(len(processed_message_set1.intersection(processed_message_set2)) / len(processed_message_set1.union(processed_message_set2)))
print(len(processed_message_set1.intersection(processed_message_set3)) / len(processed_message_set1.union(processed_message_set3)))
print(len(processed_message_set1.intersection(processed_message_set4)) / len(processed_message_set1.union(processed_message_set4)))
print(len(processed_message_set1.intersection(processed_message_set5)) / len(processed_message_set1.union(processed_message_set5)))



# dataset = [processed_message2, processed_message3, processed_message4, processed_message5]
# dataset_list = []
# for _dataset in dataset:
#     for i in _dataset:
#         dataset_list.append(i)

# print(dataset_list)
# # print(dataset_list)
# dataset_set = set(dataset_list)
# print('dataset_set', dataset_set)

# dataset_dict = {}
# for i in dataset_set:
#     count = 0
#     for j in dataset_list:
#         if i == j:
#             count += 1
#     # print(i, count, hashlib.sha256(i.encode('utf-8')).hexdigest())
#     dataset_dict[hashlib.sha256(i.encode('utf-8')).hexdigest()] = 1 / count

# dataset_dict = json.load(open("dataset_dict.json"))
# print(dataset_dict)


# question_dict2 = get_vectorize_dict(processed_message2)
# question_dict3 = get_vectorize_dict(processed_message3)
# question_dict4 = get_vectorize_dict(processed_message4)
# question_dict5 = get_vectorize_dict(processed_message5)

# print("HASH", question_dict3)
# print("HASH", question_dict3.get("c60d1ea3b46257482ab1ffc600b2f30a28d513a8409cbde19b1de73f7333b6b4"))

questions_vec1 = model.encode(processed_message1)
questions_vec2 = model.encode(processed_message2)
questions_vec3 = model.encode(processed_message3)
questions_vec4 = model.encode(processed_message4)
questions_vec5 = model.encode(processed_message5)

print("MSG SHAPES")
print(questions_vec1.shape)
print(questions_vec2.shape)
print(questions_vec3.shape)
print(questions_vec4.shape)
print(questions_vec5.shape)

# TODO: Get the MAX shape of the vertors

# EUCLIDEAN DISTANCE

q_np2 = np.pad(questions_vec2, ((0, questions_vec1.shape[0] - questions_vec2.shape[0]), (0, 0)), mode="constant", constant_values=0)
q_np3 = np.pad(questions_vec3, ((0, questions_vec1.shape[0] - questions_vec3.shape[0]), (0, 0)), mode="constant", constant_values=0)
q_np4 = np.pad(questions_vec4, ((0, questions_vec1.shape[0] - questions_vec4.shape[0]), (0, 0)), mode="constant", constant_values=0)
q_np5 = np.pad(questions_vec5, ((0, questions_vec1.shape[0] - questions_vec5.shape[0]), (0, 0)), mode="constant", constant_values=0)


# questions_vec = model.encode(["eservic", "en", "librari", "servic", "sasd", "ice"])

print(processed_message1)
print(processed_message2)
print(processed_message3)
print(processed_message4)
print(processed_message5)

# print("QUESTION VEC", questions_vec)

# cs1 = cosine_similarity(questions_vec1, question_dict2) # ['eservic']
cs2 = np.transpose(cosine_similarity(questions_vec1, questions_vec2))
cs3 = np.transpose(cosine_similarity(questions_vec1, questions_vec3))
cs4 = np.transpose(cosine_similarity(questions_vec1, questions_vec4))
cs5 = np.transpose(cosine_similarity(questions_vec1, questions_vec5))

css = [cs2, cs3, cs4, cs5]
# dataset_dict.get(question_dict2.get())
# print(type(question_dict3.keys()))
# print(list(question_dict3.keys())[0])
# print()
print(np.linalg.norm(questions_vec1 - q_np2, ord=1))
print(np.linalg.norm(questions_vec1 - q_np3, ord=1))
print(np.linalg.norm(questions_vec1 - q_np4, ord=1))
print(np.linalg.norm(questions_vec1 - q_np5, ord=1))

print("===================================")
print(cs2, np.max(cs2), np.average(cs2), np.sum(cs2) / question_length)
print(cs3, np.max(cs3), np.average(cs3), np.sum(cs3) / question_length)
print(cs4, np.max(cs4), np.average(cs4), np.sum(cs4) / question_length)
print(cs5, np.max(cs5), np.average(cs5), np.sum(cs5) / question_length)
print("===================================")


score = []
dump_score = []

for i in range(len(css)):
    # qtag_value = questions.get('qtag')[i] # list of unique id
    current_score = (i, np.average(css[i]))
    if len(dump_score) == 0:
        dump_score.append(current_score)
    else:
        for j in range(len(dump_score)):
            if dump_score[j][1] < current_score[1]:
                dump_score.insert(j, current_score)
    
    print(dump_score, score, current_score)
            
    if np.average(css[i]) > 0.50:
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

if len(score) == 0:
    if len(dump_score) == 0:
        ans_disp = ""
    else:
        score = dump_score
    

print(score)

# display results
ans_disp = ""
tags=[]
for index, value in score[:200]:
    # tag = answers.get('qtag')[index]
    print(index, value)
    # TODO: JACCARD SIMILARITY

    # if tag not in tags:    
    #     tags.append(tag)
    # ans_disp +=list(answers.loc[answers['qtag']==tag]['answers'])[0]+"\n"
            




# temp = [[]]
# for i in range(question_length):
#     if cs2[i] is not None:
#         temp[i].append(np.max(cs2[i]))
#         temp[i].append(np.max(cs3[i]))
#         temp[i].append(np.max(cs4[i]))
#         temp[i].append(np.max(cs5[i]))
#     else:
#         temp[i].append(0)

# print(temp)

# for i in range(len(cs2)):
#     # print(question_dict3.keys()[i])
#     cs2[i] = cs2[i] * dataset_dict.get(list(question_dict2.keys())[i])

# for i in range(len(cs3)):
#     # print(question_dict3.keys()[i])
#     cs3[i] = cs3[i] * dataset_dict.get(list(question_dict3.keys())[i])

# for i in range(len(cs4)):
#     # print(question_dict3.keys()[i])
#     cs4[i] = cs4[i] * dataset_dict.get(list(question_dict4.keys())[i])

# for i in range(len(cs5)):
#     # print(question_dict3.keys()[i])
#     cs5[i] = cs5[i] * dataset_dict.get(list(question_dict5.keys())[i])


# print(cs2, np.max(cs2), np.average(cs2), np.sum(cs2))
# print(cs3, np.max(cs3), np.average(cs3), np.sum(cs3))
# print(cs4, np.max(cs4), np.average(cs4), np.sum(cs4))
# print(cs5, np.max(cs5), np.average(cs5), np.sum(cs5))



# print(pre_process(message1))
# print(pre_process(get_proper_link(message2)))
# print(pre_process(get_proper_link(message3)))

# keywords1 = kw_model.extract_keywords(message1.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
# keywords2 = kw_model.extract_keywords(message2.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
# keywords3 = kw_model.extract_keywords(message3.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
# print(*keywords1, sep="\n")
# print()
# print(*keywords2, sep="\n")
# print()
# print(*keywords3, sep="\n")

# CORPUS length = [10]
# list of unique words = [eservic, en, librari, servic, sasd, ice]
# word_score = [1, 3, 1, 3, 1, 1]
# 
# 