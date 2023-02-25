import re
from autocorrect import Speller
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


ps = PorterStemmer()
spell = Speller(lang="en")


def list_to_str(_list):
    _str = ""
    if len(_list) > 1:
        for i in _list:
            if i == _list[-1]:
                _str += i

            else:
                _str += i + " "

    return _str


def remove_zu(_str_list, _str_to_insert, _sub_str="zu"):
    for i in _str_list:
        if _sub_str in i:
            idx = _str_list.index(i)
            _str_list.insert(idx, _str_to_insert)
            _str_list.pop(idx + 1)
    
    return list_to_str(_str_list)
    

text = "about the zayed university's"
_text = text.lower()

_data = {'user_email': 'sohail@gmail.com', 'event_type': '4', 'event_question': "University's accreditation and substaintial equivalency", 'session_value': '', 'intent': '', 'spell_check_bool': True}
print(_data)
print(_data["spell_check_bool"])
# text = _data['event_question']
# text = "get me zu's eparticipation policy"
text = "Get me ZU's eparticipation policy".lower()
# print(spell(text_main))
print(spell(text), text, spell(text) != text)

uncorrect = spell(text).lower()
u_list = uncorrect.split()
temp = u_list.copy()
if "university" in uncorrect or "university?" in uncorrect:
    try:
        uni_pos = u_list.index("university")
    except:
        try:
            uni_pos = u_list.index("university?")
        except:
            uni_pos = u_list.index("university's")
    try:
        if u_list[uni_pos - 1] == "based":
            u_list[uni_pos - 1] = "zayed"
        
        temp[uni_pos] = "university"
    except:
        pass

res = ' '.join([str(elem) for elem in u_list])
print("RES", res)
if res.lower() != text.lower() and _data['spell_check_bool'] == True:
    print({'session_id': "session_id_", 'answer': f'{res}', 'intent': 'spell'})

else:
    res = ' '.join([str(i) for i in temp])
    text = res
    print("TEXT", text)

text_list = text.split(" ")
text_main = remove_zu(text_list, "zayed university")

print("TEXT", text_main)
