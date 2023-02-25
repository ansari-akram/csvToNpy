from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from utils import *
from autocorrect import Speller
import pandas as pd


tag_model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer('bert-base-nli-mean-tokens')
kw_model = KeyBERT(model='all-mpnet-base-v2')
spell = Speller(lang='en')


EXTENSTION_LIST = ["JPG", "PDF", "DOC", "PNG", "DOCX", "GIF", "XLSX", "JPEG", "ASPX", "ASP"]


while True:
    text = input("=====================================================================\nEnter Question\n")
    if text.lower() == "exit":
        break

    uncorrect = spell(text).lower()
    u_list = uncorrect.split()
    temp = u_list.copy()
    temp_1 = text.split()
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
    if res.lower() != text.lower():
        print({'session_id': "session_id_", 'answer': f'{res}', 'intent': 'spell'})
        # return JsonResponse({'session_id': session_id_, 'answer': f'{res}', 'intent': 'spell'})

    else:
        res = ' '.join([str(i) for i in temp_1])
        text = res

    _text = text.lower()
    _main_input = str_to_list(_text)
    _text = remove_zu(_main_input, "zayed university")
    _main_input = str_to_list(_text)

    _main_input_list = [i for i in _main_input if i]
    # _main_input_list = remove_custom('i', _main_input_list)
    # _main_input_list = remove_custom('a', _main_input_list)
    # _main_input_list = remove_custom('the', _main_input_list)
    print("MAIN INPUT LIST", _main_input_list)
    _main_input_string = remove_stopwords(_main_input_list, text)
    print("MAIN INPUT STRING", _main_input_string)
    # _main_input_string = list_to_str(_main_input_list)

    intents = ""
    main_df = pd.DataFrame()
    all_csv_ = pd.read_csv("NEW_DF.csv").values.tolist()
    # print("ALL CSV _")
    # print(all_csv_[:3])

    # all_csv = []
    # all_csv = get_ratios(_main_input_list, all_csv_, all_csv)
    # print("ALL CSV")
    # print(all_csv[:3])

    # _main_input_string = list_to_str(_main_input_list)
    # print("MAIN INPUT STRING", _main_input_string)

    links_ratio = []
    for i in all_csv_:
        try:
            temp_link = i[3].replace("_", " ").replace("/", " ")
            links_ratio.append([i[0], string_similarity(temp_link, _main_input_string), i[2], i[3], i[4]])
        except:
            pass

    # print("LINKS RATIOS")l
    # print(links_ratio[:3])


    df1 = pd.DataFrame(links_ratio, columns=['single_ratio', 'actual_ratio', 'name', 'path', 'timestamp'])
    # df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    main_df = df1.drop_duplicates(subset="path", keep="last")
    main_df = main_df.loc[~main_df['path'].str.contains("_hidden")]
    print(main_df.sort_values('actual_ratio', ascending=False).head(10))
    top_df1 = main_df.sort_values('actual_ratio', ascending=False).head(5).values.tolist()

    max_ratio = top_df1[0][1]
    print("MAX RATIO", max_ratio)

    top_df1 = [i[3] for i in top_df1]
    top_df_extension = get_proper_extension(top_df1)

    df1_str = ""
    for i in top_df_extension:
        df1_str += i + "\n"
            
    if len(top_df_extension) > 0:
        print(*top_df_extension, sep="\n")
        # print({'session_id': "session_id_", 'answer': df1_str, 'intent': 'General', 'url': top_df_extension})
        # return JsonResponse({'session_id': session_id_, 'answer': df1_str, 'intent': 'General', 'url': top_df_extension})

    else:
        print("[ELSE] Sorry, I am not able to detect the language you are asking.")
        # eid = EventType.objects.get(id=int(5))
        # Log.objects.create(event_type_id=eid, user_email=user_email, user_ip=ip, event_question=text,
        #                 event_answer='', intent='General')
        # return JsonResponse(
        #     {'session_id': session_id_,
        #     'answer': "Sorry, I am not able to detect the language you are asking."})