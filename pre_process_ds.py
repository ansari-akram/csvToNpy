from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from utils import string_similarity


STOP_WORDS = set(stopwords.words('english'))
PS = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOP_WORDS])


def pre_process(text):
    text = text.lower()
    text = remove_stopwords(text)
    words = word_tokenize(text)
    # print("WORDS", words)
    text = ""
    for word in words:
        # print(PS.stem(word), LEMMATIZER.lemmatize(PS.stem(word)))
        if word == words[-1]: text += LEMMATIZER.lemmatize(PS.stem(word))
        else: text += LEMMATIZER.lemmatize(PS.stem(word)) + " "
    return text

# text = "Pull out the list of library services"
# text = "list of eservices"

# process_text = pre_process(text)
# print("TEXT", process_text)

# link = "https://www.zu.ac.ae/main/en/library/services"
# link = link.replace("_", " ").replace("/", " ").replace(".", " ")
# print(link)
# process_link = pre_process(link[29:])
# print("LINK", process_link)
# print(string_similarity(process_text, process_link))

# link = "https://www.eservices.zu.ac.ae/"
# link = link.replace("_", " ").replace("/", " ").replace(".", " ")
# process_link = pre_process(link)
# print("LINK", process_link)

# # ratio_list = []
# print(string_similarity(process_text, process_link))

# text = "_deleted_item"
# process_text = pre_process(text)
# print("TEXT", process_text)