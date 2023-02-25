import re
from nltk.corpus import stopwords


STOP_WORDS = set(stopwords.words('english'))
CONTRADICTION_DICT = {"ain't": "are not","'s":" is","aren't": "are not"}
CONTRADICTION_RE=re.compile('(%s)' % '|'.join(CONTRADICTION_DICT.keys()))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOP_WORDS])


def expand_contractions(text,contractions_dict=CONTRADICTION_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return CONTRADICTION_RE.sub(replace, text)


def pre_process(text):
    text = expand_contractions(text)
    text = text.lower()
    text = remove_stopwords(text)
    return text


text = "Pull out the list of library services"
print(pre_process(text))