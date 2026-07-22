import re

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)