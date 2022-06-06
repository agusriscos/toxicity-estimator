import re
import string
import numpy as np
import pandas as pd

import spacy
from sklearn.model_selection import train_test_split

STR_LIST_TO_DELETE = [
    "&#x200[bB];",  # caracter UNICODE que representa el espacio
    r"^\/?[ru]\/[A-Za-z0-9_-]+",  # menciones Reddit al principio del comentario
    r"[^A-Za-z0-9()=]\/?[ru]\/[A-Za-z0-9_-]+"  # "menciones Reddit"
]

EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)


def read_ruddit(file_path):
    df = pd.read_csv(file_path, usecols=["txt", "offensiveness_score"])
    df = df.rename(columns={"txt": "text", "offensiveness_score": "score"})
    return df


def clean_ruddit(df):
    out = df[~df.text.isin(["[deleted]", "[removed]"])].reset_index(drop=True)  # [deleted] [removed]
    out = out.drop(4354, axis=0).reset_index(drop=True)  # comentario sin información
    out.text = out.text.str.lower()  # minúsculas

    out.text = np.where(out.text.str.contains("|".join(STR_LIST_TO_DELETE), regex=True),
                        out.text.str.replace("|".join(STR_LIST_TO_DELETE), "", regex=True), out.text)  # ruido
    out.text = out["text"].apply(lambda text: EMOJI_PATTERN.sub(r"", text))  # emoji

    # caracteres de control (\n, \r y \t), comillas, asteriscos y signos mayor/menor
    control_pattern = re.compile(r'[><*\"\n\r\t]')
    out.text = out.text.apply(lambda text: control_pattern.sub(r" ", text))
    out.text = out.text.apply(
        lambda x: " ".join([w for w in str(x).split()])
    )
    out.text = out.text.str.replace("thats", "that is")
    return out


def tokenize_ruddit(df):
    out = df.copy()
    pipe = spacy.load("en_core_web_lg", exclude=["textcat", "textcat_multilabel", "transformer"])
    stop_words = pipe.Defaults.stop_words
    stop_words.remove("not")
    punct_set = set(string.punctuation)
    punct_set.remove("!")
    tokens_to_delete = stop_words.union(punct_set)

    tokens_list = []
    for text in out.text:
        tokens = [token.lemma_ for token in pipe(text)]
        tokens = [token.lower() for token in tokens if not token in tokens_to_delete]
        tokens_list.append(" ".join(tokens))
    out["clean_text"] = tokens_list
    out["clean_text"] = out["clean_text"].str.replace(r"\.+", "", regex=True)  # sigue habiendo puntos seguidos
    out["clean_text"] = out["clean_text"].apply(
        lambda x: " ".join([w for w in str(x).split()])
    )
    out = out.loc[out["clean_text"] != ""].reset_index(drop=True) # eliminar nulos tras el procesado
    return out


def save_ruddit(df, dir_path):
    out = df[["text", "clean_text", "score"]]
    train, test = train_test_split(out, test_size=.2)
    train.to_csv(dir_path + "/train_ruddit.csv", index=False)
    test.to_csv(dir_path + "/test_ruddit.csv", index=False)
    out.to_csv(dir_path + "/clean_ruddit.csv", index=False)


def run(input_file_path, output_dir_path):
    df = read_ruddit(file_path=input_file_path)
    print("Cleaning text...")
    df = clean_ruddit(df)
    print("Cleaned!")
    print("Tokenizing text...")
    df = tokenize_ruddit(df)
    print("Tokenized!")
    print("Saving files...")
    save_ruddit(df=df, dir_path=output_dir_path)
    print("Done!")


if __name__ == "__main__":
    run(
        input_file_path="C:\\Users\\agusr\\toxicity-estimator\\data\\raw\\ruddit.csv",
        output_dir_path="C:\\Users\\agusr\\toxicity-estimator\\data\\prep"
    )
