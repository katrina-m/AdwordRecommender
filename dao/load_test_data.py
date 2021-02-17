import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_hot_data():
    path = "/tf/code/chenjiazhen/baojie_kg/processed_data"

    relation_data = pd.read_csv(os.path.join(path, "keyword_board_relation.csv"))
    board_data = pd.read_csv(os.path.join(path, "board.csv"))
    keyword_data = pd.read_csv(os.path.join(path, "keyword.csv"))
    relation_data = relation_data[["StartDay", "BoardID", "KeywordID", "点击率"]]
    relation_data.rename(columns={"StartDay": "Date"}, inplace=True)
    relation_data["Date"] = pd.to_datetime(relation_data["Date"])

    board_data["Category"] = LabelEncoder().fit_transform(board_data["Category"])
    board_data["Sub_category"] = LabelEncoder().fit_transform(board_data["Sub_category"])

    # Keep position 0 for masking purpose.
    relation_data["KeywordID"] = relation_data["KeywordID"] + 1
    keyword_data["Rootword_ids"] = [np.array(eval(rwIds))+1 for rwIds in keyword_data["Rootword_ids"].values]
    keyword_data["KeywordID"] = keyword_data["KeywordID"] + 1
    return relation_data, board_data, keyword_data


