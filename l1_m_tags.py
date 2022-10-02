# Libraries for data management
import os
import json
import pandas as pd
import numpy as np
import ast

#FOR PROCESSING
import nltk
import re
nltk.download("omw-1.4")

#FORR W2V
import gensim
import gensim.downloader as gensim_api

#FOR PLOTTING
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#FOR BERT MODEL -> TO STUDY BERT MODEL
import transformers

class m_tags():
    def __init__(self,df_videos, df_tags_master, remove_values, param = 500) -> None:
        self.df_videos = df_videos
        self.df_tags_master = df_tags_master
        self.param = param
        self.remove_values = remove_values

    def master_tags(self):
        df_videos = self.df_videos
        df_videos["tags"] = self.df_videos.tags.str.split(",")
        df_videos = df_videos.explode("tags")
        lst_stopwords = nltk.corpus.stopwords.words("english")

        df_videos["tags_clean"] = df_videos["tags"].apply(lambda x: self.clean_tags(
            x,
            flg_stemm=False,
            flg_lemm=True,
            lst_stopwords=lst_stopwords
        ))

        self.m_tags_agg = self.agg_tags(df_videos)
        m_tags_matched = self.tags_dataframe()
        m_tags_missing = self.tags_missing(m_tags_matched)

        return m_tags_matched, m_tags_missing

    def clean_tags(self, text, flg_stemm = False, flg_lemm=True, lst_stopwords = None):
        ## Clean (convert to lowercase and remove punctuation and characters and then strip)
        text = re.sub(r'[^\w\s]', '', str(text).lower())

        ## Tolenize (Convert from string to List)
        lst_text = text.split()

        ## Remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in lst_stopwords]

        ## Stemming (to remove -ly, -ing, etc.)
        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        ## Lemmatisation (Convert the word into root word)
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        ## Back to string from list
        text = " ".join(lst_text)
        return text
    
    def agg_tags(self, df_videos):

        df_tags_analysis = df_videos.groupby('tags_clean', as_index=False).agg(
            view_sum = pd.NamedAgg(column='viewCount', aggfunc='sum'),
            like_sum = pd.NamedAgg(column='likeCount', aggfunc='sum'),
            comment_sum = pd.NamedAgg(column='commentCount', aggfunc='sum')
        )

        df_tags_analysis = df_tags_analysis.loc[df_tags_analysis["like_sum"]>self.param]

        df_tags_analysis = df_tags_analysis[~df_tags_analysis["tags_clean"].isin(self.remove_values)]

        return df_tags_analysis

    def tags_dataframe(self):
        tags_dataframe = pd.DataFrame()

        for i,j in self.df_tags_master.iterrows():
            link = self.m_tags_agg[self.m_tags_agg.tags_clean.str.contains(j.key_word)]
            link["key_word"] = j.key_word
            link["categorie"]= j.categorie

            tags_dataframe = tags_dataframe.append(link)

        return tags_dataframe

    def tags_missing(self, m_tags_matched):
        analysis = self.m_tags_agg.merge(m_tags_matched[["tags_clean", "categorie"]], on="tags_clean", how="left")
        analysis = analysis[analysis["categorie"].isnull()]
        analysis = analysis[["tags_clean", "categorie"]]

        return analysis

if __name__ == '__main__':
    df_tags_master =  pd.read_csv("param_labels.csv")
    df_videos = pd.read_csv("/Users/jesustellez/Desktop/aiDynamics/API Data extraction/Data/videos_eeuu.csv")
    df_videos["Date"] = df_videos["publishedAt"].str.split("T", expand=True)[0]

    remove_values = ["breaking news", "live video", "toriginal", "real time coverage", "news", "washington post", "anational", "apolitics", "spolitics", "snational", 
    "anational", "washington post video" ,"md va", "aworld", "sworld", "wapo video"]

    aux = m_tags(df_videos, df_tags_master, remove_values)
    tags_matched, tags_mising = aux.master_tags()
    print("SUCCESSFULL")