import re
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch


class Dataset:

    eos_token = "[EOS]"
    padding_token = "[PAD]"

    def __init__(self):
        pass

    def __init__(self, filename, text_key):
        self.load_data(filename, text_key)

    def load_data(self, filename, text_key):
        self.df = pd.read_csv(filename)
        self.text_key = text_key
        self.tweets = self.df[text_key]

    def clean_data(self, links=True, hashtags=False, mentions=False):
        self.tweets = self.tweets.apply(
            lambda x: self.__clean_text(x, links, hashtags, mentions))

    def __clean_text(self, text, links, hashtags, mentions):
        text = text.lower()
        if links:
            # text = re.sub(r'http\S+', '', text)
            text = re.sub(r'https?://\S+', '', text)
        if hashtags:
            text = re.sub(r'@ [\w]+', '', text)
        if mentions:
            text = re.sub(r'# [\w]+', '', text)
        return text

    def set_padding_token(self, token: str):
        self.padding_token = token

    def set_eos_token(self, token: str):
        self.eos_token = token

    def get_data(self):
        return self.tweets.values.to_list()

    def get_data_df(self):
        return self.tweets

    # def get_dataloader(self, tokenizer=None, max_length=280, batch_size=32, shuffle=True, pin_memory=True, num_workers=0):
    #     dataset = __TweetDataset(tweets=self.tweets.values.to_list(
    #     ), tokenizer=tokenizer, max_length=max_length)

    #     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
