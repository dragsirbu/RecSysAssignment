import polars as pl
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Optional, List, Dict, Union, Any


class NumpyRowDF:

    def __init__(self, data: np.array, columns: Union[List[str], Dict[str, int]]):
        self.data = data
        if isinstance(columns, list):
            self.columns = {c: i for i, c in enumerate(columns)}
        else:
            self.columns = columns
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        return self.data[self.columns[item]]


class NumpyDF:

    def __init__(self, data: np.array, columns: Union[List[str], Dict[str, int]]):
        self.data = data
        if isinstance(columns, list):
            self.columns = {c: i for i, c in enumerate(columns)}
        else:
            self.columns = columns
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            return NumpyRowDF(self.data[item, :], self.columns)
        if isinstance(item, str):
            return self.data[:, self.columns[item]]
        return self.data[item[1], self.columns[item[0]]]

    def __len__(self):
        return self.data.shape[0]


class EBDataset(Dataset):

    def __init__(self, behavior: pl.DataFrame, history: pl.DataFrame,
                 articles: pl.DataFrame, article_embs: pl.DataFrame,
                 category_embs: pl.DataFrame, limit: Optional[int] = None,
                 emb_col: str = 'embeddings', labels: bool = True):
        if limit is not None:
            print('Limiting history...')
            history = history.with_columns(
                pl.col('article_id_fixed').list.tail(limit),
                pl.col('article_delta_time').list.tail(limit),
                pl.col('impression_time_fixed').list.tail(limit),
                pl.col('impression_weekday').list.tail(limit),
                pl.col('impression_hour').list.tail(limit),
                pl.col('scroll_percentage_fixed').list.tail(limit),
                pl.col('read_time_fixed').list.tail(limit),
            )
        print('Converting format')
        self.behavior = NumpyDF(behavior.to_numpy().copy(), behavior.columns)
        self.history = NumpyDF(history.to_numpy().copy(), history.columns)
        self.articles = NumpyDF(articles.to_numpy().copy(), articles.columns)

        print('Loading embeddings...')
        self.art_emb = torch.from_numpy(np.vstack(article_embs[emb_col].to_numpy()).copy()).float()
        arte_pos_idx = list(article_embs['article_id'])
        art_idx_pos = {art: e for e, art in enumerate(articles['article_id'])}
        sort_pos = [art_idx_pos[art] for art in arte_pos_idx if art in art_idx_pos]
        out_of_order = False
        for e, i in enumerate(sort_pos):
            if e != i:
                out_of_order = True
                break
        if out_of_order:
            print('Out of order!')
            self.art_emb = self.art_emb[sort_pos, :]

        self.cat_emb = torch.from_numpy(np.vstack(category_embs['embeddings'].to_numpy()).copy()).float()
        self.labels = labels
        pass

    def __getitem__(self, idx):
        arts_in_view = self.behavior['article_ids_inview', idx]
        user = self.behavior['user_id', idx]
        arts_in_history = self.history['article_id_fixed', user]
        data = (self.behavior['impression_id', idx],
                (self.get_article_information(arts_in_view),
                 torch.from_numpy(self.behavior['article_delta_time', idx]).long(),
                 self.behavior['impression_weekday', idx],
                 self.behavior['impression_hour', idx],
                 self.behavior['device_type', idx],
                 self.behavior['is_sso_user', idx],
                 self.behavior['gender', idx],
                 self.behavior['postcode', idx],
                 self.behavior['age', idx],
                 self.behavior['is_subscriber', idx]),
                (self.get_article_information(arts_in_history),
                 torch.from_numpy(self.history['article_delta_time', user]).long(),
                 torch.from_numpy(self.history['impression_weekday', user]).long(),
                 torch.from_numpy(self.history['impression_hour', user]).long(),
                 )
                )
        if self.labels:
            return data, (self.behavior['article_ids_clicked', idx],  # indeces within inview
                          self.behavior['next_scroll_percentage', idx])
        return data

    def __len__(self):
        return len(self.behavior)

    def get_article_long(self, col, articles_list: List[int]) -> torch.Tensor:
        return torch.from_numpy(self.articles[col, articles_list].astype(np.int64).copy()).long()

    def get_article_information(self, article_list: List[int]) -> List[Any]:
        category_link = self.get_article_long('category_link', article_list)
        category_embs = self.cat_emb[category_link, ...]
        return (self.art_emb[article_list, ...],
                category_embs,
                self.get_article_long('premium', article_list),
                self.get_article_long('sentiment_label', article_list))


def article_collate(arts_data, *extra):
    embs, cat_embs, premium, sentiment = tuple(zip(*arts_data))
    embs_size = embs[0].shape[1]
    cat_embs_size = cat_embs[0].shape[1]
    in_view_len = [x.shape[0] for x in embs]
    max_len = max(in_view_len)

    r_embs = torch.zeros((len(embs), max_len, embs_size), dtype=torch.float32)
    r_cat_embs = torch.zeros((len(embs), max_len, cat_embs_size), dtype=torch.float32)
    r_premium = torch.zeros((len(embs), max_len), dtype=torch.int64)
    r_sentiment = torch.zeros((len(embs), max_len), dtype=torch.int64)
    mask = torch.zeros((len(embs), max_len), dtype=torch.bool)
    r_extra = tuple([torch.zeros((len(embs), max_len), dtype=torch.int64) for _ in range(len(extra))])
    for i, (e, c, p, s) in enumerate(zip(embs, cat_embs, premium, sentiment)):
        c_len = e.shape[0]
        r_embs[i, :c_len, :] = e
        r_cat_embs[i, :c_len, :] = c
        r_premium[i, :c_len] = p
        r_sentiment[i, :c_len] = s
        mask[i, c_len:] = True
        for j, x in enumerate(extra):
            r_extra[j][i, :c_len] = x[i]

    return in_view_len, (r_embs, r_cat_embs, r_premium, r_sentiment, mask), *r_extra


def pad_behaviour(data):
    articles, article_delta_time, impression_weekday, \
        impression_hour, device_type, is_sso_user, \
        gender, postcode, age, is_subscriber = tuple(zip(*data))

    in_view_len, articles, article_delta_time = article_collate(articles, article_delta_time)
    impression_weekday = torch.from_numpy(np.asarray(impression_weekday, dtype=np.int64)[..., np.newaxis])
    impression_hour = torch.from_numpy(np.asarray(impression_hour, dtype=np.int64)[..., np.newaxis])
    device_type = torch.from_numpy(np.asarray(device_type, dtype=np.int64)[..., np.newaxis])
    is_sso_user = torch.from_numpy(np.asarray(is_sso_user, dtype=np.int64)[..., np.newaxis])
    gender = torch.from_numpy(np.asarray(gender, dtype=np.int64)[..., np.newaxis])
    postcode = torch.from_numpy(np.asarray(postcode, dtype=np.int64)[..., np.newaxis])
    age = torch.from_numpy(np.asarray(age, dtype=np.int64)[..., np.newaxis])
    is_subscriber = torch.from_numpy(np.asarray(is_subscriber, dtype=np.int64)[..., np.newaxis])

    return in_view_len, (articles, article_delta_time, \
                         impression_weekday, impression_hour, device_type, is_sso_user, \
                         gender, postcode, age, is_subscriber)


def pad_history(data):
    articles, article_delta_time, impression_weekday, impression_hour = tuple(zip(*data))
    _, articles, article_delta_time, impression_weekday, impression_hour = article_collate(articles, article_delta_time,
                                                                                           impression_weekday,
                                                                                           impression_hour)
    return articles, article_delta_time, impression_weekday, impression_hour


def pad_inference(data):
    idx, behavior, history = tuple(zip(*data))
    return idx, pad_behaviour(behavior), pad_history(history)


def pad_train(data):
    inference, labels = tuple(zip(*data))
    arts, scrolls = tuple(zip(*labels))
    return pad_inference(inference), (arts, scrolls)
