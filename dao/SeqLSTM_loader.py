from utility.dao_helper import get_timespan, fill_leaks, get_interval, special_holiday_encoder, special_holiday_encode
import numpy as np
import pandas as pd
from datetime import timedelta
import torch
from keras_preprocessing.sequence import pad_sequences


def create_time_datasets(board_df, keyword_df, df_reindex, pred_start_time, time_span, freq):
    interval = get_interval(freq)

    # main performance field for time series
    df_reindex = fill_leaks(df_reindex, pred_start_time - timedelta(days=time_span * interval), pred_start_time + timedelta(days=1 * interval), freq = freq)

    # added as a mask for main time series
    isnull_X = get_timespan(df_reindex.isnull(), pred_start_time, time_span, time_span, freq=freq).astype(
        int).values.reshape(-1, time_span)
    X = get_timespan(df_reindex, pred_start_time, time_span, time_span, freq=freq).fillna(0).values.reshape(-1, time_span, 1)
    month_day = np.tile([d.day - 1 for d in
                         pd.date_range(pred_start_time - timedelta(days=time_span * interval), periods=time_span + 1,
                                       freq=freq)],
                        (X.shape[0], 1))

    month = np.tile([d.month - 1 for d in
                     pd.date_range(pred_start_time - timedelta(days=time_span * interval), periods=time_span + 1,
                                   freq=freq)],
                    (X.shape[0], 1))
    year = np.tile([d.year - 1 for d in
                    pd.date_range(pred_start_time - timedelta(days=time_span * interval), periods=time_span + 1,
                                  freq=freq)],
                   (X.shape[0], 1))
    year = year - np.min(year)
    holiday = np.tile([special_holiday_encoder.transform([special_holiday_encode(d, freq=freq)])[0] for \
                       d in pd.date_range(pred_start_time - timedelta(days=time_span * interval), periods=time_span + 1,
                                          freq=freq)],
                      (X.shape[0], 1))
    timestamp = np.tile([d.timestamp() - 1 for d in
                         pd.date_range(pred_start_time - timedelta(days=time_span * interval), periods=time_span + 1,
                                       freq=freq)],
                        (X.shape[0], 1))

    time_features = [month_day, month, year, holiday, timestamp]
    y = df_reindex[pd.date_range(pred_start_time, periods=1, freq=freq)].fillna(0).values.reshape(-1)

    board_features = []
    board_features.append(board_df["Category"].values)
    board_features.append(board_df["Sub_category"].values)

    keyword_features = []
    rootwordIds = keyword_df["Rootword_ids"].values
    maxlen = np.max([len(rwIds) for rwIds in rootwordIds])
    #sequences = [eval(rwIds) for rwIds in rootwordIds]
    keyword_features.append(pad_sequences(rootwordIds, maxlen=maxlen, padding='pre', value=0.0))
    return (X, isnull_X, *time_features, *board_features, *keyword_features), y


class FeatureGen(object):

    def __init__(self, freq, time_span, offset, train_range, device="cpu"):
        self.freq = freq
        self.time_span = time_span
        self.offset = offset
        self.interval = get_interval(freq)
        self.device = device
        self.train_range = train_range

    def prepare_loader(self, df, board_df, keyword_df, batch_size, valid_batch_size):
        val_periods = 5
        val_start_time = df.Date.max()-timedelta(self.interval * val_periods)
        val_dates = [val_start_time + timedelta(self.interval * i) for i in range(val_periods)]
        train_dates = [val_start_time - timedelta(self.interval * i) for i in range(1, self.train_range)]
        loader_train = LSTMDataloader(df, board_df, keyword_df, train_dates, field="点击率", freq=self.freq, time_span = self.time_span, \
                                      offset=self.offset, device=self.device, mode="train", batch_size = batch_size)
        loader_val = LSTMDataloader(df, board_df, keyword_df, val_dates, field="点击率", freq=self.freq, time_span = self.time_span,\
                                    offset=self.offset, device=self.device, mode="valid", batch_size = valid_batch_size)
        return loader_train, loader_val


class LSTMDataloader(object):

    def __init__(self, df, board_df, keyword_df, dates, field, batch_size, freq="7D", time_span=365, offset=1.2, device="cpu", mode="train"):
        super().__init__()
        self.freq = freq
        self.interval = get_interval(freq)
        self.batch_size = batch_size
        self.time_span = time_span
        self.df = df
        self.keyword_df = keyword_df.set_index("KeywordID")
        self.board_df = board_df.set_index("BoardID")
        self.field = field
        self.device = device
        self.offset = offset
        self.mode = mode
        self.dates = dates

        self.df_reindex = self.df.set_index(["BoardID", "KeywordID", "Date"])[self.field].unstack(-1)
        self.name_index = self.df_reindex.index
        self.df_reindex = self.df_reindex.reset_index(drop=True)

        self.n_batch = 200
        self.val_n_batch = 200

    def prepare_datasets(self, pred_start_time, board_df_tmp, keyword_df_tmp, df_reindex_tmp):
        return create_time_datasets(board_df_tmp, keyword_df_tmp, df_reindex_tmp, pred_start_time, self.time_span, self.freq)

    def __iter__(self):
        if self.mode == 'train':
            count = 0
            while count < self.n_batch:
                date = np.random.choice(self.dates, 1)[0]
                if date not in self.df_reindex:
                    continue

                # Pre-filtering logic
                # valid_dates = self.df_reindex.columns.intersection(pd.date_range(date-timedelta(self.interval *
                # len(self.dates)), date, freq=self.freq)) df_reindex_valid = self.df_reindex[(self.df_reindex[
                # valid_dates] > 0).sum(axis=1) > 0]

                df_reindex_valid = self.df_reindex
                pos_df_index = df_reindex_valid[df_reindex_valid[date] > self.offset].index  # Recommend keyword with the impression bigger than offset.
                neg_df_index = list(set(df_reindex_valid.index) - set(pos_df_index))  # Used for negative sampling.
                # Randomly select keywords which have implicit performance.
                keep_pos_idx = np.random.choice(pos_df_index, size=self.batch_size)
                keep_neg_idx = np.random.choice(neg_df_index, size=self.batch_size)

                idx = np.concatenate([keep_pos_idx, keep_neg_idx])
                df_reindex_tmp = self.df_reindex.loc[idx, :]

                board_index = self.name_index[idx].get_level_values(0)
                board_df_tmp = self.board_df.loc[board_index, :]
                keyword_index = self.name_index[idx].get_level_values(1)
                keyword_df_tmp = self.keyword_df.loc[keyword_index, :]
                features, targets = self.prepare_datasets(date, board_df_tmp, keyword_df_tmp, df_reindex_tmp)

                # Convert to tensors.
                pos_feature_list = []
                neg_feature_list = []
                neg_start_idx = len(keep_pos_idx)
                for pos, neg in zip(range(neg_start_idx), range(neg_start_idx, neg_start_idx+len(keep_neg_idx))):
                    pos_feature = self.extract_feature(features, pos)
                    neg_feature = self.extract_feature(features, neg)
                    pos_feature_list.append(pos_feature)
                    neg_feature_list.append(neg_feature)
                pos_feature_tensors = self.convert_to_tensor(pos_feature_list)
                neg_feature_tensors = self.convert_to_tensor(neg_feature_list)
                yield pos_feature_tensors, neg_feature_tensors
                count += 1
        else:
            count = 0
            while count < self.val_n_batch:
                date = np.random.choice(self.dates, 1)[0]

                # Some dates have no records.
                if date not in self.df_reindex:
                    continue

                # 1 pos item + 100 neg item
                pos_df_index = self.df_reindex[self.df_reindex[date] > self.offset].index
                neg_df_index = self.df_reindex[self.df_reindex[date] <= self.offset].index
                keep_pos_idx = np.random.choice(pos_df_index, size=1)
                keep_neg_idx = np.random.choice(neg_df_index, size=100)
                idx = np.concatenate([keep_pos_idx, keep_neg_idx])

                # Sample the partial datasets.
                df_reindex_tmp = self.df_reindex.loc[idx, :]
                board_index = self.name_index[idx].get_level_values(0)
                board_df_tmp = self.board_df.loc[board_index, :]
                keyword_index = self.name_index[idx].get_level_values(1)
                keyword_df_tmp = self.keyword_df.loc[keyword_index, :]

                # Create features.
                features, targets = self.prepare_datasets(date, board_df_tmp, keyword_df_tmp, df_reindex_tmp)

                # Convert to tensors.
                feature_list = []
                for i in range(len(targets)):
                    feature_list.append(self.extract_feature(features, i))
                feature_tensors = self.convert_to_tensor(feature_list)
                yield feature_tensors
                count = count + 1

    def extract_feature(self, features, index):
        feature_pair = [feature[index] for feature in features]
        X, isnull_X, month_day, month, year, holiday, timestamp, Categories, Sub_categories, Rootwords = feature_pair
        return X, isnull_X, month_day, month, year, holiday, timestamp, Categories, Sub_categories, Rootwords

    def convert_to_tensor(self, feature_list):
        X, isnull_X, month_day, month, year, holiday, timestamp, Categories, Sub_categories, Rootwords = zip(*feature_list)
        X = torch.FloatTensor(np.stack(X, axis=0)).to(self.device)
        isnull_X = torch.BoolTensor(np.stack(isnull_X, axis=0)).to(self.device)
        month_day = torch.LongTensor(np.stack(month_day, axis=0)).to(self.device)
        month = torch.LongTensor(np.stack(month, axis=0)).to(self.device)
        year = torch.LongTensor(np.stack(year, axis=0)).to(self.device)
        holiday = torch.LongTensor(np.stack(holiday, axis=0)).to(self.device)
        timestamp = torch.LongTensor(np.stack(timestamp, axis=0)).to(self.device)
        categories = torch.LongTensor(np.stack(Categories, axis=0)).to(self.device)
        Sub_categories = torch.LongTensor(np.stack(Sub_categories, axis=0)).to(self.device)
        Rootwords = torch.LongTensor(np.stack(Rootwords, axis=0)).to(self.device)
        return X, isnull_X, month_day, month, year, holiday, timestamp, categories, Sub_categories, Rootwords

    def __len__(self):
        if self.mode == "train":
            return self.n_batch
        else:
            return self.val_n_batch

#
# class LSTMDataset(Dataset):
#
#     def __init__(self, df, valid_start_time, field, freq="7D", time_span=365, offset=1.2, train_range=-1, device="cpu", mode="train"):
#         super().__init__()
#         self.freq = freq
#         self.interval = get_interval(freq)
#         self.time_span = time_span
#         self.df = df
#         self.field = field
#         self.device = device
#         self.offset = offset
#         self.mode = mode
#         self.valid_start_time = valid_start_time
#         self.train_range = train_range
#         self.df_reindex = self.df.set_index(["BoardID", "KeywordID", "Date"])[self.field].unstack(-1)
#         self.df_reindex = fill_leaks(self.df_reindex, start_date=self.valid_start_time - timedelta(self.interval * (self.train_range-1)), \
#                                      end_date=self.valid_start_time, freq=self.freq).fillna(-1)
#         condition = (self.df_reindex[pd.date_range(self.valid_start_time - timedelta(self.interval * (self.train_range-1)), \
#                                                    self.valid_start_time, freq=self.freq)] > 0).sum(axis=1) > 0
#         self.df_reindex_valid = self.df_reindex[condition]
#
#     def __getitem__(self, idx):
#         series = self.df_reindex_valid.iloc[idx]
#         pos_index = self.df_reindex_valid.index[idx]
#
#         if self.mode == "train":
#
#             neg_features = []
#             pos_features = []
#             count = 0
#             for i in reversed(range(len(series))):
#                 value = series[i]
#                 date = series.index[i]
#                 if date < self.valid_start_time:
#                     if value > self.offset:
#                         pos_feature = self.extract_features(series, date)
#                         neg_feature = self.sample_neg_series(self.df_reindex, pos_index, date)
#                         pos_features.append(pos_feature)
#                         neg_features.append(neg_feature)
#                 count += 1
#
#                 if self.train_range != -1 and count >= self.train_range:
#                     break
#             return [np.array(feature) for feature in zip(*pos_features)], [np.array(feature) for feature in zip(*neg_features)]
#         else:
#             pos_features = []
#             for i in reversed(range(len(series))):
#                 value = series[i]
#                 date = series.index[i]
#                 if date < self.valid_start_time:
#                     break
#
#                 if value > self.offset:
#                     pos_feature = self.extract_features(series, date)
#                     pos_features.append(pos_feature)
#             return pos_features
#
#     def extract_features(self, series, pred_start_time):
#         # series = fill_leaks(series, pred_start_time - timedelta(days=self.time_span * self.interval), \
#         #                         pred_start_time + timedelta(days=1 * self.interval), freq = self.freq)
#         seq = np.zeros(self.time_span)-999
#         dates = pd.date_range(pred_start_time - timedelta(self.interval * (self.time_span-1)), pred_start_time, freq="7D")
#         cur_pos = len(series) - 1
#         s_date = series.index[-1]
#
#         while s_date > dates[-1]:
#             cur_pos = cur_pos - 1
#             s_date = series.index[cur_pos]
#         s_value = series[cur_pos]
#
#         for idx in reversed(range(len(dates))):
#             date = dates[idx]
#             if date == s_date and s_value >= 0:
#                 seq[idx] = s_value
#                 cur_pos = cur_pos - 1
#                 s_date = series.index[cur_pos]
#                 s_value = series[cur_pos]
#
#
#
#         #series = series.loc[pd.date_range(pred_start_time - timedelta(self.interval * self.time_span), pred_start_time, freq="7D")]
#         #isnull_seq = series.isnull().astype(int).values
#         isnull_seq = (seq < 0).astype(int)
#         seq[np.where(seq < 0)] = 0
#         #seq = series.values
#         return seq, isnull_seq
#
#     def sample_neg_series(self, df_reindex, pos_index, pred_start_time):
#         series = df_reindex.loc[pos_index[0]].query("`{}` < {}".format(pred_start_time, self.offset)).sample(n=1).iloc[0]
#         features = self.extract_features(series, pred_start_time)
#         return features
#
#     def collate_fn(self, batch):
#         if self.mode == "train":
#             pos_features, neg_features = zip(*batch)
#             pos_features = list(filter(None, pos_features))
#             neg_features = list(filter(None, neg_features))
#             if len(pos_features) == 0:
#                 return None
#             pos_feature_tensors = self.convert_to_tensor(pos_features)
#             neg_feature_tensors = self.convert_to_tensor(neg_features)
#             return pos_feature_tensors, neg_feature_tensors
#
#     def convert_to_tensor(self, features):
#         features = list(filter(None, features))
#         X = [feature[0] for feature in features]
#         isnull_X = [feature[1] for feature in features]
#
#         feature_tensors = [torch.FloatTensor(np.concatenate(X, axis=0)).unsqueeze(-1).to(self.device),\
#                            torch.FloatTensor(np.concatenate(isnull_X, axis=0)).to(self.device)]
#         return feature_tensors
#
#     def __len__(self):
#         return len(self.df_reindex_valid)
#
#
