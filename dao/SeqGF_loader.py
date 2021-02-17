from utility.dao_helper import get_timespan, fill_leaks, get_interval, special_holiday_encoder, special_holiday_encode, sample_neg_items_for_u
import numpy as np
import pandas as pd
from datetime import timedelta
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader
import torch


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
    time_features = [month_day, month, year, holiday]
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

    def __init__(self, fan_outs, device="cpu"):
        self.device = device
        self.fan_outs = fan_outs

    def prepare_loader(self, kg_df, batch_size, valid_batch_size):
        val_periods = 5
        val_start_time = kg_df.StartDay.max()-timedelta(self.interval * val_periods)
        val_dates = [val_start_time + timedelta(self.interval * i) for i in range(val_periods)]
        train_dates = [val_start_time - timedelta(self.interval * i) for i in range(1, self.train_range)]
        board_keyword_relation_data = kg_df.query("relation == 'Board-Keyword'")
        train_data = board_keyword_relation_data.query(f"EndDay < {val_start_time}")
        val_data = board_keyword_relation_data.query(f"EndDay >= {val_start_time}")
        rest_node_data = kg_df.query("relation != 'Board-Keyword'")
        train_graph = pd.concat([train_data, rest_node_data])
        val_graph = pd.concat([val_data, rest_node_data])
        keywordID_range = (board_keyword_relation_data.EndID.min(), board_keyword_relation_data.EndID.max())
        train_dataset = SeqGFDataset(train_data, train_graph, keywordID_range, self.fan_outs)
        val_dataset = SeqGFDataset(val_data, val_graph, keywordID_range, self.fan_outs)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=valid_batch_size, collate_fn=val_dataset.collate_fn)
        return train_dataloader, val_dataloader


class SeqGFDataset(Dataset):

    def __init__(self, data, graph, keywordID_range, fan_outs):
        self.data = data
        self.ngh_finder = graph
        self.keywordID_range = keywordID_range
        self.fan_outs = fan_outs
        pass

    def collate(self, batch):

        boardIDs, keywordIDs, neg_keywordIDs, StartDays, EndDays = zip(*batch)
        board_blocks = self.ngh_finder.find_k_hop_temporal(self, boardIDs, cut_time_l=EndDays, fan_outs=self.fan_outs, sort_by_time=True)
        keyword_blocks = self.ngh_finder.find_k_hop_temporal(self, keywordIDs, cut_time_l=EndDays, fan_outs=self.fan_outs, sort_by_time=True)
        neg_keyword_blocks = self.ngh_finder.find_k_hop_temporal(self, neg_keywordIDs, cut_time_l=EndDays, fan_outs=self.fan_outs, sort_by_time=True)
        return board_blocks, keyword_blocks, neg_keyword_blocks

    def convert_block_to_tensor(self, seeds, blocks):
        block_tensors = []
        seeds = torch.LongTensor(np.array(seeds)).to(self.device)
        for i, block in enumerate(blocks):
            ngh_batch, ngh_src_type, ngh_dst_type, ngh_edge_type, ngh_startDay, ngh_endDay = zip(*block)
            ngh_batch = torch.LongTensor(ngh_batch).to(self.device)
            ngh_src_type = torch.LongTensor(ngh_src_type).to(self.device)
            ngh_dst_type = torch.LongTensor(ngh_dst_type).to(self.device)
            ngh_edge_type = torch.LongTensor(ngh_edge_type).to(self.device)
            ngh_startDay = torch.FloatTensor(ngh_startDay).to(self.device)
            ngh_endDay = torch.FloatTensor(ngh_endDay).to(self.device)
            block_tensors.append((ngh_batch.view(-1, self.fan_outs[i]), \
                                  seeds.flatten(), \
                                  ngh_src_type.view(-1, self.fan_outs[i]), \
                                  ngh_dst_type.view(-1, self.fan_outs[i]), \
                                  ngh_edge_type.view(-1, self.fan_outs[i]), \
                                  ngh_startDay.view(-1, self.fan_outs[i]), \
                                  ngh_endDay.view(-1, self.fan_outs[i])))
            seeds = ngh_batch.view(-1)

        return block_tensors

    def __getitem__(self, index):
        record = self.data.iloc[index, :]
        StartDay = record.StartDay.timestamp()
        EndDay = record.EndDay.timestamp()
        boardID = record.StartID
        keywordID = record.EndID
        neg_keywordID = sample_neg_items_for_u([keywordID], self.keywordID_range[0], self.keywordID_range[1], 1)
        return boardID, keywordID, neg_keywordID, StartDay, EndDay




