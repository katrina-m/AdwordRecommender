import pandas as pd
from datetime import timedelta
import numpy as np
import chinese_calendar
from sklearn.preprocessing import LabelEncoder


def get_interval(freq):
    if freq == "D":
        interval = 1
    elif freq == "7D":
        interval = 7
    else:
        raise NotImplementedError
    return interval


def get_timespan(df, end_date, minus=1, periods=1, freq="D", start_date=None):
    """

    :param df:
    :param end_date:
    :param minus:
    :param periods:
    :param freq:
    :param start_date:
    :return:
    """
    interval = get_interval(freq)

    if start_date is None:
        return df[pd.date_range(end_date - timedelta(days=minus * interval), periods=periods, freq=freq)]
    else:
        return df[pd.date_range(start_date, periods=(end_date - start_date).days / interval, freq=freq)]


def fill_leaks(df, start_date=None, end_date=None, freq="D"):
    """

    :param df:
    :param start_date:
    :param end_date:
    :param freq:
    :return:
    """
    interval = get_interval(freq)

    if start_date is None and end_date is None:
        date_range = pd.date_range(df.columns.min(), periods=(df.columns.max() - df.columns.min()).days / interval,
                                   freq=freq)
    else:
        date_range = pd.date_range(start_date, periods=(end_date - start_date).days / interval, freq=freq)

    if isinstance(df, pd.Series):
        dates = df.index
        leak_dates = set(date_range) - set(dates)
        for date in leak_dates:
            df.loc[date] = np.nan

        df_reindex = df # df.reindex(sorted(df.index), axis=0)
    else:
        dates = df.columns
        leak_dates = set(date_range) - set(dates)
        for date in leak_dates:
            df[date] = np.nan

        df_reindex = df.reindex(sorted(df.columns), axis=1)

    return df_reindex


special_holiday_encoder = LabelEncoder()
special_holiday_encoder.fit(list(chinese_calendar.holidays.values()) + ["normal", "female", "double11"])


def special_holiday_encode(date, freq="D"):
    """
    Verify whether the given date is a holiday or special day like 11.11 or 03.08
    :param date:
    :param freq:
    :return:
    """
    if freq == "D":
        day_type = chinese_calendar.get_holiday_detail(date)[1]
        if day_type:
            return day_type
        elif date.month == 11 and date.day == 11:
            return "double11"
        elif date.month == 3 and date.day == 8:
            return "female"
        else:
            return "normal"

    elif freq == "7D":
        weekdays = range(7)

        for day in weekdays:
            week_day = get_specific_weekday_by_date(date, day)
            day_type = chinese_calendar.get_holiday_detail(week_day)[1]
            if day_type:
                return day_type
            elif week_day.month == 3 and week_day.day == 8:
                return "female"
            elif week_day.month == 11 and week_day.day == 11:
                return "double11"
            continue

        return "normal"


def get_specific_weekday_by_date(time, day = 0):

    diff = day - time.weekday()
    week_day = time +timedelta(days=diff)
    return week_day


def sample_neg_items_for_u(pos_items, start_item_id, end_item_id, n_sample_neg_items, sequential=False):
    """!
    Sample the negative items for a user, if sequential is true, the items are sampled only with respect to one positive item.
    """

    sample_neg_items = []

    if sequential is True:
        for pos_item in pos_items:
            for _ in range(n_sample_neg_items):
                while True:
                    neg_item_id = np.random.randint(low=start_item_id, high=end_item_id, size=1)[0]
                    if neg_item_id != pos_item and neg_item_id not in sample_neg_items:
                        sample_neg_items.append(neg_item_id)
                        break
    else:
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break
            else:
                neg_item_id = np.random.randint(low=start_item_id, high=end_item_id, size=1)[0]
                if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                    sample_neg_items.append(neg_item_id)

    return np.array(sample_neg_items)


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


class NeighborFinder:

    def __init__(self, kg_data, uniform=False, bidirectional=True):
        """
        Params
        ------
        node_idx_list: List[int], contains the list of node index.
        node_ts_list: List[int], contain the list of timestamp for the nodes in node_idx_list.
        off_set_list: List[int], such that node_idx_list[off_set_list[i]:off_set_list[i + 1]] = adjacent_list[i]. \
                Using this can help us quickly find the adjacent node indexes.
        """
        self.bidirectional = bidirectional
        adj_list = self.init_data(kg_data)
        node_idx_l, node_startDay_l, node_endDay_l, s_type_l, d_type_l, edge_type_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_list = node_idx_l
        self.node_startDay_list = node_startDay_l
        self.node_endDay_list = node_endDay_l
        self.edge_type_list = edge_type_l
        self.src_type_list = s_type_l
        self.dst_type_list = d_type_l
        self.off_set_list = off_set_l

        self.uniform = uniform

    def init_data(self, kg_data):
        src_idx_list = kg_data.h
        dst_idx_list = kg_data.t
        e_type_list = kg_data.r
        h_type_list = kg_data.r
        t_type_list = kg_data.r
        startDay_list = kg_data.StartDay.values
        endDay_list = kg_data.EndDay.values
        max_idx = max(max(src_idx_list), max(dst_idx_list))

        # The graph is bi-directional
        if self.bidirectional is True:
            adj_list = [[] for _ in range(max_idx + 1)]
            for src, dst, hType, tType, eType, startDay, endDay in zip(src_idx_list, dst_idx_list, h_type_list, t_type_list, e_type_list, startDay_list, endDay_list):
                adj_list[src].append((dst, hType, tType, eType, startDay, endDay))
                adj_list[dst].append((src, tType, hType, eType, startDay, endDay))
        else:
            adj_list = [[] for _ in range(max_idx + 1)]
            for src, dst, hType, tType, eType, startDay, endDay in zip(src_idx_list, dst_idx_list, h_type_list, t_type_list, e_type_list, startDay_list, endDay_list):
                adj_list[src].append((dst, hType, tType, eType, startDay, endDay))
        return adj_list

    def init_off_set(self, adj_list):
        """
        Params ------ Input: adj_list: List[List[(node_idx, edge_idx, node_ts)]], the inner list at each index is the
        adjacent node info of the node with the given index.

        Return:
            n_idx_list: List[int], contain the node index.
            n_ts_list: List[int], contain the timestamp of node index.
            e_idx_list: List[int], contain the edge index.
            off_set_list: List[int], such that node_idx_list[off_set_list[i]:off_set_list[i + 1]] = adjacent_list[i]. \
                Using this can help us quickly find the adjacent node indexes.
        """
        n_idx_list = []
        n_startDay_list = []
        n_endDay_list = []
        s_type_list = []
        d_type_list = []
        e_type_list = []
        off_set_list = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[5]) # Sort by endDay
            n_idx_list.extend([x[0] for x in curr])
            s_type_list.extend([x[1] for x in curr])
            d_type_list.extend([x[2] for x in curr])
            e_type_list.extend([x[3] for x in curr])
            n_startDay_list.extend([x[4] for x in curr])
            n_endDay_list.extend([x[5] for x in curr])

            off_set_list.append(len(n_idx_list))
        n_idx_list = np.array(n_idx_list)
        s_type_list = np.array(s_type_list)
        d_type_list = np.array(d_type_list)
        n_startDay_list = np.array(n_startDay_list)
        n_endDay_list = np.array(n_endDay_list)
        e_type_list = np.array(e_type_list)
        off_set_list = np.array(off_set_list)

        assert(len(n_idx_list) == len(n_startDay_list))
        assert(off_set_list[-1] == len(n_startDay_list))

        return n_idx_list, n_startDay_list, n_endDay_list, s_type_list, d_type_list, e_type_list, off_set_list

    def find_before(self, src_idx, cut_time=None, sort_by_time=True):
        """
        Find the neighbors for src_idx with edge time right before the cut_time.
        Params
        ------
        Input:
            src_idx: int
            cut_time: float
        Return:
            neighbors_idx: List[int]
            neighbors_e_idx: List[int]
            neighbors_ts: List[int]
        """
        node_idx_list = self.node_idx_list
        src_type_list = self.src_type_list
        dst_type_list = self.dst_type_list
        edge_type_list = self.edge_type_list
        off_set_list = self.off_set_list
        node_startDay_list = self.node_startDay_list
        node_endDay_list = self.node_endDay_list

        neighbors_endDay = node_endDay_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_startDay = node_startDay_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_idx = node_idx_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_e_type = edge_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_src_type = src_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_dst_type = dst_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]

        if sort_by_time is False:
            return neighbors_idx, neighbors_src_type, neighbors_dst_type, neighbors_e_type, neighbors_startDay, neighbors_endDay

        # If no neighbor find, returns the empty list.
        if len(neighbors_idx) == 0:
            return neighbors_idx, neighbors_src_type, neighbors_dst_type, neighbors_e_type, neighbors_startDay, neighbors_endDay

        # Find the neighbors which has timestamp < cut_time.
        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_endDay[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_endDay[right] < cut_time:
            return neighbors_idx[:right], neighbors_src_type[:right], neighbors_dst_type[:right], neighbors_e_type[:right], neighbors_startDay[:right], neighbors_endDay[:right]
        else:
            return neighbors_idx[:left], neighbors_src_type[:left], neighbors_dst_type[:left], neighbors_e_type[:left], neighbors_startDay[:left], neighbors_endDay[:left]

    def get_temporal_neighbor(self, src_idx_list, cut_time_list, num_neighbors=20, sort_by_time=True):
        """
        Find the neighbor nodes before cut_time in batch.
        Params
        ------
        Input:
            src_idx_list: List[int]
            cut_time_list: List[float],
            num_neighbors: int
        Return:
            out_ngh_node_batch: int32 matrix (len(src_idx_list), num_neighbors)
            out_ngh_t_batch: int32 matrix (len(src_idx_list), num_neighbors)
            out_ngh_eType_batch: int32 matrix (len(src_idx_list), num_neighbors)
            out_ngh_sType_batch: int32 matrix (len(src_type_list), num_neighbors)
            out_ngh_dType_batch: int32 matrix (len(dst_type_list), num_neighbors)
        """
        out_ngh_node_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_startDay_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.float32)
        out_ngh_endDay_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.float32)
        out_ngh_eType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_sType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_dType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_list, cut_time_list)):
            ngh_idx, ngh_sType, ngh_dType, ngh_eType, ngh_startDay, ngh_endDay = self.find_before(src_idx, cut_time, sort_by_time)
            ngh_endDay[ngh_endDay == 0] = cut_time
            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_startDay_batch[i, :] = ngh_startDay[sampled_idx]
                    out_ngh_endDay_batch[i, :] = ngh_endDay[sampled_idx]
                    out_ngh_sType_batch[i, :] = ngh_sType[sampled_idx]
                    out_ngh_dType_batch[i, :] = ngh_dType[sampled_idx]
                    out_ngh_eType_batch[i, :] = ngh_eType[sampled_idx]

                    # resort based on time
                    pos = out_ngh_endDay_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_sType_batch = out_ngh_sType_batch[i, :][pos]
                    out_ngh_dType_batch = out_ngh_dType_batch[i, :][pos]
                    out_ngh_startDay_batch[i, :] = out_ngh_startDay_batch[i, :][pos]
                    out_ngh_endDay_batch[i, :] = out_ngh_endDay_batch[i, :][pos]
                    out_ngh_eType_batch[i, :] = out_ngh_eType_batch[i, :][pos]
                else:
                    ngh_startDay = ngh_startDay[:num_neighbors]
                    ngh_endDay = ngh_endDay[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eType = ngh_eType[:num_neighbors]
                    ngh_sType = ngh_sType[:num_neighbors]
                    ngh_dType = ngh_dType[:num_neighbors]

                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_startDay) <= num_neighbors)
                    assert(len(ngh_endDay) <= num_neighbors)
                    assert(len(ngh_eType) <= num_neighbors)
                    assert(len(ngh_sType) <= num_neighbors)
                    assert(len(ngh_dType) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_sType_batch[i, num_neighbors - len(ngh_sType):] = ngh_sType
                    out_ngh_dType_batch[i, num_neighbors - len(ngh_dType):] = ngh_dType
                    out_ngh_startDay_batch[i, num_neighbors - len(ngh_startDay):] = ngh_startDay
                    out_ngh_endDay_batch[i, num_neighbors - len(ngh_endDay):] = ngh_endDay
                    out_ngh_eType_batch[i,  num_neighbors - len(ngh_eType):] = ngh_eType

        return out_ngh_node_batch, out_ngh_sType_batch, out_ngh_dType_batch, out_ngh_eType_batch, out_ngh_startDay_batch, out_ngh_endDay_batch

    def find_k_hop_temporal(self, src_idx_l, cut_time_l=None, fan_outs=[15], sort_by_time=True):
        """Sampling the k-hop sub graph before the cut_time
        """

        x, s, d, y, start, end = self.get_temporal_neighbor(src_idx_l, cut_time_l, fan_outs[0], sort_by_time=sort_by_time)
        node_records = [x]
        sType_records = [s]
        dType_records = [d]
        eType_records = [y]
        startDay_records = [start]
        endDay_records = [end]

        for i in range(1, len(fan_outs)):
            ngn_node_est, ngh_endDay_est = node_records[-1], endDay_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_endDay_est = ngh_endDay_est.flatten()
            out_ngh_node_batch, out_ngh_sType_batch, out_ngh_dType_batch, out_ngh_eType_batch, out_ngh_startDay_batch, \
                        out_ngh_endDay_batch = self.get_temporal_neighbor(ngn_node_est, ngn_endDay_est, fan_outs[i])
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, fan_outs[i]) # [N, *([num_neighbors] * k)]
            out_ngh_sType_batch = out_ngh_sType_batch.reshape(*orig_shape, fan_outs[i]) # [N, *([num_neighbors] * k)]
            out_ngh_dType_batch = out_ngh_dType_batch.reshape(*orig_shape, fan_outs[i]) # [N, *([num_neighbors] * k)]
            out_ngh_eType_batch = out_ngh_eType_batch.reshape(*orig_shape, fan_outs[i])
            out_ngh_startDay_batch = out_ngh_startDay_batch.reshape(*orig_shape, fan_outs[i])
            out_ngh_endDay_batch = out_ngh_endDay_batch.reshape(*orig_shape, fan_outs[i])

            node_records.append(out_ngh_node_batch)
            sType_records.append(out_ngh_sType_batch)
            dType_records.append(out_ngh_dType_batch)
            eType_records.append(out_ngh_eType_batch)
            startDay_records.append(out_ngh_startDay_batch)
            endDay_records.append(out_ngh_endDay_batch)
        return node_records, sType_records, dType_records, eType_records, startDay_records, endDay_records



