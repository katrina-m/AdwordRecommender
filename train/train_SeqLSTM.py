import random
from train.parsing_args import parse_SeqLSTM_args
from utility.log_helper import *
from utility.dao_helper import *
from model.SeqLSTM import SeqLSTM
from dao.SeqLSTM_loader import FeatureGen
from dao.load_test_data import load_hot_data
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    df, board_df, keyword_df = load_hot_data()
    featureGen = FeatureGen(args.freq, args.time_span, args.offset, args.train_range, device=args.device)
    loader_train, loader_val = featureGen.prepare_loader(df, board_df, keyword_df, args.batch_size, args.valid_batch_size)
    model = SeqLSTM(args)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    model.fit(loader_train, loader_val, adam_optimizer)


if __name__ == '__main__':
    args = parse_SeqLSTM_args()
    train(args)