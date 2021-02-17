import argparse
import torch


def common_args(parser):
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--data_name', nargs='?', default='hot', help='Choose a dataset from {hot, history}')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=20, help='Number of epoch for early stopping')
    parser.add_argument('--print_every', type=int, default=10, help='Iter interval of printing CF loss.')
    parser.add_argument('--evaluate_every', type=int, default=1, help='Epoch interval of evaluating CF.')
    parser.add_argument('--K', type=int, default=10, help='Calculate metric@K when evaluating.')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose.')
    return parser


def parse_SeqLSTM_args():
    parser = argparse.ArgumentParser(description="Run SeqLSTM.")
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--valid_batch_size', default=256, type=int, help='Valid batch size')
    parser.add_argument('--time_span', default=200, type=int, help='Max sequence lengths')
    parser.add_argument('--freq', default="7D", type=str)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--offset', default=0, type=float)
    parser.add_argument('--train_range', default=100, type=int)
    parser.add_argument('--hidden_units', default=16, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout rate.")
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--l2', default=0.0, type=float)
    parser.add_argument('--use_time', type=str, default="time", choices=['time', 'pos', 'empty'], help='number of neighbors')

    parser = common_args(parser)
    args = parser.parse_args()

    save_dir = '../trained_model/SeqLSTM/{}/hiddendim{}_lr{}/'.format(args.data_name, args.hidden_units, \
                                                                                         args.lr)
    args.save_dir = save_dir
    return args


def parse_SeqGF_args():
    parser = argparse.ArgumentParser(description="Run SeqGF.")
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--valid_batch_size', default=256, type=int, help='Valid batch size')
    parser.add_argument('--fan_outs', default=[20, 15], type=int, help='Max sequence lengths')
    parser.add_argument('--hidden_units', default=16, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout rate.")
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--use_time', type=str, default="time", choices=['time', 'pos', 'empty'], help='number of neighbors')

    parser = common_args(parser)
    args = parser.parse_args()

    save_dir = '../trained_model/SeqGF/{}/hiddendim{}_numblocks{}_numheads{}_usertime()_fanouts{}_lr{}/'.format(args.data_name, \
                                                        args.hidden_units, args.num_blocks, args.num_heads, args.use_time, "-".join(args.fan_outs), args.lr)

    args.save_dir = save_dir
    return args