import torch
import logging
from utility.model_helper import EarlyStopping, adjust_learning_rate
from time import time
from tqdm import tqdm
import numpy as np
import abc


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.__dict__.update(vars(args))

    @abc.abstractmethod
    def update_loss(self, optimizer, batch_data):
        pass

    @abc.abstractmethod
    def reset_parameter(self):
        pass

    def fit(self, loader_train, loader_val, optimizer):
        self.reset_parameter()

        earlyStopper = EarlyStopping(self.stopping_steps, self.verbose)
        self.train().to(device=self.device)

        logging.info(self)

        n_batch = len(loader_train)
        for epoch in range(0, self.num_epochs + 1):

            time1 = time()
            total_loss = 0
            time2 = time()
            for step, batch_data in enumerate(loader_train):
                loss = self.update_loss(optimizer, batch_data)
                total_loss += loss.item()
                if self.verbose and step % self.print_every == 0 and step != 0:
                    logging.info(
                        'Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
                        'Loss {:.4f}'.format(
                            epoch, step, n_batch, time() - time2, loss.item(), total_loss / step))
                    time2 = time()
            logging.info(
                'Training: Epoch {:04d} Total Records {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                               n_batch,
                                                                                                               time() - time1,
                                                                                                               total_loss / step))
            if epoch % self.evaluate_every == 0:
                time1 = time()
                self.eval()
                ndcg, recall = self.evaluate(loader_val)
                logging.info(
                    'Evaluation: Epoch {:04d} | Total Time {:.1f}s | Recall {:.4f} NDCG {'':.4f}'.format(
                        epoch, time() - time1, recall, ndcg))

                earlyStopper(recall, self, self.save_dir, epoch)

                if earlyStopper.early_stop:
                    break
                self.train()

            adjust_learning_rate(optimizer, epoch, self.lr)

    def evaluate(self, loader_val):

        n_batch = len(loader_val)
        NDCG = 0
        HT = 1
        with torch.no_grad():
            with tqdm(total=len(loader_val), desc='Evaluating Iteration') as pbar:
                for features in loader_val:
                    predictions = -self.predict(features).transpose(0, 1)
                    rank = predictions[0].argsort().argsort()[0].item()

                    if rank < self.K:
                        NDCG += 1/np.log2(rank+2)
                        HT += 1

                    pbar.update(1)

        return NDCG/n_batch, HT/n_batch


