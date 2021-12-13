import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import numpy as np
import pickle

import config
from data_utils.vocab import Vocab
from model.mcan import MCAN
from data_utils.vivqa_extracted_features import ViVQA, get_loader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker

import os


total_iterations = 0
metrics = Metrics()

def run(net, loaders, fold_idx, stage, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    for loader in loaders[fold_idx:]:
        tq = tqdm(loader, desc='Epoch {:03d} - {} - Fold {}'.format(epoch, prefix, loaders.index(loader)+1), ncols=0)

        if train:
            loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params)) 
        else:
            acc_tracker = tracker.track('{}_accuracy'.format(prefix), tracker_class(**tracker_params))
            pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
            rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
            f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))

        loss_objective = nn.CrossEntropyLoss(label_smoothing=0.2).cuda()
        for v, q, a in tq:
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()

            out = net(v, q)
            scores = metrics.get_scores(out.cpu(), a.cpu())

            if train:
                optimizer.zero_grad()
                loss = loss_objective(out, a)
                loss_tracker.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                loss = np.array(0)
                acc_tracker.append(scores["accuracy"])
                pre_tracker.append(scores["precision"])
                rec_tracker.append(scores["recall"])
                f1_tracker.append(scores["F1"])

            fmt = '{:.4f}'.format
            if train:
                tq.set_postfix(loss=fmt(loss.item()))
            else:
                tq.set_postfix(accuracy=fmt(acc_tracker.mean.value), 
                                precision=fmt(pre_tracker.mean.value), recall=fmt(rec_tracker.mean.value), f1=fmt(f1_tracker.mean.value))

            tq.update()

        torch.save({
            "fold": loaders.index(loader),
            "epoch": epoch,
            "stage": stage,
            "loss": loss_tracker.mean.value,
            "weights": net.state_dict()
        }, os.path.join(config.tmp_model_checkpoint, "last_model.pth"))

    if not train:
        return {
            "accuracy": acc_tracker.mean.value,
            "precision": pre_tracker.mean.value,
            "recall": rec_tracker.mean.value,
            "F1": f1_tracker.mean.value
        }
    else:
        return loss_tracker.mean.value


def main():

    cudnn.benchmark = True

    if os.path.isfile(os.path.join(config.model_checkpoint, "vocab.pkl")):
        vocab = pickle.load(open(os.path.join(config.model_checkpoint, "vocab.pkl"), "rb"))
    else:
        vocab = Vocab([config.json_train_path, config.json_test_path], 
                            specials=["<pad>", "<sos", "<eos>"])
        pickle.dump(vocab, open(os.path.join(config.model_checkpoint, "vocab.pkl"), "wb"))

    metrics.vocab = vocab
    train_dataset = ViVQA(config.json_train_path, config.preprocessed_path, vocab)
    test_dataset = ViVQA(config.json_test_path, config.preprocessed_path, vocab)

    if os.path.isfile(os.path.join(config.model_checkpoint, "folds.pkl")):
        folds, test_fold = pickle.load(open(os.path.join(config.model_checkpoint, "folds.pkl"), "rb"))
    else:
        folds, test_fold = get_loader(train_dataset, test_dataset)
        pickle.dump((folds, test_fold), open(os.path.join(config.model_checkpoint, "folds.pkl"), "wb"))

    if config.start_from:
        saved_info = torch.load(config.start_from)
        from_epoch = saved_info["epoch"]
        from_stage = saved_info["stage"]
        from_fold = saved_info["fold"] + 1
        loss = saved_info["loss"]
        net = nn.DataParallel(MCAN(vocab, config.backbone, config.d_model, config.embedding_dim, config.dff, config.nheads, 
                                    config.nlayers, config.dropout)).cuda()
        net.load_state_dict(saved_info["weights"])
    else:
        from_epoch = 0
        from_stage = 0
        from_fold = 0
        net = None
        loss = None
    
    k_fold = len(folds) - 1

    for k in range(from_stage, k_fold):
        print(f"Stage {k+1}:")
        if net is None:
            net = nn.DataParallel(MCAN(vocab, config.backbone, config.d_model, config.embedding_dim, config.dff, config.nheads, 
                                        config.nlayers, config.dropout)).cuda()
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)

        tracker = Tracker()
        config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

        max_f1 = 0 # for saving the best model
        f1_test = 0
        for e in range(from_epoch, config.epochs):
            loss = run(net, folds[:-1], from_fold, k, optimizer, tracker, train=True, prefix='Training', epoch=e)
            val_returned = run(net, [folds[-1]], 0, k, optimizer, tracker, train=False, prefix='Validation', epoch=e)
            test_returned = run(net, [test_fold], 0, k, optimizer, tracker, train=False, prefix='Evaluation', epoch=e)

            if loss:
                print(f"Training loss: {loss}")
            print("+"*13)

            results = {
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.state_dict(),
                'eval': {
                    'accuracy': val_returned["accuracy"],
                    "precision": val_returned["precision"],
                    "recall": val_returned["recall"],
                    "f1-val": val_returned["F1"],
                    "f1-test": test_returned["F1"]

                },
                'vocab': train_dataset.vocab,
            }
        
            torch.save(results, os.path.join(config.model_checkpoint, f"model_last_stage_{k+1}.pth"))
            if val_returned["F1"] > max_f1:
                max_f1 = val_returned["F1"]
                f1_test = test_returned["F1"]
                torch.save(results, os.path.join(config.model_checkpoint, f"model_best_stage_{k+1}.pth"))

            from_fold = 0

        from_epoch = 0

        print(f"Finished for stage {k+1}. Best F1 score: {max_f1}. F1 score on test set: {f1_test}")
        print("="*31)

        # change roles of the folds
        tmp = folds[0]
        folds[:-1] = folds[1:]
        folds[-1] = tmp

if __name__ == '__main__':
    main()
