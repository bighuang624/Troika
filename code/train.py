#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None
    
    val_results = []
    
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):

            predict = model(batch, train_pairs)

            loss = model.loss_calu(predict, batch)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        progress_bar.close()
        progress_bar.write(f"epoch {i+1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        val_results.append(val_result)

        if config.val_metric == 'best_loss' and val_result[config.val_metric] < best_loss:
            best_loss = val_result['best_loss']
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))
        if config.val_metric != 'best_loss' and val_result[config.val_metric] > best_metric:
            best_metric = val_result[config.val_metric]
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))

        final_model_state = model.state_dict()
        if i + 1 == config.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "val_best.pt"
            )))
            evaluate(model, test_dataset, config)

    if config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    test_saved_results['loss'] = loss_avg
    return test_saved_results



if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample)

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = get_optimizer(model, config)

    os.makedirs(config.save_path, exist_ok=True)

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
