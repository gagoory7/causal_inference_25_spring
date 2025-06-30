import argparse

import numpy as np
import pandas as pd

import torch
from transformers import BertConfig

from omegaconf import OmegaConf

from src.Crawl import start_crwal
from src.Dataset import Dataset
from src.Runner import Causal_KoBert_Runner
from src.model import Causal_Kobert
from src.Utils import (
    set_seed,
    make_dataframe,
    ATT_hypothsis
)

from src.tokenization_kobert import KoBertTokenizer


def main(args) :
    
    # data crawling
    if args.crawl.start :
        start_crwal(args.crawl.media_company, args.crawl.date, args.crawl.crawl_duration , args.crawl.workers)

    # seed setting 
    set_seed()

    # dataframe setting
    df, num_labels, confounder_col_for_train = make_dataframe(args.confounder, args.thresholding)

    # model name    
    model_config = "monologg/kobert"
    tokenizer = KoBertTokenizer.from_pretrained(model_config)

    # dataset setting
    dataset = Dataset(df, text='text', confounder=confounder_col_for_train, treatment='gender', outcome='outcome', tokenizer=tokenizer, batch_size=16)
    data_loader = dataset.get_dataloaders()

    # config file setting
    config = BertConfig.from_pretrained(model_config)
    config.num_labels = num_labels
    config.output_attentions=False  
    config.output_hidden_states=False  
    config.MASK_IDX = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # device setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model & runner setting 
    casual_model = Causal_Kobert.from_pretrained(model_config, config=config)
    casual_model.to(device)
    causal_run = Causal_KoBert_Runner(casual_model, args.loss_weight)

    # same batch order
    fixed_batches = list(data_loader)

    for i in range(1,args.epochs+1):
        print(f'epoch : {i}')
        causal_run.train(fixed_batches, learning_rate =  args.lr)

    results, DR = causal_run.inference(fixed_batches)

    ATT_hypothsis(DR, 0.05)

    if args.save_result : 
        results.to_csv('result/results.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser')
    
    arg = parser.add_argument

    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    arg('--config', '-c', '--c', type=str, 
        help='Configuration file setting.', required=True)

    args = parser.parse_args()

    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    print(OmegaConf.to_yaml(config_yaml))

    main(config_yaml)