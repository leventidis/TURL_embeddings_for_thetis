from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import trange, tqdm
from tqdm import tqdm

from data_loader.hybrid_data_loaders import *
from data_loader.header_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
from data_loader.RE_data_loaders import *
from data_loader.EL_data_loaders import *
from model.configuration import TableConfig

from model.model import HybridTableMaskedLM, HybridTableCER, TableHeaderRanking, HybridTableCT,HybridTableEL,HybridTableRE,BertRE
from model.transformers.configuration_bert import BertConfig
from model.transformers.tokenization_bert import BertTokenizer
from model.transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision,ndcg_at_k
from baselines.cell_filling.cell_filling import *
from model import metric

import glob
import nltk
import string
from collections import Counter
from random import sample

def CF_build_input1(pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, entity_cand, config):
    '''
    This is an example of converting an arbitrary table to input
    Here we show an example for cell filling task
    The input entites are entities in the subject column, we append [ENT_MASK] and use its representation to match with the candidate entities    
    '''
    #     print(pgEnt) 
    #     print(pgTitle)
    #     print(secTitle)
    #     print(caption)
    #     print(headers)
    #     print(core_entities)
    #     print(core_entities_text)
    #     print(entity_cand)
    tokenized_pgTitle = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_meta = tokenized_pgTitle+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False) for header in headers]
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0]*tokenized_meta_length
    header_span = []
    for tokenized_header in tokenized_headers:
        tokenized_header_length = len(tokenized_header)
        header_span.append([len(input_tok), len(input_tok)+tokenized_header_length])
        input_tok += tokenized_header
        input_tok_pos += list(range(tokenized_header_length))
        input_tok_type += [1]*tokenized_header_length
    
    input_ent = [config.entity_wikid2id[pgEnt] if pgEnt!=-1 else 0]
    input_ent_text = [tokenized_pgTitle[:config.max_cell_length]]
    input_ent_type = [2]
    
    # core entities in the subject column
    input_ent += [config.entity_wikid2id[entity] for entity in core_entities]
    input_ent_text += [config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False) if len(entity_text)!=0 else [] for entity_text in core_entities_text]
    input_ent_type += [3]*len(core_entities)

    # append [ent_mask]
    input_ent += [config.entity_wikid2id['[ENT_MASK]']]*len(core_entities)
    input_ent_text += [[]]*len(core_entities)
    input_ent_type += [4]*len(core_entities)

    input_ent_cell_length = [len(x) if len(x)!=0 else 1 for x in input_ent_text]
    max_cell_length = max(input_ent_cell_length)
    input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
    for i,x in enumerate(input_ent_text):
        input_ent_text_padded[i, :len(x)] = x
    assert len(input_ent) == 1+2*len(core_entities)

    input_tok_mask = np.ones([1, len(input_tok), len(input_tok)+len(input_ent)], dtype=int)
    for header_i in header_span:
        input_tok_mask[0, header_i[0]:header_i[1], len(input_tok)+1+len(core_entities):] = 0
    input_tok_mask[0, :, len(input_tok)+1+len(core_entities):] = 0

    # build the mask for entities
    input_ent_mask = np.ones([1, len(input_ent), len(input_tok)+len(input_ent)], dtype=int)
    for header_i in header_span[1:]:
        input_ent_mask[0, 1:1+len(core_entities), header_i[0]:header_span[1][1]] = 0
        input_ent_mask[0, 1:1+len(core_entities), len(input_tok)+1+len(core_entities):] = np.eye(len(core_entities), dtype=int)
    input_ent_mask[0, 1+len(core_entities):, header_span[0][0]:header_span[0][1]] = 0
    input_ent_mask[0, 1+len(core_entities):, len(input_tok)+1:len(input_tok)+1+len(core_entities)] = np.eye(len(core_entities), dtype=int)
    input_ent_mask[0, 1+len(core_entities):, len(input_tok)+1+len(core_entities):] = np.eye(len(core_entities), dtype=int)

    input_tok_mask = torch.LongTensor(input_tok_mask)
    input_ent_mask = torch.LongTensor(input_ent_mask)

    input_tok = torch.LongTensor([input_tok])
    input_tok_type = torch.LongTensor([input_tok_type])
    input_tok_pos = torch.LongTensor([input_tok_pos])
    
    input_ent = torch.LongTensor([input_ent])
    input_ent_text = torch.LongTensor([input_ent_text_padded])
    input_ent_cell_length = torch.LongTensor([input_ent_cell_length])
    input_ent_type = torch.LongTensor([input_ent_type])

    input_ent_mask_type = torch.zeros_like(input_ent)
    input_ent_mask_type[:,1+len(core_entities):] = config.entity_wikid2id['[ENT_MASK]']
    
    candidate_entity_set = [config.entity_wikid2id[entity] for entity in entity_cand]
    candidate_entity_set = torch.LongTensor([candidate_entity_set])
    

    return input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
            input_ent, input_ent_text, input_ent_cell_length, input_ent_type, input_ent_mask_type, input_ent_mask, candidate_entity_set

def read_tables(folder):
    tables = {}
    for file in tqdm(glob.glob(folder)):
        table_name = os.path.basename(file)
        try:
            table_content = pd.read_csv(file)
            tables[table_name] = table_content
        except:
            print("Failed reading table:", table_name)
    return tables

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    punc_with_ = [s for s in string.punctuation if s != '_']
    text = re.sub('[%s]' % punc_with_, '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text

def select_k_diverse_row(table, k=10):
    # assuming no null values in the tables
    seed_row = table.sample(n=1, random_state=1)
    table = table.drop(seed_row.index)
    seen_values = seed_row.to_dict('list')
    column_order = [k for k, v in 
                    sorted({col: len(table[col].unique())
                            for col in table.columns}.items(), 
                           key=lambda item: item[1])]
    for _ in range(1, k):
        reduced_table = table.copy()
        for col in column_order:
            temp = reduced_table[~reduced_table[col].isin(seen_values[col])]
            if len(temp) > 0:
                reduced_table = temp
            else:
                column_order.remove(col)
                break
        sampled_row = reduced_table.sample(n=1, random_state=1)
        table = table.drop(sampled_row.index)
        seed_row = pd.concat([seed_row, sampled_row])
        seen_values = seed_row.to_dict('list')
    return seed_row

def get_tables_represenation_data_lake(tables, text_to_entity, device, dataset, model, sampling_size = 10,
                                       sampling_method = 'random', save_only_centroid = True):
    table_representations = {}
    for table_id in tqdm(tables):
        try:
            pgEnt = '[PAD]'
            pgTitle = ''
            secTitle = ''
            caption = ''
            headers = list(tables[table_id].columns)
            table = tables[table_id]
            core_entities = [] # This will be the subject column entities
            core_entities_text = []
            all_entity_cand = []
            if sampling_method == 'head':
                table_subset = table.head(sampling_size) # first 10 rows  
            elif sampling_method == 'random':
                table_subset = table.sample(n=sampling_size, random_state=1) # random 10 rows
            elif sampling_method == 'diverse':
                table_subset = select_k_diverse_row(table, sampling_size) # diversed row sampling with 10 rows
            for index, row in table_subset.iterrows():
                for columnIndex, value in row.items():
                    entity = text_preprocessing(str(value).replace(' ', '_'))
                    if entity in text_to_entity:
                        core_entities.append(text_to_entity[entity])
                        core_entities_text.append(entity) 
                        all_entity_cand.append(text_to_entity[entity])
                    else:
    #                     print('--', entity, 'Not found--')
    #                     print('Looking for sub entities...')
                        sub_entities = entity.split('_')
                        if sub_entities != None:
                            for sub_entity in sub_entities:
                                if sub_entity in text_to_entity:
                                    core_entities.append(text_to_entity[sub_entity])
                                    core_entities_text.append(sub_entity) 
                                    all_entity_cand.append(text_to_entity[sub_entity])
    #                             else:
    #                                 print('\t skipping', sub_entity)
            all_entity_cand = list(set(all_entity_cand))
    #         print(len(core_entities))
            input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                    input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, \
                    candidate_entity_set = CF_build_input1(pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, all_entity_cand, dataset)
            input_tok = input_tok.to(device)
            input_tok_type = input_tok_type.to(device)
            input_tok_pos = input_tok_pos.to(device)
            input_tok_mask = input_tok_mask.to(device)
            input_ent_text = input_ent_text.to(device)
            input_ent_text_length = input_ent_text_length.to(device)
            input_ent = input_ent.to(device)
            input_ent_type = input_ent_type.to(device)
            input_ent_mask_type = input_ent_mask_type.to(device)
            input_ent_mask = input_ent_mask.to(device)
            candidate_entity_set = candidate_entity_set.to(device)
            with torch.no_grad():
                tok_outputs, ent_outputs = model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                                input_ent_text, input_ent_text_length, input_ent_mask_type,
                                input_ent, input_ent_type, input_ent_mask, candidate_entity_set)
            if save_only_centroid:
                table_mean_rep = {}
                table_mean_rep['entities_only'] = torch.mean(ent_outputs[1][0], 0, dtype=torch.float)
                table_mean_rep['metadata_only'] = torch.mean(tok_outputs[1][0], 0, dtype=torch.float)
                table_comb = torch.cat((tok_outputs[1][0], ent_outputs[1][0]), dim=0)
                table_mean_rep['metadata_entities'] = torch.mean(table_comb, 0, dtype=torch.float)
                table_representations[table_id] = table_mean_rep
            else:
                table_representations[table_id] = tok_outputs, ent_outputs
        except Exception as er:
            print("Failed table embedding extraction for table:", table_id)
            print(er)
            print()
    return table_representations

def get_text_to_entity_dict(entity_vocab):
    text_to_entity = {}
    for e in entity_vocab:
        wiki_title = text_preprocessing(entity_vocab[e]['wiki_title'])
        wiki_id = entity_vocab[e]['wiki_id']
        text_to_entity[wiki_title] = wiki_id
    return text_to_entity

def main():
    # Load and initialize the Pre-Trained Model
    print("Loading Models...")
    data_dir = 'data/'
    config_name = "configs/table-base-config_v2.json"
    device = torch.device('cpu')
    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
    entity_wikid2id = {entity_vocab[x]['wiki_id']:x for x in entity_vocab}


    config_class, model_class, _ = (TableConfig, HybridTableMaskedLM, BertTokenizer)
    config = config_class.from_pretrained(config_name)
    config.output_attentions = True

    checkpoint = "checkpoint/"
    model = model_class(config, is_simple=True)
    # checkpoint = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'))
    checkpoint = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'), map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dataset = WikiHybridTableDataset(data_dir,entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150,
        src="dev", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0)
    text_to_entity = get_text_to_entity_dict(entity_vocab)
    print("Finished loading Models.\n\n")

    # Read tables
    print("Reading Tables...")
    tables = read_tables(args.input_dir+'*')
    print("Finished reading tables\n\n")

    # Get Table Embeddings
    print("Extrating Embeddings...")
    table_representations = get_tables_represenation_data_lake(
        tables=tables, text_to_entity=text_to_entity,
        device=device, dataset=dataset, model=model,
        sampling_size=args.sample_size, sampling_method='random', save_only_centroid=True)
    print("Finished extrating embeddings\n\n")


    print("Saving Embeddings...")
    with open(args.output_dir+'embeddings.pickle', 'wb') as handle:
        pickle.dump(table_representations, handle)
    print("Finished saving embeddings.")

if __name__ == "__main__":
    
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Return TURL table embeddings given a directory of .csv tables')

    parser.add_argument('--input_dir', required=True,
        help='Path to the input directory containing all .csv files to be processed'
    )

    parser.add_argument('--output_dir', required=True,
        help='Path to the output directory where the embeddings are stored'
    )

    parser.add_argument('--sample_size', default=10, required=False, type=int,
        help='Sample size used')

    # Parse the arguments
    args = parser.parse_args()

    print("\nInput Directory:", args.input_dir)
    print("Output Directory:", args.output_dir)
    print("Sample Size:", args.sample_size)
    print('\n\n')
     
    main()