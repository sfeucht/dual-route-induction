'''
Calculate chunk attention scores for heads in a given model. 
*Value weighting is on by default.* 

Build sequences out of random pile tokens that look like this:
    foo bar quack emmy plink doer Ed.mont.on \n foo bar quack emmy plink doer
(make sure it's not always be exactly at the end.)
(make sure to balance the CF, spacy, or mtw entity lengths.)

Then for each attention head using value-weighted attention, calculate:
    - Token attention score: how much attn goes towards `Ed`
    - Chunk attention score: how much attn goes towards `on` 

Also have an option --random_tok_entities to replace with random tokens... 
    Token attn score -> indicates token induction heads 
    Chunk attn score -> indicates heads that like to copy far forward 

Then I'll be able to suss out the 
    (1) pure token induction heads: high P(`Ed`) and P(random_next)
    (2) mixed concept heads: high P(`on`) and P(random_next)
    (3) pure concept heads: high P(`on`) and low for everything else 
    (4) skip-ahead induction heads: high P(`on`) and high P(random_far)
'''
import json
import os 
import argparse 
import torch 
import random 
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from collections import defaultdict

from nnsight import LanguageModel
from datasets import load_dataset

from utils import pile_chunk, get_l2_attn_weights, get_l3_attn_weights, get_olmo2_attn_weights, get_pythia_attn_weights, json_tuple_keys

torch.set_grad_enabled(False)

def generate_ragged_batch(batch_ents, pile, tok, seq_len):
    assert len({len(e) for e in batch_ents}) == 1

    newline = tok('\n', bos=False)[-1]
    bos = tok('', bos=True)[0]

    sequences = []
    start_idxs, end_idxs = [], []
    for ent in batch_ents:
        position = random.choice(range(seq_len // 2, seq_len - len(ent) + 1))
        rand1 = pile_chunk(position, pile, tok)
        rand2 = pile_chunk(seq_len - position - len(ent), pile, tok)

        start_idxs.append(position + 1)
        end_idxs.append(position + len(ent))
        sequences.append(
            [bos] + rand1 + ent + rand2 + [newline] + rand1 
        )

    # since batches have ragged ends by design, save padding offsets 
    flipped_masks = [m - 1 for m in tok(sequences, pad_mask=True)]
    pad_offsets = [-sum(f).item() for f in flipped_masks]

    return sequences, torch.tensor(start_idxs), torch.tensor(end_idxs), torch.tensor(pad_offsets)

def retrieve_attention(model, tokenized, layer, value_weighting=True):
    func = {
        'meta-llama/Llama-2-7b-hf' : get_l2_attn_weights,
        'meta-llama/Meta-Llama-3-8B' : get_l3_attn_weights,
        'allenai/OLMo-2-1124-7B' : get_olmo2_attn_weights,
        'EleutherAI/pythia-6.9b' : get_pythia_attn_weights
    }[model.config._name_or_path]

    return func(model, tokenized, layer, value_weighting)

def normalize(d, total):
    for k in d.keys():
        d[k] /= total 
    return d 

def main(args):
    random.seed(8)
    torch.manual_seed(8)
    np.random.seed(8)

    if args.ckpt is not None: 
        assert args.model in ['allenai/OLMo-2-1124-7B', 'EleutherAI/pythia-6.9b']
        model = LanguageModel(args.model, device_map='cuda', revision=args.ckpt, cache_dir='/share/u/models')
    else: 
        model = LanguageModel(args.model, device_map='cuda', cache_dir='/share/u/models')
    
    model_name = args.model.split('/')[-1]
    d = model.tokenizer.decode

    assert args.bsz <= args.n // 4 

    def tok(s, bos=False, model=model, pad_mask=False):
        if pad_mask:
            assert type(s) == list and type(s[0]) == list and type(s[0][0]) == int
            return model.tokenizer.pad({'input_ids' : s}, return_tensors='pt')['attention_mask']

        # otherwise get actual tokens 
        if 'llama' in model.config._name_or_path:
            if not bos: 
                return model.tokenizer(s)['input_ids'][1:]
            else:
                return model.tokenizer(s)['input_ids']
        elif model.config._name_or_path in ['allenai/OLMo-2-1124-7B', 'EleutherAI/pythia-6.9b']:
            if not bos:
                return model.tokenizer(s)['input_ids']
            else:
                return [model.tokenizer.bos_token_id] + model.tokenizer(s)['input_ids']

    # load in pile sample to use as basic material that we shuffle around 
    # pile = load_dataset('NeelNanda/pile-10k')['train']
    pile = load_dataset('JeanKaddour/minipile')['test']

    # dummy entities for comparison 
    sorted_entities = defaultdict(list)
    if args.random_tok_entities:
        for i in range(args.n):
            doc_toks = []
            while len(doc_toks) < 5:
                doc = pile.shuffle()[0]['text']
                doc_toks = tok(doc)

            random.shuffle(doc_toks)
            if i % 4 == 0: 
                sorted_entities['bigram'].append(doc_toks[:2])
            elif i % 4 == 1: 
                sorted_entities['trigram'].append(doc_toks[:3])
            elif i % 4 == 2:
                sorted_entities['fourgram'].append(doc_toks[:4])
            elif i % 4 == 3: 
                sorted_entities['fivegram'].append(doc_toks[:5])

    # load and sort entities of different token lengths
    else:
        str_entities = list(pd.read_csv('../data/counterfact_expanded.csv')['subject'])
        for ent in str_entities:
            toks = tok(ent)
            if len(toks) == 2:
                sorted_entities['bigram'].append(toks)
            elif len(toks) == 3: 
                sorted_entities['trigram'].append(toks)
            elif len(toks) == 4: 
                sorted_entities['fourgram'].append(toks)
            elif len(toks) == 5: 
                sorted_entities['fivegram'].append(toks) 
    
    # For each head, save the stuff 
    total_examples = 0 
    next_tok_attn = defaultdict(int)
    end_tok_attn = defaultdict(int)

    # I guess we're doing each batch is the same length entity 
    for l, ents in sorted_entities.items():
        selected_ents = ents[ : args.n // 4]
        n_batches = len(selected_ents) // args.bsz
        print('attention for', l, model.tokenizer.decode(selected_ents[0]))

        for batch_idx in tqdm(range(n_batches)):
            batch_ents = selected_ents[batch_idx * args.bsz : (batch_idx + 1) * args.bsz]
            batch_seqs, start_idxs, end_idxs, pad_offsets = generate_ragged_batch(batch_ents, pile, tok, args.sequence_len)

            print(repr(model.tokenizer.decode(batch_seqs[0])))
            print(start_idxs[0].item(), end_idxs[0].item(), model.tokenizer.decode(batch_seqs[0][start_idxs[0]]), model.tokenizer.decode(batch_seqs[0][end_idxs[0]]))

            # get attention patterns for each head and example 
            for layer in range(model.config.num_hidden_layers):
                # [bsz, n_heads, seq_from, seq_to]
                attns = retrieve_attention(model, batch_seqs, layer)

                # index in and save beginnings, ends 
                for head in range(model.config.num_attention_heads):
                    next_tok_attn[(layer, head)] += attns[torch.arange(len(attns)), head, -1, start_idxs + pad_offsets].sum().item()
                    end_tok_attn[(layer, head)] += attns[torch.arange(len(attns)), head, -1, end_idxs + pad_offsets].sum().item()
            
            total_examples += len(batch_ents)
         
    
    results = {
        'next_tok_attn' : json_tuple_keys(normalize(next_tok_attn, total_examples)),
        'end_tok_attn' : json_tuple_keys(normalize(end_tok_attn, total_examples))
    }

    path = f'../cache/attention_scores/{model_name}/'
    path += f'{args.ckpt}/' if args.ckpt is not None else ''
    os.makedirs(path, exist_ok=True)

    fname = f'n{args.n}_seqlen{args.sequence_len}'
    fname += f'_randomtokents' if args.random_tok_entities else ''
    fname += '.json'
    print(path + fname)

    with open(path + fname, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
        choices=['meta-llama/Llama-2-7b-hf', 
                 'allenai/OLMo-2-1124-7B', 
                 'meta-llama/Meta-Llama-3-8B',
                 'EleutherAI/pythia-6.9b'])
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--n', default=2048, type=int)
    parser.add_argument('--bsz', default=128, type=int, help='may have bugs with bsz=1.')
    parser.add_argument('--sequence_len', default=30)
    parser.add_argument('--random_tok_entities', action='store_true')
    parser.set_defaults(random_tok_entities=False)
    args = parser.parse_args()

    main(args)