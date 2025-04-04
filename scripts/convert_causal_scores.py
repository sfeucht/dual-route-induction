import argparse 
import pickle 
import json 
import torch 

from utils import flatidx_to_grididx, json_tuple_keys
from causal_scores import ChunkOutputSaver

def flat_to_dict(flattensor):
    d = {}
    for idx in range(len(flattensor)):
        d[flatidx_to_grididx(idx)] = flattensor[idx].item()
    return d 

def flat_to_ranking(flattensor):
    _, idxs = torch.topk(flattensor, k=len(flattensor))
    return [flatidx_to_grididx(i) for i in idxs]

def main(args):
    if args.ckpt is not None: 
        model_folder = f'{args.model_name}/{args.ckpt}'
        fname = 'len30_n256' 
    else:
        model_folder = args.model_name
        fname = 'len30_n1024'

    with open(f'../cache/causal_scores/{model_folder}/{fname}.pkl', 'rb') as f:
        clean, corrupt, patched = pickle.load(f)
    conceptcopying = patched.get_m1() - corrupt.get_m1()

    with open(f'../cache/causal_scores/{model_folder}/{fname}_randoments.pkl', 'rb') as f:
        cleanr, corrr, patchedr = pickle.load(f)
    tokencopying = patchedr.get_m2() - corrr.get_m2()

    if args.head_orderings:
        concept_rankings = flat_to_ranking(conceptcopying)
        token_rankings = flat_to_ranking(tokencopying)

        with open(f'../cache/head_orderings/{model_folder}/concept_copying.json', 'w') as f:
            json.dump(concept_rankings, f)
        
        with open(f'../cache/head_orderings/{model_folder}/token_copying.json', 'w') as f:
            json.dump(token_rankings, f)

    else:
        # convert to dictionaries with (layer, head_idx) tuples
        concept_scores = flat_to_dict(conceptcopying)
        token_scores = flat_to_dict(tokencopying)

        with open(f'../cache/causal_scores/{model_folder}/concept_copying_{fname}.json', 'w') as f:
            json.dump(json_tuple_keys(concept_scores), f)
        
        with open(f'../cache/causal_scores/{model_folder}/token_copying_{fname}.json', 'w') as f:
            json.dump(json_tuple_keys(token_scores), f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-hf',
                        choices=['Llama-2-7b-hf', 'Meta-Llama-3-8B', 'OLMo-2-1124-7B', 'pythia-6.9b'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--head_orderings', action='store_true')
    parser.set_defaults(head_orderings=False)
    args = parser.parse_args()
    main(args)