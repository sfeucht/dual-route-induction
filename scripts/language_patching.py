''' 
Use the same approach and data as Dumas et al. (2025) ("Separating Tongue From Thought"), except we patch the outputs of concept heads to change the concept:
    source: "Buch -> книга, Wolke -> облако, Tasche -> сумка, Boden -> земля, Tuch -> материя, Herz -> сердце, Hand -> рука, Sonne -> Солнце, Stern -> звезда, Holz ->"
    base: "rareté -> scarcity, réchauffe -> warms, éroder -> eroding, rôti -> roasted, disait -> said, regarde -> watch, nécessitant -> requiring, chérie -> honey, rang -> rank, fournitures ->"
    generation: ['<s> rareté -> scarcity, réchauffe -> warms, éroder -> eroding, rôti -> roasted, disait -> said, regarde -> watch, nécessitant -> requiring, chérie -> honey, rang -> rank, fournitures ->], ['wood, ruisseau ->']

Or alternatively the outputs of FV heads to change the language:
    source: "rareté -> scarcity, réchauffe -> warms, éroder -> eroding, rôti -> roasted, disait -> said, regarde -> watch, nécessitant -> requiring, chérie -> honey, rang -> rank, fournitures ->"
    base: "Buch -> книга, Wolke -> облако, Tasche -> сумка, Boden -> земля, Tuch -> материя, Herz -> сердце, Hand -> рука, Sonne -> Солнце, Stern -> звезда, Holz ->"
    generation: ['</s></s></s></s><s> Buch -> книга, Wolke -> облако, Tasche -> сумка, Boden -> земля, Tuch -> материя, Herz -> сердце, Hand -> рука, Sonne -> Солнце, Stern -> звезда, Holz -> wood, wood, wood,']
'''
import os 
import ast 
import pickle 
import argparse 
import nnsight 
import random 
import numpy as np 
import pandas as pd 
import json 
import torch 
from torch.utils.data import Dataset
from nnsight import LanguageModel

# dataset class for clement data 
class TranslationPairDataset(Dataset):
    def __init__(self, source_from, source_to, base_from, base_to, tokenizer, n_pairs=10, max_examples=256, output_fv_answer=False):
        self.source_from = source_from  # strings representing which language, e.g. 'es'
        self.source_to = source_to
        self.base_from = base_from
        self.base_to = base_to
        self.tok = tokenizer
        self.n_pairs = n_pairs
        self.max_examples = max_examples
        self.output_fv_answer = output_fv_answer

        # read in their dataset 
        with open(f'../data/dumasetal_2025/{self.source_from}/{self.source_to}_word_translation2_prompts.json', 'r') as f:
            self.source_prompts = json.load(f)
        
        with open(f'../data/dumasetal_2025/{self.base_from}/{self.base_to}_word_translation2_prompts.json', 'r') as f:
            self.base_prompts = json.load(f)
        lst = list(self.base_prompts.items())
        # random.shuffle(lst)
        random.Random(4).shuffle(lst) # reproducibility for notebooks 
        self.base_prompts = dict(lst)

        # read in the lookup table 
        self.source_original = pd.read_csv(f'../data/dumasetal_2025/{self.source_from}/word_translation2.csv')
        self.base_original = pd.read_csv(f'../data/dumasetal_2025/{self.base_from}/word_translation2.csv')

    # tokenize with or without preceding spaces, with or without bos.
    def ttok(self, s, space=False, bos=False):
        pfix = ' ' if space else ''
        s = pfix + s 
        if 'llama' in self.tok.name_or_path:
            if not bos: 
                out = self.tok(s)['input_ids'][1:]
            else:
                out = self.tok(s)['input_ids']
        elif 'OLMo' in self.tok.name_or_path or 'pythia' in self.tok.name_or_path:
            if not bos:
                out = self.tok(s)['input_ids']
            else:
                out = [self.tok.bos_token_id] + self.tok(s)['input_ids']
        
        if self.tok.name_or_path == 'meta-llama/Llama-2-7b-hf' and space:
            return out[1:]
        else:
            return out 

    def __len__(self):
        return min(self.max_examples, len(self.source_prompts), len(self.base_prompts))

    def __getitem__(self, index):
        _, source_info = list(self.source_prompts.items())[index]
        _, base_info = list(self.base_prompts.items())[index]

        # tokenize them together 
        source_tokens = self.tok(source_info['prompt'])['input_ids']
        base_tokens = self.tok(base_info['prompt'])['input_ids']

        # get label for desired output and the default unmodified output 
        # if source de -> ru; base fr -> es, you want to see the source word in Spanish. untouched will be base word in Spanish
        desired_answer = self.source_original.loc[self.source_original['word_original']==source_info['word original']][self.base_to].iloc[0]
        original_answer = self.base_original.loc[self.base_original['word_original']==base_info['word original']][self.base_to].iloc[0]

        desired_answers = ast.literal_eval(desired_answer)
        original_answers = ast.literal_eval(original_answer)

        if self.output_fv_answer:
            fv_answer = self.base_original.loc[self.base_original['word_original']==base_info['word original']][self.source_to].iloc[0] 
            fv_answers = ast.literal_eval(fv_answer)
            return source_tokens, base_tokens, desired_answers, original_answers, fv_answers 
        else:
            return source_tokens, base_tokens, desired_answers, original_answers


# get raw head activations for a single head on tokens 
# only bsz 1 
def raw_head_activations(head, tokens, model):
    assert type(tokens[0]) == int
    bsz = 1

    layer, head_idx = head 
    n_heads = 32; head_dim = 128
    if model.config._name_or_path == 'EleutherAI/pythia-6.9b':
        o_proj = model.gpt_neox.layers[layer].attention.dense
    else:
        o_proj = model.model.layers[layer].self_attn.o_proj

    with torch.no_grad(), model.trace(tokens):
        o_proj_in = o_proj.input
        attn_out = o_proj_in.reshape(bsz, -1, n_heads, head_dim) #(bsz, seq_len, n_heads, head_dim)
        head_acts = attn_out[:, :, head_idx, :].detach().save()
    
    return head_acts

# get the activations of a bunch of heads 
def source_head_activations(model, source_sequences, heads_to_grab):
    all_acts = {}
    for head in heads_to_grab:
        all_acts[tuple(head)] = raw_head_activations(head, source_sequences, model)
    return all_acts # a dict of (bsz, seq_len, head_dim) guys 

# head_means was shape [32, 32, 128]
# other_run_dict should be {(l, h) : [bsz, seq_len, 128]}
def subbed_generation(model, this_sequences, heads_to_sub, source_run_dict, max_toks=10):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads 

    layers_in_order = sorted(list(set([layer for layer, _ in heads_to_sub])))

    with torch.no_grad():
        with model.generate(this_sequences, max_new_tokens=max_toks):
            generated = nnsight.list().save()
            model.all() # with model.all()
            for curr_layer in layers_in_order:
                if 'pythia' in model.config._name_or_path:
                    o_proj =  model.gpt_neox.layers[curr_layer].attention.dense
                else:
                    o_proj = model.model.layers[curr_layer].self_attn.o_proj

                # [bsz, seq_len, model_dim]
                o_proj_inp = o_proj.inputs[0][0]
                
                # get activations for the last token [model_dim], and then 
                # reshape into heads [bsz, seq_len, model_dim] -> [bsz, seq_len, n_heads, head_dim=128]
                bsz = o_proj_inp.shape[0]; seq_len = o_proj_inp.shape[1]
                head_acts = o_proj_inp.view(bsz, seq_len, n_heads, head_dim)
                
                curr_heads = [head for layer, head in heads_to_sub if layer == curr_layer]
                for h in curr_heads:
                    head_acts[:, -1, h, :] = source_run_dict[(curr_layer, h)][:, -1, :].cuda()
            
                # replace the output of self_attn.q_proj with modified vector
                new_guy = ((head_acts.reshape(bsz, seq_len, model.config.hidden_size),),{})
                o_proj.inputs = new_guy

            generated.append(model.output.logits[:, -1])
            out = model.generator.output.save()

        generated = torch.stack(generated).argmax(dim=-1).squeeze()
        return out.detach().cpu(), generated

# really coarse: did the model predict a string that's in the answer list or no? 
def generation_correct(generated_str, possible_answers, prnt=False):
    if prnt:
        print('generated', repr(generated_str), 'possible_answers', possible_answers)
    if type(possible_answers) == str:
        return (generated_str in possible_answers) or (possible_answers in generated_str)
    elif type(possible_answers) == list: 
        for ans in possible_answers:
            if (generated_str in ans) or (ans in generated_str): 
                return 1 
        return 0 

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = LanguageModel(args.model, device_map='cuda', cache_dir='/share/u/models')
    model_name = args.model.split('/')[-1]

    with open(f'../cache/head_orderings/{model_name}/{args.head_ordering}.json', 'r') as f:
        heads_to_patch = json.load(f)[:args.k]

    dataset = TranslationPairDataset(args.source_from, args.source_to, args.base_from, args.base_to, model.tokenizer, max_examples=args.max_n, output_fv_answer=True)

    LEN = 10
    base_desired_corr = 0 
    base_default_corr = 0 
    base_fv_corr = 0

    patched_desired_corr = 0 
    patched_default_corr = 0 
    patched_fv_corr = 0

    ct = 0 
    for source_tokens, base_tokens, desired_answer, original_answer, fv_answer in dataset:
        print('source', model.tokenizer.decode(source_tokens))
        print('base', model.tokenizer.decode(base_tokens))
        print('desired', desired_answer)
        print('default', original_answer)

        # get the baseline, what is the original base output?
        # model, this_sequences, heads_to_sub, source_run_dict, max_toks=10, mask=None, sample=False):
        _, base_generation = subbed_generation(model, base_tokens, [], None, max_toks=LEN)
        base_desired_corr += generation_correct(model.tokenizer.decode(base_generation), desired_answer, prnt=False)
        base_default_corr += generation_correct(model.tokenizer.decode(base_generation), original_answer, prnt=False)
        base_fv_corr += generation_correct(model.tokenizer.decode(base_generation), fv_answer, prnt=False)

        # first we want to get the outputs of the heads to patch at end of the source prompt 
        source_run_dict = source_head_activations(model, source_tokens, heads_to_patch)
        _, generation = subbed_generation(model, base_tokens, heads_to_patch, source_run_dict, max_toks=LEN)
        patched_desired_corr += generation_correct(model.tokenizer.decode(generation), desired_answer, prnt=(args.head_ordering == 'cf_nextnext'))
        patched_default_corr += generation_correct(model.tokenizer.decode(generation), original_answer, prnt=False)
        patched_fv_corr += generation_correct(model.tokenizer.decode(generation), fv_answer, prnt=(args.head_ordering == 'fv'))

        ct += 1
        if ct >= args.max_n:
            break 
    
    details = f'{args.source_from}-{args.source_to}_{args.base_from}-{args.base_to}_{args.head_ordering}{args.k}_n{ct}'
    os.makedirs(f'../cache/language_patching/{model_name}', exist_ok=True)

    results = {
        'base_desired_acc' : base_desired_corr / ct, 
        'base_default_acc' : base_default_corr / ct, 
        'base_fv_acc' : base_fv_corr / ct, 
        'patched_desired_acc' : patched_desired_corr / ct, # source word, base language
        'patched_default_acc' : patched_default_corr / ct, # base word, base language
        'patched_fv_acc' : patched_fv_corr / ct, # base word, source language
        'ct' : ct
    }
    with open(f'../cache/language_patching/{model_name}/{details}.json', 'w') as f:
        json.dump(results, f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--max_n', default=128, type=int)
    parser.add_argument('--k', default=80, type=int)
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                        choices=[
                            'meta-llama/Llama-2-7b-hf', 
                            'meta-llama/Meta-Llama-3-8B',
                            'allenai/OLMo-2-1124-7B',
                            'EleutherAI/pythia-6.9b', 
                        ])
    parser.add_argument('--source_from', default='fr')
    parser.add_argument('--source_to', default='en')
    parser.add_argument('--base_from', default='de')
    parser.add_argument('--base_to', default='ru')
    parser.add_argument('--head_ordering', default='concept_copying', 
                        choices=[
                            'concept_copying',
                            'token_copying',
                            'fv'
                        ])
    args = parser.parse_args()

    main(args)