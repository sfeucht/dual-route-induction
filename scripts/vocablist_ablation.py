'''
Take in some ranking of attention heads and mean-ablate several top-k values.

See how well the model can still do for these "list" tasks:
    - Translation (fr -> en, ...)
    - Copying (en -> en, ...)
    - Random copying
    - Synonyms
    - Antonyms 

Saves results as a json in `../cache/vocablist_ablation/[model_name]`
'''
import os 
import torch 
import json
import random 
import argparse 
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from nnsight import LanguageModel
from datasets import load_dataset
from utils import get_mean_head_values

VALID_MODELS = [
    'meta-llama/Llama-2-7b-hf', 
    'meta-llama/Meta-Llama-3-8B',
    'EleutherAI/pythia-6.9b',
    'allenai/OLMo-2-1124-7B'
]
TOPKS = [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 128, 256, 512, 1024] 

class VocabListDataset(Dataset):
    def __init__(self, language, tokenizer, word_len, seq_len, max_n, multitok_answers=False):
        self.language = language 
        self.tok = tokenizer
        self.word_len = word_len
        self.seq_len = seq_len
        self.max_n = max_n
        self.multitok_answers = multitok_answers
        self.newline = self.ttok('\n')[-1]
        
        # load in the synonym data 
        assert self.language in ['fr', 'de', 'es', 'it', 'pt', 'en', 'synonym', 'antonym', 'title', 'CAPS']

        if self.language in ['fr', 'de', 'es', 'it', 'pt']:
            sep = '\t' if self.language == 'pt' else ' '
            self.word_pairs = pd.read_csv(f'../data/conneauetal_2017/en-{self.language}.txt', sep=sep, names=['en', 'fr'], encoding='utf-8')
            self.word_pairs = self.word_pairs.loc[self.word_pairs['en'] != self.word_pairs['fr']]
            self.word_pairs['from'] = self.word_pairs['fr']
            self.word_pairs['to'] = self.word_pairs['en']

        elif self.language == 'en': # copying 
            # nguyen words are probably better, but we used conneau words here. 
            self.word_pairs = pd.read_csv(f'../data/conneauetal_2017/all_synonyms.csv')
            self.word_pairs['from'] = self.word_pairs['word1']
            self.word_pairs['to'] = self.word_pairs['word1']
        
        elif self.language in ['synonym', 'antonym']:
            self.word_pairs = pd.read_csv(f'../data/nguyenetall_2017/all_{self.language}s.csv')
            self.word_pairs['from'] = self.word_pairs['word1']
            self.word_pairs['to'] = self.word_pairs['word2']

        elif self.language == 'title':
            self.word_pairs = pd.read_csv(f'../data/nguyenetal_2017/all_antonyms.csv')
            self.word_pairs = self.word_pairs.dropna()
            self.word_pairs['from'] = self.word_pairs['word1'].apply(lambda s: s.lower())
            self.word_pairs['to'] = self.word_pairs['word1'].apply(lambda s: s.title())
        
        elif self.language == 'CAPS':
            self.word_pairs = pd.read_csv(f'../data/nguyenetal_2017/all_antonyms.csv')
            self.word_pairs = self.word_pairs.dropna()
            self.word_pairs['from'] = self.word_pairs['word1'].apply(lambda s: s.lower())
            self.word_pairs['to'] = self.word_pairs['word1'].apply(lambda s: s.upper())

        # drop rows with nans 
        self.word_pairs = self.word_pairs.dropna()
        
        # filter the pairs 
        new = []
        for _, row in self.word_pairs.iterrows():
            tok1 = self.ttok(row['from'], space=True)
            tok2 = self.ttok(row['to'], space=True)
            if self.word_len > 0 and (len(tok1) == self.word_len and len(tok2) == self.word_len):
                    if self.language == 'en':
                        new.append(row)
                    else: # check that first tok isn't the same. 
                        if tok1[0] != tok2[0]:
                            new.append(row)
            
            # if limit is 0, don't check the lengths at all. 
            elif self.word_len == 0: 
                if self.language == 'en':
                    new.append(row)
                else: # check that first tok isn't the same. 
                    if tok1[0] != tok2[0]:
                        new.append(row)

            # if limit is -1, make sure both of them are multi-token
            elif self.word_len == -1 and (len(tok1) > 1 and len(tok2) > 1): 
                if self.language == 'en':
                    new.append(row)
                else: # check that first tok isn't the same. 
                    if tok1[0] != tok2[0]:
                        new.append(row)
            

        self.word_pairs = pd.DataFrame(new, columns=self.word_pairs.columns)
        self.word_pairs.reset_index(inplace=True, drop=True)

        if len(self.word_pairs) > self.max_n:
            self.word_pairs = self.word_pairs.sample(n=self.max_n)
            self.word_pairs.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.word_pairs) 

    def ttok(self, s, space=False, bos=False):
        # otherwise get actual tokens 
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

    def __getitem__(self, index):
        # index indicates which one will be at the end 
        other_examples = self.word_pairs.drop(index).sample(n=(self.seq_len - 1))
        last_example = self.word_pairs.iloc[index]
        
        if self.language in ['fr', 'de', 'it', 'pt', 'es']:
            full = {
                'fr' : 'French',
                'de' : 'German',
                'it' : 'Italian',
                'pt' : 'Portuguese',
                'es' : 'Spanish'
            }[self.language]
            first_half = self.tok(f'{full} vocab:\n')['input_ids'] # include bos
            second_half = self.ttok('English translation:\n')

        elif self.language == 'en':
            first_half = self.tok('English vocab:\n')['input_ids'] # include bos
            second_half = self.ttok('English vocab:\n')
        
        elif self.language == 'synonym':
            first_half = self.tok('Words:\n')['input_ids'] # include bos
            second_half = self.ttok('Synonyms:\n')
        
        elif self.language == 'antonym':
            first_half = self.tok('Words:\n')['input_ids'] # include bos
            second_half = self.ttok('Antonyms:\n')
        
        elif self.language in ['title', 'CAPS']:
            first_half = self.tok('Lowercase Words:\n')['input_ids'] # include bos
            second_half = self.ttok('Capitalized Words:\n')

        other_examples.reset_index(inplace=True, drop=True)
        for i, row in other_examples.iterrows():
            first_half += self.ttok(f'{i+1}.') + self.ttok(row['from'], space=True) + [self.newline]
            second_half += self.ttok(f'{i+1}.') + self.ttok(row['to'], space=True) + [self.newline]
        
        first_half += self.ttok(f'{len(other_examples)+1}.') + self.ttok(last_example['from'], space=True) 
        second_half += self.ttok(f'{len(other_examples)+1}.') 
       
        full = first_half + [self.newline] + second_half
        if self.multitok_answers:
            return full, self.ttok(last_example['to'], space=True) 
        else: 
            answer = self.ttok(last_example['to'], space=True)[0]
            return full, answer

class NonsenseListDataset(Dataset):
    def __init__(self, tokenizer, word_len, seq_len, max_n, uniform=False):
        self.tok = tokenizer
        self.word_len = word_len
        self.seq_len = seq_len
        self.max_n = max_n
        self.newline = self.ttok('\n')[-1]
        self.uniform = uniform

        self.words = []
        if self.uniform: 
            for _ in range(self.max_n):
                self.words.append(
                    [random.randint(0, self.tok.vocab_size) for _ in range(self.word_len)]
                )
        else: 
            print('loading pile for nonsense set...')
            pile = load_dataset('NeelNanda/pile-10k')['train']
            for _ in range(self.max_n):
                doc = pile.shuffle()[0]['text'] # sample huggingface document
                toks = [t for t in self.ttok(doc) if t != 13] # get rid of newlines 
                random.shuffle(toks)
                left = random.randint(0, len(toks) - self.word_len)
                self.words.append(toks[left : left + self.word_len])

    def __len__(self):
        return len(self.words)

    def ttok(self, s, bos=False):
        # otherwise get actual tokens 
        if 'llama' in self.tok.name_or_path:
            if not bos: 
                return self.tok(s)['input_ids'][1:]
            else:
                return self.tok(s)['input_ids']
        elif self.tok.name_or_path in ['allenai/OLMo-2-1124-7B', 'EleutherAI/pythia-6.9b']:
            if not bos:
                return self.tok(s)['input_ids']
            else:
                return [self.tok.bos_token_id] + self.tok(s)['input_ids']

    def __getitem__(self, index):
        # index indicates which one will be at the end 
        other_examples = random.sample(self.words, self.seq_len - 1)
        last_example = self.words[index]

        first_half = self.tok('Vocab:\n')['input_ids'] # include bos
        second_half = self.ttok('Vocab:\n')

        for i, word in enumerate(other_examples):
            first_half += self.ttok(f'{i+1}.') + word + [self.newline]
            second_half += self.ttok(f'{i+1}.') + word + [self.newline]
        
        first_half += self.ttok(f'{len(other_examples)+1}.') + last_example
        second_half += self.ttok(f'{len(other_examples)+1}.')
       
        full = first_half + [self.newline] + second_half
        answer = last_example[0]
        return full, answer

# gets ablated predictions for a given batch 
def get_ablated_preds(model, sequences, heads_to_ablate, head_means):
    assert model.config._name_or_path in VALID_MODELS
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads 

    layers_in_order = sorted(list(set([layer for layer, _ in heads_to_ablate])))
    
    with torch.no_grad():
        with model.trace(sequences):
            for curr_layer in layers_in_order:
                if 'pythia' in model.config._name_or_path:
                    o_proj = model.gpt_neox.layers[curr_layer].attention.dense
                else: 
                    o_proj = model.model.layers[curr_layer].self_attn.o_proj

                # [bsz, seq_len, model_dim]
                o_proj_inp = o_proj.inputs[0][0]
                
                # get activations for the last token [model_dim], and then 
                # reshape into heads [bsz, seq_len, model_dim] -> [bsz, seq_len, n_heads, head_dim=128]
                bsz = o_proj_inp.shape[0]; seq_len = o_proj_inp.shape[1]
                head_acts = o_proj_inp.view(bsz, seq_len, n_heads, head_dim)
                
                curr_heads = [head for layer, head in heads_to_ablate if layer == curr_layer]
                for h in curr_heads:
                    the_mean = head_means[curr_layer, h]
                    head_acts[:, :, h, :] = the_mean.cuda()
            
                # replace the output of self_attn.q_proj with modified vector
                new_guy = ((head_acts.reshape(bsz, seq_len, model.config.hidden_size),),{})
                o_proj.inputs = new_guy

            logits = model.output.logits.save()
    
        probs = logits[:, -1, :].softmax(dim=-1) # [bsz, seq_len, 32k] 
        preds = logits[:, -1, :].argmax(dim=-1) 
        return preds.detach().cpu(), probs.detach().cpu()  # [bsz], [bsz, 32k]

def topk_acc(probs, answers, k):
    # probs shape [bsz, vocab_size]
    # answers shape [bsz]
    n_corr = 0 
    _, idxs = torch.topk(probs, k=k)
    for i in range(len(answers)):
        if answers[i] in idxs[i]:
            n_corr += 1
    return n_corr

# returns dictionary of k_values and activations for each k value 
def evaluate_dataset(model, loader, heads_to_ablate, head_means):
    total = 0 
    dset_corr = {k : 0 for k in TOPKS}
    dset_top5 = {k: 0 for k in TOPKS}
    dset_top10 = {k: 0 for k in TOPKS}
    dset_probs = {k: 0 for k in TOPKS}

    for sequences, answers in tqdm(loader):
        answers = torch.tensor(answers)
        total += len(answers)
        if type(sequences[0]) == list:
            print(model.tokenizer.decode(sequences[0]), '[ans: ' + model.tokenizer.decode(answers[0]) + ']')
        else:
            print(sequences[0], '[ans:' + model.tokenizer.decode(answers[0]) +']')
        for k in TOPKS: 
            preds, probs = get_ablated_preds(model, sequences, heads_to_ablate[:k], head_means)

            # print(probs[torch.arange(len(answers)), answers])
            dset_corr[k] += torch.sum(preds == answers).item()
            dset_top5[k] += topk_acc(probs, answers, k=5) 
            dset_top10[k] += topk_acc(probs, answers, k=10) 
            dset_probs[k] += torch.sum(probs[torch.arange(len(answers)), answers]).item()

    avg_cors = {k : n_correct / total for k, n_correct in dset_corr.items()}
    avg_top5 = {k : top5 / total for k, top5 in dset_top5.items()}
    avg_top10 = {k : top10 / total for k, top10 in dset_top10.items()}
    avg_probs = {k : probsum / total for k, probsum in dset_probs.items()}
    return avg_cors, avg_top5, avg_top10, avg_probs, total 

def save_results(accs, top5s, top10s, probs, n, fname):
    print(fname, n, accs, probs)
    results = {
        'accs' : accs,
        'top5_accs' : top5s,
        'top10_accs' : top10s,
        'probs' : probs,
        'n' : n
    }
    with open(fname, 'w') as f:
        json.dump(results, f)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = LanguageModel(args.model, device_map='cuda')
    model_name = args.model.split('/')[-1]

    if args.task in ['fr-en', 'de-en', 'es-en', 'it-en', 'pt-en']:
        dataset = VocabListDataset(args.task[:2], model.tokenizer, args.word_len, args.seq_len, args.max_n)
    elif args.task == 'copy':
        dataset = VocabListDataset('en', model.tokenizer, args.word_len, args.seq_len, args.max_n)
    elif args.task == 'nonsense':
        word_len = args.word_len if args.word_len > 0 else 2
        dataset = NonsenseListDataset(model.tokenizer, word_len, args.seq_len, args.max_n)
    elif args.task == 'random':
        word_len = args.word_len if args.word_len > 0 else 2
        dataset = NonsenseListDataset(model.tokenizer, word_len, args.seq_len, args.max_n, uniform=True)
    else:
        dataset = VocabListDataset(args.task, model.tokenizer, args.word_len, args.seq_len, args.max_n)

    def collate_fn(batch):
        return [seq for seq, _ in batch], [ans for _, ans in batch]
    
    loader = DataLoader(dataset, batch_size=args.bsz, collate_fn=collate_fn)

    # load in the heads we want to ablate. pre-processed from head_patching. 
    with open(f'../cache/head_orderings/{model_name}/{args.head_ordering}.json', 'r') as f: 
        heads_to_ablate = json.load(f)

    # load mean head values for this model 
    head_means = get_mean_head_values(model_name)

    # save information 
    path = f'../cache/vocablist_ablation/{model_name}/{args.task}/'
    fname = f'wordlen{args.word_len}_seqlen{args.seq_len}_maxn{args.max_n}_{args.head_ordering}'
    os.makedirs(path, exist_ok=True)

    save_results(*evaluate_dataset(model, loader, heads_to_ablate, head_means), path + fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--bsz', default=16, type=int)
    parser.add_argument('--max_n', default=1024, type=int, 
                        help='maximum number of examples (there may not be this many if you restrict word length)')
    parser.add_argument('--word_len', default=0, type=int,
                        help='number of tokens in each word. 0=no filtering, -1=only multi-token words.') 
    parser.add_argument('--seq_len', default=10, type=int,
                        help='number of examples in the first repetition of the vocabulary list.') 
    parser.add_argument('--task', default='fr-en', 
                        choices=['fr-en', 'de-en', 'es-en', 'it-en', 'pt-en', 'copy', 'nonsense', 'random', 'synonym', 'antonym', 'title', 'CAPS'])
    parser.add_argument('--head_ordering', default='concept_copying', 
                        choices=[
                            'concept_copying',
                            'token_copying',
                            'fv'
                        ])
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                        choices=VALID_MODELS)
    args = parser.parse_args()
    main(args)