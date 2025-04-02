import random 
import torch 

def pile_chunk(random_len, pile, tok, shuf_pile=True):
    sample = []
    while len(sample) < random_len:
        doc = pile.shuffle()[0]['text'] # sample from huggingface
        sample = tok(doc, bos=False)[: random_len]
        if shuf_pile:
            random.shuffle(sample)
    return sample 

def flatidx_to_grididx(flat_idx):
    if type(flat_idx) == torch.Tensor:
        flat_idx = flat_idx.item()
    layer, head = divmod(flat_idx, 32)
    return (layer, head)

def grididx_to_flatidx(grid_idx):
    layer, head = grid_idx
    return layer * 32 + head 

# function to load in the cached head means for ablation purposes
def get_mean_head_values(model_name):
    dir = f'../activations/{model_name}_pile-10k/'
    return torch.load(dir + 'mean.ckpt')