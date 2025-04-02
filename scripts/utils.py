import random 
import torch 
import math 

import llama 
import pythia 

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

def get_l2_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
            attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

            # [bsz, seq_len, model_size]
            query_states = model.model.layers[layer].self_attn.q_proj.output
            key_states = model.model.layers[layer].self_attn.k_proj.output 

            bsz = query_states.shape[0]; seq_len = query_states.shape[1] 
            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = model.model.layers[layer].self_attn.v_proj.output
                value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama.apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            # not needed because num_key_value_heads == num_attention_heads 
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            # has to be eager implementation 
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective 

def get_l3_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            # self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
            n_kv_groups = model.model.layers[layer].self_attn.num_key_value_groups

            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
            attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

            # grouped query means that we have more queries than we do keys/values
            query_states = model.model.layers[layer].self_attn.q_proj.output # [bsz, seq_len, model_size=4096]
            key_states = model.model.layers[layer].self_attn.k_proj.output  # [bsz, seq_len, 1024]
            bsz = query_states.shape[0]; seq_len = query_states.shape[1] 

            if value_weighting:
                value_states = model.model.layers[layer].self_attn.v_proj.output # [bsz, seq_len, 1024]
                value_states = value_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2) # [bsz, seq_len, 8, head_dim] -> [bsz, 8, seq_len, head_dim]
                value_states = llama.repeat_kv(value_states, n_kv_groups).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama.apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            # not needed because num_key_value_heads == num_attention_heads 
            key_states = llama.repeat_kv(key_states, n_kv_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            # has to be eager implementation 
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective   

def get_olmo2_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
            attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

            # [bsz, seq_len, model_size]
            query_states = model.model.layers[layer].self_attn.q_norm.output
            key_states = model.model.layers[layer].self_attn.k_norm.output 
            bsz = query_states.shape[0]; seq_len = query_states.shape[1] 

            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = model.model.layers[layer].self_attn.v_proj.output
                value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            bsz = query_states.shape[0]; seq_len = query_states.shape[1]
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama.apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            scaling = model.model.layers[layer].self_attn.scaling # it's just 1/math.sqrt(head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective 

def get_pythia_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            attention_mask = model.gpt_neox.layers[layer].attention.inputs[1]['attention_mask']
            pos = model.gpt_neox.layers[layer].attention.inputs[1]['position_embeddings']

            qkv = model.gpt_neox.layers[layer].attention.query_key_value.output
            bsz = qkv.shape[0]; seq_len = qkv.shape[1]
            qkv = qkv.view((bsz, seq_len, n_heads, 3 * head_dim)).transpose(1, 2).chunk(3, dim=-1)

            query_states = qkv[0]; key_states = qkv[1]; value_states = qkv[2]
            query_states, key_states = pythia.apply_rotary_pos_emb(query_states, key_states, pos[0], pos[1])

            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = value_states.reshape(bsz, seq_len, n_heads, head_dim).save()
            
            scaling = model.gpt_neox.layers[layer].attention.scaling # it's just 1/math.sqrt(head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective 