import ast
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances

device = "cuda" if torch.cuda.is_available() else "cpu"

STOP_WORDS = {
    'di','del','della','dello','degli','delle','dei',
    'a','al','alla','agli','alle','ai',
    'in','nel','nella','negli','nelle','nei',
    'da','dal','dalla','dagli','dalle','dai',
    'su','sul','sulla','sugli','sulle','sui',
    'con','per','tra','fra','e','o','un','una','uno'
}


def load_main_csv(path):
    return pd.read_csv(path)


def load_psyn_xlsx(path):
    return pd.read_excel(path).set_index("NC")


def load_prand_xlsx(prand_path):

    sheets = pd.read_excel(prand_path, sheet_name=None)
    nat_dfs, neut_dfs = [], []
    for name in sorted(sheets.keys()):
        df = sheets[name].set_index("NC")
        nat_dfs.append(df[["PRand", "sentence_nat_1_PRand",
                           "sentence_nat_2_PRand", "sentence_nat_3_PRand"]])
        neut_dfs.append(df[["PRand", "sentence_neut_PRand"]])
    return nat_dfs, neut_dfs

def replace_nc(sentence, nc, replacement):
    return sentence.replace(nc, replacement, 1)


def extract_content_words(nc):
    return [t for t in nc.strip().split() if t.lower() not in STOP_WORDS]


def parse_bool_tags(tag_str):
    return ast.literal_eval(str(tag_str))

def calculating_similarity(emb1, emb2):
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.unsqueeze(0)
        emb2 = emb2.unsqueeze(0)
    else:
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
    return float(1 - paired_cosine_distances(emb1, emb2)[0])


def batch_cosine_similarity(embed1_list, embed2_list):
    e1 = torch.stack(embed1_list).numpy()
    e2 = torch.stack(embed2_list).numpy()
    return (1 - paired_cosine_distances(e1, e2)).tolist()

def get_embeddings_bert(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return [emb.squeeze(0) for emb in outputs.hidden_states]


def get_word_ids_bert(text, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return encoding.word_ids(batch_index=0)


def get_sent_embedding_bert(layers, n_layers=4):
    last_n = torch.stack(layers[-n_layers:]).mean(dim=0)
    return last_n[1:-1].mean(dim=0).cpu()


def get_nc_embedding_bert(layers, word_ids, bool_tags, n_layers=4):
    last_n = torch.stack(layers[-n_layers:]).mean(dim=0)
    word_to_subtokens = {}
    for si, wi in enumerate(word_ids):
        if wi is None:
            continue
        word_to_subtokens.setdefault(wi, []).append(si)
    nc_indices = []
    for word_idx, is_nc in enumerate(bool_tags):
        if is_nc and word_idx in word_to_subtokens:
            nc_indices.extend(word_to_subtokens[word_idx])
    if nc_indices:
        return last_n[nc_indices].mean(dim=0).cpu()
    return last_n[1:-1].mean(dim=0).cpu()


def get_target_embedding_bert(layers, word_ids, target, sentence, n_layers=4):
    last_n = torch.stack(layers[-n_layers:]).mean(dim=0)
    word_tokens  = sentence.split()
    target_toks  = target.lower().split()
    target_len   = len(target_toks)
    word_indices = None
    for i in range(len(word_tokens) - target_len + 1):
        window = [w.lower().rstrip('.,;:!?)\'"') for w in word_tokens[i:i + target_len]]
        if window == target_toks:
            word_indices = list(range(i, i + target_len))
            break
    if word_indices is not None:
        subtoken_indices = [si for si, wi in enumerate(word_ids) if wi in word_indices]
        if subtoken_indices:
            return last_n[subtoken_indices].mean(dim=0).cpu()
    return last_n[1:-1].mean(dim=0).cpu()

def get_embeddings_minerva(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    layers = [emb.squeeze(0) for emb in outputs.hidden_states]
    return layers, inputs["input_ids"][0]


def _align_words_minerva(tokens):
    word_boundaries = []
    word_idx = -1
    for tok in tokens:
        if tok in ('<s>', '</s>', '<unk>', '<pad>'):
            word_boundaries.append(None)
        elif tok.startswith('▁') or word_idx == -1:
            word_idx += 1
            word_boundaries.append(word_idx)
        else:
            word_boundaries.append(word_idx)
    return word_boundaries


def get_sent_embedding_minerva(layers, n_layers=4):
    last_n = torch.stack(layers[-n_layers:]).mean(dim=0)
    return last_n.mean(dim=0).float().cpu()


def get_nc_embedding_minerva(layers, input_ids, bool_tags, tokenizer, n_layers=4):
    last_n = torch.stack(layers[-n_layers:]).mean(dim=0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_boundaries = _align_words_minerva(tokens)
    word_to_subtokens = {}
    for si, wi in enumerate(word_boundaries):
        if wi is None:
            continue
        word_to_subtokens.setdefault(wi, []).append(si)
    nc_indices = []
    for word_idx, is_nc in enumerate(bool_tags):
        if is_nc and word_idx in word_to_subtokens:
            nc_indices.extend(word_to_subtokens[word_idx])
    if nc_indices:
        return last_n[nc_indices].mean(dim=0).float().cpu()
    return last_n.mean(dim=0).float().cpu()


def get_target_embedding_minerva(layers, input_ids, target, sentence, tokenizer, n_layers=4):
    last_n = torch.stack(layers[-n_layers:]).mean(dim=0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_boundaries = _align_words_minerva(tokens)
    word_to_subtokens = {}
    for si, wi in enumerate(word_boundaries):
        if wi is None:
            continue
        word_to_subtokens.setdefault(wi, []).append(si)
    target_toks = target.lower().split()
    target_len  = len(target_toks)
    word_tokens = sentence.split()
    word_indices = None
    for i in range(len(word_tokens) - target_len + 1):
        window = [w.lower().rstrip('.,;:!?)\'"') for w in word_tokens[i:i + target_len]]
        if window == target_toks:
            word_indices = list(range(i, i + target_len))
            break
    if word_indices is not None:
        subtoken_indices = [si for si, wi in enumerate(word_boundaries) if wi in word_indices]
        if subtoken_indices:
            return last_n[subtoken_indices].mean(dim=0).float().cpu()
    return last_n.mean(dim=0).float().cpu()

def get_sent_embedding_sbert(text, model):
    return model.encode(text)


def get_sent_embedding_fasttext(text, model):
    words = text.strip().split()
    vecs  = [model.get_word_vector(w) for w in words]
    return np.mean(vecs, axis=0)

def select_most_sim(sim1, sim2):
    return sim1 if sim1 >= sim2 else sim2
