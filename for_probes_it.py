import numpy as np
import torch
from tqdm import tqdm

from utils_it import (
    replace_nc, extract_content_words, parse_bool_tags,
    calculating_similarity,
    get_embeddings_bert, get_word_ids_bert,
    get_sent_embedding_bert, get_nc_embedding_bert, get_target_embedding_bert,
    get_embeddings_minerva,
    get_sent_embedding_minerva, get_nc_embedding_minerva, get_target_embedding_minerva,
    get_sent_embedding_sbert,
    get_sent_embedding_fasttext,
  
    select_most_sim,
)

def _get_sent_emb(text, model_name, model, tokenizer, device):

    if model_name == "mSBERT":
        return get_sent_embedding_sbert(text, model)
    elif model_name == "fastText":
        return get_sent_embedding_fasttext(text, model)
    elif model_name == "Minerva":
        layers, _ = get_embeddings_minerva(text, tokenizer, model, device)
        return get_sent_embedding_minerva(layers)
    else: 
        layers = get_embeddings_bert(text, tokenizer, model, device)
        return get_sent_embedding_bert(layers)


def _get_nc_emb(text, bool_tags, model_name, model, tokenizer, device):
    
    if model_name == "mSBERT":
        return get_sent_embedding_sbert(text, model)
    elif model_name == "fastText":
        return get_sent_embedding_fasttext(text, model)
    elif model_name == "Minerva":
        layers, input_ids = get_embeddings_minerva(text, tokenizer, model, device)
        return get_nc_embedding_minerva(layers, input_ids, bool_tags, tokenizer)
    else:
        layers   = get_embeddings_bert(text, tokenizer, model, device)
        word_ids = get_word_ids_bert(text, tokenizer)
        return get_nc_embedding_bert(layers, word_ids, bool_tags)


def _get_target_emb(text, target, bool_tags, model_name, model, tokenizer, device):
    
    if model_name == "mSBERT":
        return get_sent_embedding_sbert(text, model)
    elif model_name == "fastText":
        return get_sent_embedding_fasttext(text, model)
    elif model_name == "Minerva":
        layers, input_ids = get_embeddings_minerva(text, tokenizer, model, device)
        return get_target_embedding_minerva(layers, input_ids, target, text, tokenizer)
    else:
        layers   = get_embeddings_bert(text, tokenizer, model, device)
        word_ids = get_word_ids_bert(text, tokenizer)
        return get_target_embedding_bert(layers, word_ids, target, text)

NO_NC_MODELS = {"mSBERT", "fastText"}

def has_nc(model_name):
    return model_name not in NO_NC_MODELS

def compute_psyn(df_main, df_psyn, model_name, model, tokenizer, device):

    frasi_nat = ["sentence_nat_1", "sentence_nat_2", "sentence_nat_3"]
    tag_nat   = ["tag_sentence_nat_1", "tag_sentence_nat_2", "tag_sentence_nat_3"]
    results = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PSyn"):
        nc = str(row["NC"]).strip()
        neutral      = str(row["sentence_neut"]).strip()
        tag_neutral  = parse_bool_tags(row["tag_sentence_neut"])

        row_out = {"NC": nc}

        if nc not in df_psyn.index:
            for col in frasi_nat:
                label = "sentence" + col.split("_")[-1]
                row_out[f"PSyn_sent_{label}"] = None
                row_out[f"PSyn_nc_{label}"]   = None
            row_out["PSyn_sent_Nat_mean"] = None
            row_out["PSyn_nc_Nat_mean"]   = None
            row_out["PSyn_sent_Neut"]     = None
            row_out["PSyn_nc_Neut"]       = None
            results.append(row_out)
            continue

        gold_syn = str(df_psyn.loc[nc, "gold_syn"]).strip()

        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            sentence      = str(row[col]).strip()
            bool_tags  = parse_bool_tags(row[tag_col])
            label      = "sentence" + col.split("_")[-1]
            sentence_psyn = str(df_psyn.loc[nc, f"{col}_PSyn"]).strip()

            emb_sent_orig = _get_sent_emb(sentence,      model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (sentence, bool_tags, model_name, model, tokenizer, device)
            emb_sent_psyn = _get_sent_emb(sentence_psyn, model_name, model, tokenizer, device)
            emb_syn       = _get_target_emb(sentence_psyn, gold_syn, bool_tags, model_name, model, tokenizer, device)

            row_out[f"PSyn_sent_{label}"] = calculating_similarity(emb_sent_orig, emb_sent_psyn)
            row_out[f"PSyn_nc_{label}"]   = calculating_similarity(emb_nc_orig,   emb_syn)
            sim_nat_sent.append(row_out[f"PSyn_sent_{label}"])
            sim_nat_nc.append  (row_out[f"PSyn_nc_{label}"])

        row_out["PSyn_sent_Nat_mean"] = float(np.mean(sim_nat_sent))
        row_out["PSyn_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))

        sentence_neut_psyn = str(df_psyn.loc[nc, "sentence_neut_PSyn"]).strip()
        emb_sent_neut      = _get_sent_emb(neutral,          model_name, model, tokenizer, device)
        emb_nc_neut        = _get_nc_emb  (neutral, tag_neutral, model_name, model, tokenizer, device)
        emb_sent_neut_psyn = _get_sent_emb(sentence_neut_psyn, model_name, model, tokenizer, device)
        emb_syn_neut       = _get_target_emb(sentence_neut_psyn, gold_syn, tag_neutral, model_name, model, tokenizer, device)

        row_out["PSyn_sent_Neut"] = calculating_similarity(emb_sent_neut, emb_sent_neut_psyn)
        row_out["PSyn_nc_Neut"]   = calculating_similarity(emb_nc_neut,   emb_syn_neut)

        results.append(row_out)

    return results

def compute_pcomp(df_main, model_name, model, tokenizer, device):
    
    frasi_nat = ["sentence_nat_1", "sentence_nat_2", "sentence_nat_3"]
    tag_nat   = ["tag_sentence_nat_1", "tag_sentence_nat_2", "tag_sentence_nat_3"]
    results = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PComp"):
        nc   = str(row["NC"]).strip()
        neutral        = str(row["sentence_neut"]).strip()
        tag_neutral    = parse_bool_tags(row["tag_sentence_neut"])
        content_words = extract_content_words(nc)
        word1, word2 = content_words[0], content_words[-1]

        row_out = {"NC": nc, "word1": word1, "word2": word2}
        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            sentence     = str(row[col]).strip()
            bool_tags = parse_bool_tags(row[tag_col])
            label     = "sentence" + col.split("_")[-1]

            emb_sent_orig = _get_sent_emb(sentence, model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (sentence, bool_tags, model_name, model, tokenizer, device)

            sentence_p1 = replace_nc(sentence, nc, word1)
            sentence_p2 = replace_nc(sentence, nc, word2)

            sim_p1_sent = calculating_similarity(emb_sent_orig, _get_sent_emb(sentence_p1, model_name, model, tokenizer, device))
            sim_p2_sent = calculating_similarity(emb_sent_orig, _get_sent_emb(sentence_p2, model_name, model, tokenizer, device))
            sim_p1_nc   = calculating_similarity(emb_nc_orig,   _get_target_emb(sentence_p1, word1, bool_tags, model_name, model, tokenizer, device))
            sim_p2_nc   = calculating_similarity(emb_nc_orig,   _get_target_emb(sentence_p2, word2, bool_tags, model_name, model, tokenizer, device))

            row_out[f"PComp_sent_{label}"] = select_most_sim(sim_p1_sent, sim_p2_sent)
            row_out[f"PComp_nc_{label}"]   = select_most_sim(sim_p1_nc,   sim_p2_nc)
            sim_nat_sent.append(row_out[f"PComp_sent_{label}"])
            sim_nat_nc.append  (row_out[f"PComp_nc_{label}"])

        row_out["PComp_sent_Nat_mean"] = float(np.mean(sim_nat_sent))
        row_out["PComp_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))

        emb_sent_neut = _get_sent_emb(neutral, model_name, model, tokenizer, device)
        emb_nc_neut   = _get_nc_emb  (neutral, tag_neutral, model_name, model, tokenizer, device)

        sentence_neut_p1 = replace_nc(neutral, nc, word1)
        sentence_neut_p2 = replace_nc(neutral, nc, word2)

        row_out["PComp_sent_Neut"] = select_most_sim(
            calculating_similarity(emb_sent_neut, _get_sent_emb(sentence_neut_p1, model_name, model, tokenizer, device)),
            calculating_similarity(emb_sent_neut, _get_sent_emb(sentence_neut_p2, model_name, model, tokenizer, device))
        )
        row_out["PComp_nc_Neut"] = select_most_sim(
            calculating_similarity(emb_nc_neut, _get_target_emb(sentence_neut_p1, word1, tag_neutral, model_name, model, tokenizer, device)),
            calculating_similarity(emb_nc_neut, _get_target_emb(sentence_neut_p2, word2, tag_neutral, model_name, model, tokenizer, device))
        )

        results.append(row_out)

    return results

def compute_pwordssyn(df_main, model_name, model, tokenizer, device):
    
    frasi_nat = ["sentence_nat_1", "sentence_nat_2", "sentence_nat_3"]
    tag_nat   = ["tag_sentence_nat_1", "tag_sentence_nat_2", "tag_sentence_nat_3"]
    results = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PWordsSyn"):
        nc = str(row["NC"]).strip()
        wordsyn     = str(row["wordsyn"]).strip()
        neutral      = str(row["sentence_neut"]).strip()
        tag_neutral  = parse_bool_tags(row["tag_sentence_neut"])

        row_out = {"NC": nc, "wordsyn": wordsyn}
        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            sentence     = str(row[col]).strip()
            bool_tags = parse_bool_tags(row[tag_col])
            label     = "sentence" + col.split("_")[-1]

            sentence_ws = replace_nc(sentence, nc, wordsyn)

            emb_sent_orig = _get_sent_emb(sentence,    model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (sentence, bool_tags, model_name, model, tokenizer, device)
            emb_sent_ws   = _get_sent_emb(sentence_ws, model_name, model, tokenizer, device)
            emb_ws        = _get_target_emb(sentence_ws, wordsyn, bool_tags, model_name, model, tokenizer, device)

            row_out[f"PWordsSyn_sent_{label}"] = calculating_similarity(emb_sent_orig, emb_sent_ws)
            row_out[f"PWordsSyn_nc_{label}"]   = calculating_similarity(emb_nc_orig,   emb_ws)
            sim_nat_sent.append(row_out[f"PWordsSyn_sent_{label}"])
            sim_nat_nc.append  (row_out[f"PWordsSyn_nc_{label}"])

        row_out["PWordsSyn_sent_Nat_mean"] = float(np.mean(sim_nat_sent))
        row_out["PWordsSyn_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))

        sentence_neut_ws = replace_nc(neutral, nc, wordsyn)
        emb_sent_neut    = _get_sent_emb(neutral,        model_name, model, tokenizer, device)
        emb_nc_neut      = _get_nc_emb  (neutral, tag_neutral, model_name, model, tokenizer, device)
        emb_sent_neut_ws = _get_sent_emb(sentence_neut_ws, model_name, model, tokenizer, device)
        emb_neut_ws      = _get_target_emb(sentence_neut_ws, wordsyn, tag_neutral, model_name, model, tokenizer, device)

        row_out["PWordsSyn_sent_Neut"] = calculating_similarity(emb_sent_neut, emb_sent_neut_ws)
        row_out["PWordsSyn_nc_Neut"]   = calculating_similarity(emb_nc_neut,   emb_neut_ws)

        results.append(row_out)

    return results

def compute_prand(df_main, nat_dfs, neut_dfs, model_name, model, tokenizer, device):
   
    frasi_nat = ["sentence_nat_1", "sentence_nat_2", "sentence_nat_3"]
    tag_nat   = ["tag_sentence_nat_1", "tag_sentence_nat_2", "tag_sentence_nat_3"]
    results = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PRand"):
        nc = str(row["NC"]).strip()
        neutral      = str(row["sentence_neut"]).strip()
        tag_neutral  = parse_bool_tags(row["tag_sentence_neut"])

        row_out = {"NC": nc}
        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            sentence     = str(row[col]).strip()
            bool_tags = parse_bool_tags(row[tag_col])
            label     = "sentence" + col.split("_")[-1]

            emb_sent_orig = _get_sent_emb(sentence, model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (sentence, bool_tags, model_name, model, tokenizer, device)

            sim_list_sent, sim_list_nc = [], []
            for nat_df in nat_dfs:
                try:
                    sentence_prand = str(nat_df.loc[nc, f"{col}_PRand"]).strip()
                    prand_str   = str(nat_df.loc[nc, "PRand"]).strip()
                except KeyError:
                    continue
                emb_sent_pr = _get_sent_emb(sentence_prand, model_name, model, tokenizer, device)
                emb_pr      = _get_target_emb(sentence_prand, prand_str, bool_tags, model_name, model, tokenizer, device)
                sim_list_sent.append(calculating_similarity(emb_sent_orig, emb_sent_pr))
                sim_list_nc.append  (calculating_similarity(emb_nc_orig,   emb_pr))

            row_out[f"PRand_sent_{label}"] = float(np.mean(sim_list_sent)) if sim_list_sent else None
            row_out[f"PRand_nc_{label}"]   = float(np.mean(sim_list_nc))   if sim_list_nc   else None
            if row_out[f"PRand_sent_{label}"] is not None:
                sim_nat_sent.append(row_out[f"PRand_sent_{label}"])
            if row_out[f"PRand_nc_{label}"] is not None:
                sim_nat_nc.append  (row_out[f"PRand_nc_{label}"])

        row_out["PRand_sent_Nat_mean"] = float(np.mean(sim_nat_sent)) if sim_nat_sent else None
        row_out["PRand_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))   if sim_nat_nc   else None

        emb_sent_neut = _get_sent_emb(neutral, model_name, model, tokenizer, device)
        emb_nc_neut   = _get_nc_emb  (neutral, tag_neutral, model_name, model, tokenizer, device)

        sim_neut_sent, sim_neut_nc = [], []
        for neut_df in neut_dfs:
            try:
                sentence_neut_pr  = str(neut_df.loc[nc, "sentence_neut_PRand"]).strip()
                prand_str_neut = str(neut_df.loc[nc, "PRand"]).strip()
            except KeyError:
                continue
            emb_sent_npr = _get_sent_emb(sentence_neut_pr, model_name, model, tokenizer, device)
            emb_npr      = _get_target_emb(sentence_neut_pr, prand_str_neut, tag_neutral, model_name, model, tokenizer, device)
            sim_neut_sent.append(calculating_similarity(emb_sent_neut, emb_sent_npr))
            sim_neut_nc.append  (calculating_similarity(emb_nc_neut,   emb_npr))

        row_out["PRand_sent_Neut"] = float(np.mean(sim_neut_sent)) if sim_neut_sent else None
        row_out["PRand_nc_Neut"]   = float(np.mean(sim_neut_nc))   if sim_neut_nc   else None

        results.append(row_out)

    return results

def sim_scores(probe, df_main, model_name, model, tokenizer, device,
               df_psyn=None, nat_dfs=None, neut_dfs=None):

    if probe == "PSyn":
        return compute_psyn(df_main, df_psyn, model_name, model, tokenizer, device)
    elif probe == "PComp":
        return compute_pcomp(df_main, model_name, model, tokenizer, device)
    elif probe == "PWordsSyn":
        return compute_pwordssyn(df_main, model_name, model, tokenizer, device)
    elif probe == "PRand":
        return compute_prand(df_main, nat_dfs, neut_dfs, model_name, model, tokenizer, device)
    else:
        raise ValueError(f"Unknown probe: {probe}")
