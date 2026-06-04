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

    frasi_nat = ["frase 1", "frase 2", "frase 3"]
    tag_nat   = ["tag frase 1", "tag frase 2", "tag frase 3"]
    risultati = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PSyn"):
        espressione = str(row["ESPRESSIONE"]).strip()
        neutra      = str(row["Frasi Neutre"]).strip()
        tag_neutra  = parse_bool_tags(row["tag Frasi Neutre"])

        riga = {"ESPRESSIONE": espressione}

        if espressione not in df_psyn.index:
            for col in frasi_nat:
                label = col.replace(" ", "")
                riga[f"PSyn_sent_{label}"] = None
                riga[f"PSyn_nc_{label}"]   = None
            riga["PSyn_sent_Nat_mean"] = None
            riga["PSyn_nc_Nat_mean"]   = None
            riga["PSyn_sent_Neut"]     = None
            riga["PSyn_nc_Neut"]       = None
            risultati.append(riga)
            continue

        gold_syn = str(df_psyn.loc[espressione, "gold_syn"]).strip()

        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            frase      = str(row[col]).strip()
            bool_tags  = parse_bool_tags(row[tag_col])
            label      = col.replace(" ", "")
            frase_psyn = str(df_psyn.loc[espressione, f"{col} PSyn"]).strip()

            emb_sent_orig = _get_sent_emb(frase,      model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (frase, bool_tags, model_name, model, tokenizer, device)
            emb_sent_psyn = _get_sent_emb(frase_psyn, model_name, model, tokenizer, device)
            emb_syn       = _get_target_emb(frase_psyn, gold_syn, bool_tags, model_name, model, tokenizer, device)

            riga[f"PSyn_sent_{label}"] = calculating_similarity(emb_sent_orig, emb_sent_psyn)
            riga[f"PSyn_nc_{label}"]   = calculating_similarity(emb_nc_orig,   emb_syn)
            sim_nat_sent.append(riga[f"PSyn_sent_{label}"])
            sim_nat_nc.append  (riga[f"PSyn_nc_{label}"])

        riga["PSyn_sent_Nat_mean"] = float(np.mean(sim_nat_sent))
        riga["PSyn_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))

        frase_neut_psyn = str(df_psyn.loc[espressione, "Frase neutra PSyn"]).strip()
        emb_sent_neut      = _get_sent_emb(neutra,          model_name, model, tokenizer, device)
        emb_nc_neut        = _get_nc_emb  (neutra, tag_neutra, model_name, model, tokenizer, device)
        emb_sent_neut_psyn = _get_sent_emb(frase_neut_psyn, model_name, model, tokenizer, device)
        emb_syn_neut       = _get_target_emb(frase_neut_psyn, gold_syn, tag_neutra, model_name, model, tokenizer, device)

        riga["PSyn_sent_Neut"] = calculating_similarity(emb_sent_neut, emb_sent_neut_psyn)
        riga["PSyn_nc_Neut"]   = calculating_similarity(emb_nc_neut,   emb_syn_neut)

        risultati.append(riga)

    return risultati

def compute_pcomp(df_main, model_name, model, tokenizer, device):
    
    frasi_nat = ["frase 1", "frase 2", "frase 3"]
    tag_nat   = ["tag frase 1", "tag frase 2", "tag frase 3"]
    risultati = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PComp"):
        espressione   = str(row["ESPRESSIONE"]).strip()
        neutra        = str(row["Frasi Neutre"]).strip()
        tag_neutra    = parse_bool_tags(row["tag Frasi Neutre"])
        content_words = extract_content_words(espressione)
        parola1, parola2 = content_words[0], content_words[-1]

        riga = {"ESPRESSIONE": espressione, "parola1": parola1, "parola2": parola2}
        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            frase     = str(row[col]).strip()
            bool_tags = parse_bool_tags(row[tag_col])
            label     = col.replace(" ", "")

            emb_sent_orig = _get_sent_emb(frase, model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (frase, bool_tags, model_name, model, tokenizer, device)

            frase_p1 = replace_nc(frase, espressione, parola1)
            frase_p2 = replace_nc(frase, espressione, parola2)

            sim_p1_sent = calculating_similarity(emb_sent_orig, _get_sent_emb(frase_p1, model_name, model, tokenizer, device))
            sim_p2_sent = calculating_similarity(emb_sent_orig, _get_sent_emb(frase_p2, model_name, model, tokenizer, device))
            sim_p1_nc   = calculating_similarity(emb_nc_orig,   _get_target_emb(frase_p1, parola1, bool_tags, model_name, model, tokenizer, device))
            sim_p2_nc   = calculating_similarity(emb_nc_orig,   _get_target_emb(frase_p2, parola2, bool_tags, model_name, model, tokenizer, device))

            riga[f"PComp_sent_{label}"] = select_most_sim(sim_p1_sent, sim_p2_sent)
            riga[f"PComp_nc_{label}"]   = select_most_sim(sim_p1_nc,   sim_p2_nc)
            sim_nat_sent.append(riga[f"PComp_sent_{label}"])
            sim_nat_nc.append  (riga[f"PComp_nc_{label}"])

        riga["PComp_sent_Nat_mean"] = float(np.mean(sim_nat_sent))
        riga["PComp_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))

        emb_sent_neut = _get_sent_emb(neutra, model_name, model, tokenizer, device)
        emb_nc_neut   = _get_nc_emb  (neutra, tag_neutra, model_name, model, tokenizer, device)

        frase_neut_p1 = replace_nc(neutra, espressione, parola1)
        frase_neut_p2 = replace_nc(neutra, espressione, parola2)

        riga["PComp_sent_Neut"] = select_most_sim(
            calculating_similarity(emb_sent_neut, _get_sent_emb(frase_neut_p1, model_name, model, tokenizer, device)),
            calculating_similarity(emb_sent_neut, _get_sent_emb(frase_neut_p2, model_name, model, tokenizer, device))
        )
        riga["PComp_nc_Neut"] = select_most_sim(
            calculating_similarity(emb_nc_neut, _get_target_emb(frase_neut_p1, parola1, tag_neutra, model_name, model, tokenizer, device)),
            calculating_similarity(emb_nc_neut, _get_target_emb(frase_neut_p2, parola2, tag_neutra, model_name, model, tokenizer, device))
        )

        risultati.append(riga)

    return risultati

def compute_pwordssyn(df_main, model_name, model, tokenizer, device):
    
    frasi_nat = ["frase 1", "frase 2", "frase 3"]
    tag_nat   = ["tag frase 1", "tag frase 2", "tag frase 3"]
    risultati = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PWordsSyn"):
        espressione = str(row["ESPRESSIONE"]).strip()
        wordsyn     = str(row["wordsyn"]).strip()
        neutra      = str(row["Frasi Neutre"]).strip()
        tag_neutra  = parse_bool_tags(row["tag Frasi Neutre"])

        riga = {"ESPRESSIONE": espressione, "wordsyn": wordsyn}
        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            frase     = str(row[col]).strip()
            bool_tags = parse_bool_tags(row[tag_col])
            label     = col.replace(" ", "")

            frase_ws = replace_nc(frase, espressione, wordsyn)

            emb_sent_orig = _get_sent_emb(frase,    model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (frase, bool_tags, model_name, model, tokenizer, device)
            emb_sent_ws   = _get_sent_emb(frase_ws, model_name, model, tokenizer, device)
            emb_ws        = _get_target_emb(frase_ws, wordsyn, bool_tags, model_name, model, tokenizer, device)

            riga[f"PWordsSyn_sent_{label}"] = calculating_similarity(emb_sent_orig, emb_sent_ws)
            riga[f"PWordsSyn_nc_{label}"]   = calculating_similarity(emb_nc_orig,   emb_ws)
            sim_nat_sent.append(riga[f"PWordsSyn_sent_{label}"])
            sim_nat_nc.append  (riga[f"PWordsSyn_nc_{label}"])

        riga["PWordsSyn_sent_Nat_mean"] = float(np.mean(sim_nat_sent))
        riga["PWordsSyn_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))

        frase_neut_ws = replace_nc(neutra, espressione, wordsyn)
        emb_sent_neut    = _get_sent_emb(neutra,        model_name, model, tokenizer, device)
        emb_nc_neut      = _get_nc_emb  (neutra, tag_neutra, model_name, model, tokenizer, device)
        emb_sent_neut_ws = _get_sent_emb(frase_neut_ws, model_name, model, tokenizer, device)
        emb_neut_ws      = _get_target_emb(frase_neut_ws, wordsyn, tag_neutra, model_name, model, tokenizer, device)

        riga["PWordsSyn_sent_Neut"] = calculating_similarity(emb_sent_neut, emb_sent_neut_ws)
        riga["PWordsSyn_nc_Neut"]   = calculating_similarity(emb_nc_neut,   emb_neut_ws)

        risultati.append(riga)

    return risultati

def compute_prand(df_main, nat_dfs, neut_dfs, model_name, model, tokenizer, device):
   
    frasi_nat = ["frase 1", "frase 2", "frase 3"]
    tag_nat   = ["tag frase 1", "tag frase 2", "tag frase 3"]
    risultati = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="PRand"):
        espressione = str(row["ESPRESSIONE"]).strip()
        neutra      = str(row["Frasi Neutre"]).strip()
        tag_neutra  = parse_bool_tags(row["tag Frasi Neutre"])

        riga = {"ESPRESSIONE": espressione}
        sim_nat_sent, sim_nat_nc = [], []

        for col, tag_col in zip(frasi_nat, tag_nat):
            frase     = str(row[col]).strip()
            bool_tags = parse_bool_tags(row[tag_col])
            label     = col.replace(" ", "")

            emb_sent_orig = _get_sent_emb(frase, model_name, model, tokenizer, device)
            emb_nc_orig   = _get_nc_emb  (frase, bool_tags, model_name, model, tokenizer, device)

            sim_list_sent, sim_list_nc = [], []
            for nat_df in nat_dfs:
                try:
                    frase_prand = str(nat_df.loc[espressione, f"{col} PRand"]).strip()
                    prand_str   = str(nat_df.loc[espressione, "PRand"]).strip()
                except KeyError:
                    continue
                emb_sent_pr = _get_sent_emb(frase_prand, model_name, model, tokenizer, device)
                emb_pr      = _get_target_emb(frase_prand, prand_str, bool_tags, model_name, model, tokenizer, device)
                sim_list_sent.append(calculating_similarity(emb_sent_orig, emb_sent_pr))
                sim_list_nc.append  (calculating_similarity(emb_nc_orig,   emb_pr))

            riga[f"PRand_sent_{label}"] = float(np.mean(sim_list_sent)) if sim_list_sent else None
            riga[f"PRand_nc_{label}"]   = float(np.mean(sim_list_nc))   if sim_list_nc   else None
            if riga[f"PRand_sent_{label}"] is not None:
                sim_nat_sent.append(riga[f"PRand_sent_{label}"])
            if riga[f"PRand_nc_{label}"] is not None:
                sim_nat_nc.append  (riga[f"PRand_nc_{label}"])

        riga["PRand_sent_Nat_mean"] = float(np.mean(sim_nat_sent)) if sim_nat_sent else None
        riga["PRand_nc_Nat_mean"]   = float(np.mean(sim_nat_nc))   if sim_nat_nc   else None

        emb_sent_neut = _get_sent_emb(neutra, model_name, model, tokenizer, device)
        emb_nc_neut   = _get_nc_emb  (neutra, tag_neutra, model_name, model, tokenizer, device)

        sim_neut_sent, sim_neut_nc = [], []
        for neut_df in neut_dfs:
            try:
                frase_neut_pr  = str(neut_df.loc[espressione, "Frase neutra PRand"]).strip()
                prand_str_neut = str(neut_df.loc[espressione, "PRand"]).strip()
            except KeyError:
                continue
            emb_sent_npr = _get_sent_emb(frase_neut_pr, model_name, model, tokenizer, device)
            emb_npr      = _get_target_emb(frase_neut_pr, prand_str_neut, tag_neutra, model_name, model, tokenizer, device)
            sim_neut_sent.append(calculating_similarity(emb_sent_neut, emb_sent_npr))
            sim_neut_nc.append  (calculating_similarity(emb_nc_neut,   emb_npr))

        riga["PRand_sent_Neut"] = float(np.mean(sim_neut_sent)) if sim_neut_sent else None
        riga["PRand_nc_Neut"]   = float(np.mean(sim_neut_nc))   if sim_neut_nc   else None

        risultati.append(riga)

    return risultati

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
