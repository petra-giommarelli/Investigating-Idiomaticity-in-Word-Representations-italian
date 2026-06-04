# Investigating Idiomaticity in Italian Word Representations

Italian extension of the NCIMP framework (He et al., 2024) for noun compound idiomaticity.

## Repository structure

- `dataset/` — Italian NC dataset and annotations
- `utils_it.py` — low-level embedding extraction for each model
- `for_probes_it.py` — probe construction and similarity computation
- `main_it.py` — entry point: runs the full pipeline

## Models

Model weights are loaded from HuggingFace:

- BERTita — `dbmdz/bert-base-italian-xxl-cased`
- mBERT — `google-bert/bert-base-multilingual-cased`
- mDistilBERT — `distilbert/distilbert-base-multilingual-cased`
- mSBERT — `sentence-transformers/distiluse-base-multilingual-cased`
- Minerva-3B — `sapienzanlp/Minerva-3B-base-v1.0`
- fastText — Italian vectors (Wikipedia + Common Crawl, 300d)
