# BookNLP

BookNLP is a natural language processing pipeline that scales to books and other long documents (in English), including:

- Part-of-speech tagging
- Dependency parsing
- Entity recognition
- Character name clustering (e.g., "Tom", "Tom Sawyer", "Mr. Sawyer", "Thomas Sawyer" -> TOM_SAWYER) and coreference resolution
- Quotation speaker identification
- Supersense tagging (e.g., "animal", "artifact", "body", "cognition", etc.)
- Event tagging
- Referential gender inference (TOM_SAWYER -> he/him/his)

BookNLP ships with two models, both with identical architectures but different underlying BERT sizes. The larger and more accurate `big` model is fit for GPUs and multi-core computers; the faster `small` model is more appropriate for personal computers. See the table below for a comparison of the difference, both in terms of overall speed and in accuracy for the tasks that BookNLP performs.

|                         | Small | Big  |
| ----------------------- | ----- | ---- |
| Entity tagging (F1)     | 88.2  | 90.0 |
| Supersense tagging (F1) | 73.2  | 76.2 |
| Event tagging (F1)      | 70.6  | 74.1 |

# booknlp-research

This repository collects preprocessing, dataset preparation, and fine-tuning utilities used for experiments on the book "Lord of the Mysteries" (LOTM). It builds on the BookNLP codebase and includes scripts to:

- prepare token / entity / coreference / speaker datasets
- train and fine-tune token-classification models (entities, speakers)
- run an end-to-end pipeline for producing annotated book outputs

The codebase is intended for research and experimentation; it's set up for local training and evaluation with Hugging Face Transformers and Datasets.

## Repository layout

- `LOTM/` — source book files and derived artifacts (.book, .tokens, .entities, .quotes, .supersense)
- `data/` — small data splits and preprocessing inputs
  - `coref/`, `ner/`, `speaker/`
- `entity_dataset/` — a Hugging Face dataset saved to disk (arrow files and cache) used for entity fine-tuning
- `models/` — pre-trained and fine-tuned model checkpoints (BERT variants and custom fine-tuned directories)
- Top-level scripts:
  - `train_LOTM.py` — fine-tune an entity token-classification model using Hugging Face `Trainer` (BERT)
  - `booknlp_run.py` — higher-level runner for processing the book and exporting annotations
  - `text_processing_to_json.py` — convert raw book text into JSON artifacts for preprocessing
  - `coref_and_entity_dataset.py` — builds datasets for coreference and entity tasks
  - `clean_output_v1.py`, `clean_output_v2.py` — post-processing / cleaning utilities

## Quick start

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# On Windows (Git Bash / bash.exe):
source venv/Scripts/activate
```

2. Install minimal dependencies (adjust versions for your environment):

```bash
pip install -U pip
pip install torch transformers datasets
```

3. Prepare dataset artifacts. Example: ensure `entity_dataset/` contains a HF dataset saved with `datasets.save_to_disk()` (this repo includes cached files under `entity_dataset/`).

4. Configure paths in `train_LOTM.py` if needed (variables at the top like `ENTITY_MODEL_PATH` and `SAVE_DIR`) and run training:

```bash
python train_LOTM.py
```

## Notes and compatibility

- Transformers API: The `TrainingArguments` API changed across transformers versions. If you see errors like:

  "TrainingArguments.**init**() got an unexpected keyword argument 'evaluation_strategy'"

  it means an older `transformers` package is installed. The `train_LOTM.py` script now contains a small compatibility helper that inspects the `TrainingArguments` constructor and only passes supported keyword arguments. Recommended transformer versions: `transformers>=4.4.0` (but avoid major 5.0 until tested).

- PyTorch: install a wheel appropriate for your CUDA/CPU configuration. For CPU-only:

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

- Model files: `models/` contains pretrained and fine-tuned checkpoints. Ensure your `ENTITY_MODEL_PATH` in `train_LOTM.py` points to a folder with tokenizer and model files (e.g., `tokenizer.json`, `vocab.txt`, `config.json`, `model.safetensors` or `pytorch_model.bin`).

## Scripts of interest

- `train_LOTM.py` — loads `entity_dataset/`, a tokenizer and model, and fine-tunes using the Hugging Face `Trainer`. Batch sizes and epochs are configurable at the top of the script. The script includes a compatibility layer for different `transformers` releases.
- `booknlp_run.py` — higher-level orchestration; run this to process the book end-to-end and generate `.tokens`, `.entities`, `.quotes`, `.supersense`, and `.book.html` outputs.
- `coref_and_entity_dataset.py` — utilities for assembling datasets for coreference / entity tasks; outputs are placed under `data/` and `entity_dataset/`.

Note: the repository currently includes a fine-tuned entity model (see `models/fine_tuned/entity` and `train_LOTM.py`). The same preprocessing and training pipeline can be extended to produce fine-tuned models for coreference and speaker identification as well — use `coref_and_entity_dataset.py` and the data under `data/speaker/` as starting points, and adapt `train_LOTM.py` (or create analogous training scripts) to train coref/speaker models.

## Troubleshooting

- If training fails due to out-of-memory, reduce `per_device_train_batch_size` in `train_LOTM.py`.
- If the tokenizer cannot be loaded, verify the model folder contains expected tokenizer files (`tokenizer.json`, `vocab.txt`/`vocab.json`).
- If transformers import is very slow or hangs during `import transformers`, ensure your Python environment and PyTorch installation are correct — importing `transformers` pulls in `torch` which may trigger heavy initialization on some setups.

## Contact / License

See `LICENSE` at the repository root for licensing. For questions about this repo, open an issue or contact the maintainer.

## Acknowledgments

This work builds on the BookNLP project and uses datasets and models described in the repository. See included files and headers for paper references and dataset citations.
