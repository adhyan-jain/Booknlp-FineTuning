import json
from datasets import Dataset
from transformers import AutoTokenizer

DATA_FILE = "output_v3.json"
ENTITY_MODEL_PATH = r"C:\Users\Adhyan\booknlps\entities_google\bert_uncased_L-6_H-768_A-12"
OUTPUT_DATASET_PATH = "new_entity_dataset"


with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from {DATA_FILE}")

def make_ner_samples(data):
    """Create token-entity pairs from dataset."""
    samples = []
    for entry in data:
        text = entry.get("text", "").strip()
        if not text:
            continue

        entities = set(entry.get("entities", []) + entry.get("characters", []))

        samples.append({
            "tokens": text.split(),
            "entities": list(entities)
        })
    return samples

ner_samples = make_ner_samples(data)
print(f"Created {len(ner_samples)} samples for entity dataset")

tokenizer = AutoTokenizer.from_pretrained(ENTITY_MODEL_PATH)

label_list = ["O", "B-ENT", "I-ENT"]
label2id = {l: i for i, l in enumerate(label_list)}

def tokenize_and_align_labels(samples):
    tokenized_inputs = tokenizer(
        samples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    all_labels = []
    for words, entities in zip(samples["tokens"], samples["entities"]):
        word_labels = ["O"] * len(words)

        for ent in entities:
            ent_tokens = ent.split()
            for j in range(len(words) - len(ent_tokens) + 1):
                if words[j:j+len(ent_tokens)] == ent_tokens:
                    word_labels[j] = "B-ENT"
                    for k in range(j + 1, j + len(ent_tokens)):
                        word_labels[k] = "I-ENT"

        label_ids = [label2id.get(l, 0) for l in word_labels]
        label_ids = label_ids[:128] + [0] * (128 - len(label_ids))
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

ner_dataset = Dataset.from_dict({
    "tokens": [x["tokens"] for x in ner_samples],
    "entities": [x["entities"] for x in ner_samples],
})

ner_dataset = ner_dataset.map(tokenize_and_align_labels, batched=True)

ner_dataset.save_to_disk(OUTPUT_DATASET_PATH)
print(f"Entity dataset saved to `{OUTPUT_DATASET_PATH}`")
print("You can now load it later with:")
print(f"from datasets import load_from_disk\ndataset = load_from_disk('{OUTPUT_DATASET_PATH}')")
