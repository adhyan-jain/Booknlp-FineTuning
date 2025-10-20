import json

INPUT_FILE = "output_v2.json"
OUTPUT_FILE = "output_v3.json"

EXCLUDED = {
    "eye", "eyes", "head", "hand", "hands", "pain", "darkness", "confusion",
    "body", "mind", "thought", "thoughts", "feeling", "breath", "sight", "time",
    "work", "face", "legs", "arms", "finger", "fingers", "memory", "sleep", "death"
}

PRONOUNS = {
    "i", "me", "my", "mine", "he", "him", "his", "she", "her", "hers",
    "it", "they", "them", "their", "we", "us", "our"
}

def normalize_name(name):
    name = name.strip()
    if name.isupper():
        name = name.capitalize()
    elif len(name.split()) > 1:
        name = " ".join(word.capitalize() for word in name.split())
    else:
        name = name.capitalize()
    return name

def clean_coref(coref):
    if not isinstance(coref, dict):
        return {}
    fixed = {}
    for pronoun, referent in coref.items():
        if not isinstance(pronoun, str) or not isinstance(referent, str):
            continue
        pronoun = pronoun.lower().strip()
        referent = referent.strip()
        if pronoun in PRONOUNS and referent and referent.lower() not in PRONOUNS:
            fixed[pronoun] = normalize_name(referent)
    return fixed

def clean_dataset(data):
    cleaned = []
    seen_texts = set()

    for obj in data:
        if not isinstance(obj, dict):
            continue

        text = obj.get("text", "").strip()
        if not text or len(text.split()) < 3:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)

        characters = obj.get("characters", [])
        entities = obj.get("entities", [])
        speaker = obj.get("speaker", "Narrator").strip()
        coref = obj.get("coref", {})

        cleaned_characters = []
        for c in characters:
            if not isinstance(c, str):
                continue
            c = c.strip()
            if not c or c.lower() in EXCLUDED or c.lower() in PRONOUNS:
                continue
            cleaned_characters.append(normalize_name(c))

        cleaned_entities = []
        for e in entities:
            if not isinstance(e, str):
                continue
            e = e.strip()
            if not e or e.lower() in EXCLUDED:
                continue
            cleaned_entities.append(e)
        for c in cleaned_characters:
            if c not in cleaned_entities:
                cleaned_entities.append(c)

        fixed_coref = clean_coref(coref)

        if not isinstance(speaker, str) or not speaker.strip():
            speaker = "Narrator"
        if speaker.lower() in PRONOUNS:
            if cleaned_characters:
                speaker = cleaned_characters[0]
            else:
                speaker = "Narrator"
        elif speaker.lower() == "unknown":
            speaker = "Narrator"

        if not cleaned_characters and speaker == "Narrator" and len(cleaned_entities) < 2:
            continue

        cleaned.append({
            "text": text,
            "characters": list(sorted(set(cleaned_characters))),
            "entities": list(sorted(set(cleaned_entities))),
            "coref": fixed_coref,
            "speaker": speaker
        })
    return cleaned

def main():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content.startswith("["):
                content = "[" + content.rstrip(",") + "]"
            data = json.loads(content)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    print(f"Loaded {len(data)} items from {INPUT_FILE}")
    cleaned = clean_dataset(data)
    print(f"Cleaned dataset size: {len(cleaned)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(cleaned, out, ensure_ascii=False, indent=2)

    print(f"Saved cleaned dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
