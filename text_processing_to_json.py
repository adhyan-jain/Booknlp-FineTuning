import json
import subprocess
import signal
import sys
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock

INPUT_FILE = "LOTM.txt"
OUTPUT_FILE = "output.json"
BATCH_SIZE = 4
DEBUG_MODE = False
MAX_RETRIES = 2
MAX_PARALLEL_BATCHES = 1

stop_processing = False

mentioned_characters = set()

write_lock = Lock()

def signal_handler(sig, frame):
    """Handles Ctrl+C interruption."""
    global stop_processing
    print("\nInterrupted! Finishing current batches then stopping...")
    stop_processing = True

signal.signal(signal.SIGINT, signal_handler)

def query_ollama(prompt, model="qwen2.5:14b"):
    """Runs a prompt against the local Ollama instance."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=120
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Ollama query timed out.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Ollama command failed: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Ensure Ollama is installed and in your PATH.")
        return None

def build_prompt(sentences_list, context_chars):
    """Constructs the prompt for the LLM with context and sentences."""
    sentences_text = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(sentences_list) if s.strip()])

    context_info = ""
    if context_chars:
        context_info = f"\n**Context**: Characters seen so far: {', '.join(sorted(context_chars))}. Use this information to resolve pronouns."

    return f"""You are extracting structured data from the fantasy web novel "Lord of the Mysteries" by Cuttlefish That Loves Diving.
Return ONLY a valid JSON array. Do not include explanations, commentary, or extra text.

==============================
CORE CONCEPTS
==============================

**"characters"** = Proper names of people/beings
- Examples: Trissy, Frye, Leonard Mitchell, Dunn Smith
- Exclude: pronouns, body parts, or generic roles

**"entities"** = Tangible story elements
- Include: objects, locations, organizations, spells, magical items, and also character names
- Exclude: body parts (head, eyes), abstract/internal things (pain, thoughts, confusion), and generic roles unless capitalized

**"coref"** = Pronoun & contraction resolution
- Every pronoun/contraction must map to a full character name if possible
- Keys must always be lowercase: i, me, my, mine, he, him, his, she, her, hers, it, they, them, their, we, us, our
- Contractions expand before resolution
- Do NOT map pronouns to "Narrator"
- Resolve **first-person pronouns to the POV character** for that sentence
- Resolve **third-person pronouns to the correct character in context**
  - If multiple characters appear in the sentence, resolve pronouns carefully based on nearest mention or logical actor
  - Avoid assigning a pronoun to the wrong character
- Leave a pronoun unmapped only if it is impossible to confidently assign

**"speaker"** = Who is expressing this sentence
- If first-person pronouns exist, speaker = POV character
- Dialogue in quotes → quoted speaker
- Exclamations/interjections → speaker = POV character if first-person, otherwise "Narrator"
- Actions described in third-person → speaker = "Narrator" unless text implies a specific character performing it

==============================
VALIDATION RULES
==============================
- Keep all characters detected in the text, even minor ones
- Keep all entities detected
- Do not remove good coref mappings or speaker assignments that are already correct
- Correct only pronouns that are currently assigned to the wrong character
- Correct speaker only if first-person pronouns contradict the current speaker

==============================
SENTENCES
==============================
{sentences_text}

JSON OUTPUT:
"""

def parse_response(response):
    """Parses the raw LLM response, attempting to extract a JSON array."""
    if DEBUG_MODE:
        print(f"\n--- RAW RESPONSE ---\n{response[:300]}\n---")
    if not response:
        return []
    try:
        parsed = json.loads(response)
        return [parsed] if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[[\s\S]*\]', response, re.DOTALL)
    if match:
        json_text = match.group(0)
        json_text = json_text.replace("'", '"')
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    return []

def clean_and_validate(parsed, original_sentences, base_index):
    """Cleans and validates the parsed JSON objects, updating global character set and adding index."""
    global mentioned_characters
    
    PRONOUNS = {'i', 'me', 'my', 'mine', 'he', 'she', 'it', 'they', 'his', 'her', 'its',
                'their', 'him', 'them', 'we', 'us', 'our'}
    BODY_PARTS = {'head', 'heads', 'hand', 'hands', 'eye', 'eyes', 'arm', 'arms', 'leg', 'legs',
                  'body', 'bodies', 'finger', 'fingers', 'palm', 'palms', 'brain', 'brains',
                  'temple', 'temples', 'back', 'limbs', 'gaze', 'vision', 'face', 'chest'}
    ABSTRACT = {'pain', 'thought', 'thoughts', 'mind', 'minds', 'strength', 'confusion',
                'darkness', 'sleep', 'death', 'work', 'time', 'focus', 'memory', 'feeling'}
    EXCLUDED = PRONOUNS | BODY_PARTS | ABSTRACT

    cleaned = []

    pov_char_fallback = None
    if mentioned_characters:
        pov_char_fallback = sorted(list(mentioned_characters))[0]

    for idx, obj in enumerate(parsed):
        try:
            if not isinstance(obj, dict):
                continue

            text = obj.get("text", "")
            if not isinstance(text, str):
                if isinstance(text, (list, tuple)):
                    text = " ".join(str(item) for item in text if item)
                elif text is not None:
                    text = str(text)
                else:
                    text = ""
            text = text.strip()
            
            if not text:
                if idx < len(original_sentences):
                    fallback = original_sentences[idx]
                    if isinstance(fallback, str):
                        text = fallback.strip()
                    elif isinstance(fallback, (list, tuple)):
                        text = " ".join(str(item) for item in fallback if item).strip()
                    elif fallback is not None:
                        text = str(fallback).strip()
            if not text:
                print(f"⚠️ Skipping entry at index {base_index + idx}: no valid text found")
                continue

            characters = obj.get("characters", [])
            if not isinstance(characters, list):
                characters = []
            valid_characters = []
            for c in characters:
                if not isinstance(c, str):
                    if c is not None:
                        c = str(c)
                    else:
                        continue
                c = c.strip()
                if (c and len(c) > 1 and c[0].isupper() and c.lower() not in PRONOUNS and c in text):
                    valid_characters.append(c)
                    mentioned_characters.add(c)

            entities = obj.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            valid_entities = []
            for e in entities:
                if not isinstance(e, str):
                    if e is not None:
                        e = str(e)
                    else:
                        continue
                e = e.strip()
                if e and e.lower() not in EXCLUDED:
                    valid_entities.append(e)
            for c in valid_characters:
                if c not in valid_entities:
                    valid_entities.append(c)

            coref = obj.get("coref", {})
            if not isinstance(coref, dict):
                coref = {}
            fixed_coref = {}
            for pronoun, referent in coref.items():
                if not isinstance(referent, str):
                    if isinstance(referent, (list, tuple)) and referent:
                        referent = next((str(r) for r in referent if r), "")
                    elif referent is not None:
                        referent = str(referent)
                    else:
                        continue
                
                referent = referent.strip()
                if not referent:
                    continue
                    
                if not isinstance(pronoun, str):
                    pronoun = str(pronoun) if pronoun is not None else ""
                
                pronoun = pronoun.strip().lower()
                if not pronoun:
                    continue
                    
                referent = re.sub(r'[\[\]]', '', referent).strip()
                if pronoun == referent.lower() or referent.lower() in PRONOUNS:
                    continue
                if len(referent) > 1 and referent[0].isupper():
                    fixed_coref[pronoun] = referent

            text_lower = text.lower()
            has_first_person = any(p in text_lower.split() for p in ['i', 'my', 'me', "i'm", "i've", "i'd"])
            
            speaker = obj.get("speaker", "Narrator")
            if not isinstance(speaker, str):
                if isinstance(speaker, (list, tuple)) and speaker:
                    speaker = str(speaker[0]) if speaker[0] else "Narrator"
                elif speaker is not None:
                    speaker = str(speaker)
                else:
                    speaker = "Narrator"
            speaker = speaker.strip()
            
            if has_first_person and (speaker in ["Narrator", ""] or speaker.lower() in PRONOUNS):
                current_pov_char = pov_char_fallback or "Unknown Character" 
                
                if valid_characters:
                     current_pov_char = valid_characters[0]

                if current_pov_char != "Unknown Character":
                    speaker = current_pov_char
                    if 'i' in text_lower.split() and 'i' not in fixed_coref:
                        fixed_coref['i'] = current_pov_char
                    if 'my' in text_lower.split() and 'my' not in fixed_coref:
                        fixed_coref['my'] = current_pov_char
                    if 'me' in text_lower.split() and 'me' not in fixed_coref:
                        fixed_coref['me'] = current_pov_char

            if not speaker or speaker.lower() in PRONOUNS:
                speaker = "Narrator"
            
            cleaned.append({
                "index": base_index + idx,
                "text": text,
                "characters": valid_characters,
                "entities": valid_entities,
                "coref": fixed_coref,
                "speaker": speaker
            })
        
        except Exception as e:
            print(f"Error processing entry at index {base_index + idx}: {e}")
            if DEBUG_MODE:
                print(f"   Problematic object: {obj}")
                import traceback
                traceback.print_exc()
            continue

    return cleaned

def process_batch(batch_idx, batch_sentences, batch_start_idx):
    if stop_processing:
        return None
    
    if not isinstance(batch_sentences, list):
        print(f"Batch {batch_idx+1}: batch_sentences is not a list: {type(batch_sentences)}")
        return None
    
    flattened_sentences = []
    for item in batch_sentences:
        if isinstance(item, str):
            flattened_sentences.append(item)
        elif isinstance(item, (list, tuple)):
            for subitem in item:
                if isinstance(subitem, str):
                    flattened_sentences.append(subitem)
        elif item is not None:
            flattened_sentences.append(str(item))
    
    if not flattened_sentences:
        print(f"Batch {batch_idx+1}:No valid sentences after flattening")
        return None
    
    parsed = None
    for attempt in range(MAX_RETRIES):
        # Pass the global set of mentioned characters for context
        prompt = build_prompt(flattened_sentences, mentioned_characters)
        response = query_ollama(prompt)
        
        if response:
            parsed = parse_response(response)
            if parsed:
                break
            
    if not parsed:
        print(f"Batch {batch_idx+1}: Failed after {MAX_RETRIES} attempts.")
        return None
    
    try:
        cleaned = clean_and_validate(parsed, flattened_sentences, batch_start_idx)
    except Exception as e:
        print(f"Batch {batch_idx+1}: Error during validation: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
            print(f"Parsed data that caused error: {parsed[:1] if parsed else 'None'}")
        return None
    
    if not cleaned:
        print(f"Batch {batch_idx+1}: Empty or fully filtered out.")
        return None
        
    print(f"Batch {batch_idx+1}: ({len(cleaned)} items)")
    return cleaned

def load_existing_output_safe():
    processed_indices = set()
    global mentioned_characters
    
    path = Path(OUTPUT_FILE)
    if not path.exists():
        return processed_indices

    print(f"Checking existing output file '{OUTPUT_FILE}' for resume point...")
    
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            return processed_indices
            
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        idx = obj.get("index", -1)
                        if idx >= 0:
                            processed_indices.add(idx)
                        for c in obj.get("characters", []):
                            if isinstance(c, str):
                                mentioned_characters.add(c)
                return processed_indices
        except json.JSONDecodeError:
            pass
        
        content_wrapped = "[" + content.rstrip(",").rstrip() + "]"
        
        try:
            data = json.loads(content_wrapped)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        idx = obj.get("index", -1)
                        if idx >= 0:
                            processed_indices.add(idx)
                        for c in obj.get("characters", []):
                            if isinstance(c, str):
                                mentioned_characters.add(c)
                if processed_indices:
                    max_idx = max(processed_indices)
                    print(f"Successfully loaded {len(processed_indices)} processed sentences (up to index {max_idx})")
                return processed_indices
        except json.JSONDecodeError:
            pass
        
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, content)
        
        for match in matches:
            try:
                obj = json.loads(match)
                idx = obj.get("index", -1)
                if idx >= 0:
                    processed_indices.add(idx)
                for c in obj.get("characters", []):
                    if isinstance(c, str):
                        mentioned_characters.add(c)
            except json.JSONDecodeError:
                continue
        
        if processed_indices:
            max_idx = max(processed_indices)
            print(f"Successfully loaded {len(processed_indices)} processed sentences (up to index {max_idx})")
        else:
            print(f"Could not parse any valid JSON objects from the file")
                
    except Exception as e:
        print(f"Could not fully parse existing JSON safely due to error: {e}. Will resume cautiously.")
        
    return processed_indices

def write_batch_results(batch_results, processed_indices):
    with write_lock:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
            for obj in batch_results:
                obj_index = obj.get("index", -1)
                if obj_index not in processed_indices and obj_index >= 0:
                    outfile.write(json.dumps(obj, ensure_ascii=False, indent=2))
                    outfile.write(",\n")
                    processed_indices.add(obj_index)
            outfile.flush()

def main():
    global mentioned_characters
    processed_indices = load_existing_output_safe()
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as infile:
            content = infile.read()
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found!")
        sys.exit(1)

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content)
                 if s.strip() and not re.match(r'\[CHAPTER_START_\d+\]', s)]

    total_sentences = len(sentences)
    sentences_to_process = []
    
    for idx, sentence in enumerate(sentences):
        if idx not in processed_indices:
            sentences_to_process.append((idx, sentence))
    
    if not sentences_to_process:
        print("All sentences already processed. Nothing to do.")
        print(f"Total unique characters found in output: {len(mentioned_characters)}")
        return

    print(f"Total sentences: {total_sentences}")
    print(f"Sentences already processed: {len(processed_indices)}")
    print(f"Characters loaded from previous run: {len(mentioned_characters)}")
    print(f"Sentences remaining to process: {len(sentences_to_process)}")

    batches = []
    current_batch_sentences = []
    current_batch_indices = []
    
    for idx, sentence in sentences_to_process:
        current_batch_sentences.append(sentence)
        current_batch_indices.append(idx)
        
        if len(current_batch_sentences) == BATCH_SIZE:
            batches.append((current_batch_sentences[:], current_batch_indices[0]))
            current_batch_sentences = []
            current_batch_indices = []
    
    if current_batch_sentences:
        batches.append((current_batch_sentences, current_batch_indices[0]))
    
    if not batches:
        print("No batches to process.")
        return
        
    print(f"Total batches to process: {len(batches)}")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        futures = {}
        for i, (batch_sentences, batch_start_idx) in enumerate(batches):
            future = executor.submit(process_batch, i, batch_sentences, batch_start_idx)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                cleaned = future.result()
                if cleaned:
                    write_batch_results(cleaned, processed_indices)
            except Exception as e:
                print(f"Batch {idx+1} failed with error: {e}")
            
            if stop_processing:
                for f in futures:
                    if not f.done():
                        f.cancel()
                print("\nStopping after finishing current batch...")
                break

    print(f"\nProcessing complete.")
    print(f"Total unique characters found: {len(mentioned_characters)}")
    print(f"Total processed indices: {len(processed_indices)}")
    print("Reminder: The output file contains comma-separated JSON objects.")
    print("You will need to manually wrap the entire content with `[` and `]` for a valid JSON array after the *final* run.")


if __name__ == "__main__":
    main()
