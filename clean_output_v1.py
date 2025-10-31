import json
from collections import defaultdict, Counter
from pathlib import Path
import random


INPUT_FILE = "output.json"
OUTPUT_FILE = "output_v2.json"
ANALYSIS_REPORT = "quality_report.txt"

class DataQualityAnalyzer:
    def __init__(self):
        self.all_characters = Counter()
        self.all_speakers = Counter()
        self.coref_patterns = defaultdict(Counter)
        self.errors = {
            'pronoun_in_characters': [],
            'invalid_coref': [],
            'speaker_mismatch': [],
            'missing_data': [],
            'duplicate_indices': []
        }
        
    def load_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content.startswith('['):
            content = '[' + content.rstrip(',') + ']'
        
        try:
            data = json.loads(content)
            print(f"Loaded {len(data)} entries")
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
    
    def analyze(self, data):
        print("\nANALYZING DATA QUALITY...\n")
        
        PRONOUNS = {'i', 'me', 'my', 'mine', 'he', 'she', 'it', 'they', 'his', 'her', 
                   'its', 'their', 'him', 'them', 'we', 'us', 'our', 'you', 'your'}
        
        seen_indices = set()
        
        for idx, entry in enumerate(data):
            entry_idx = entry.get('index', idx)
            
            if entry_idx in seen_indices:
                self.errors['duplicate_indices'].append(entry_idx)
            seen_indices.add(entry_idx)
            
            characters = entry.get('characters', [])
            for char in characters:
                if char.lower() in PRONOUNS:
                    self.errors['pronoun_in_characters'].append({
                        'index': entry_idx,
                        'pronoun': char,
                        'text': entry.get('text', '')[:100]
                    })
                else:
                    self.all_characters[char] += 1
            
            speaker = entry.get('speaker', 'Narrator')
            self.all_speakers[speaker] += 1
            
            coref = entry.get('coref', {})
            for pronoun, referent in coref.items():
                self.coref_patterns[pronoun][referent] += 1
                
                if referent.lower() in PRONOUNS or referent == 'Narrator':
                    self.errors['invalid_coref'].append({
                        'index': entry_idx,
                        'pronoun': pronoun,
                        'referent': referent,
                        'text': entry.get('text', '')[:100]
                    })
            
            text = entry.get('text', '').lower()
            has_first_person = any(p in text.split() for p in ['i', 'my', 'me'])
            
            if has_first_person and speaker == 'Narrator':
                self.errors['speaker_mismatch'].append({
                    'index': entry_idx,
                    'speaker': speaker,
                    'text': entry.get('text', '')[:100]
                })
            
            if not entry.get('text'):
                self.errors['missing_data'].append(entry_idx)
        
        self.print_analysis()
    
    def print_analysis(self):
        report = []
        report.append("=" * 80)
        report.append("DATA QUALITY ANALYSIS REPORT")
        report.append("=" * 80)
        
        report.append(f"\nCHARACTER STATISTICS:")
        report.append(f"Total unique characters: {len(self.all_characters)}")
        report.append(f"Top 20 characters:")
        for char, count in self.all_characters.most_common(20):
            report.append(f"      {char}: {count} mentions")
        
        report.append(f"\nSPEAKER STATISTICS:")
        report.append(f"Total unique speakers: {len(self.all_speakers)}")
        report.append(f"Top 10 speakers:")
        for speaker, count in self.all_speakers.most_common(10):
            report.append(f"{speaker}: {count} times")
        
        report.append(f"\nCOREFERENCE PATTERNS:")
        for pronoun in ['he', 'she', 'i', 'they', 'him', 'her']:
            if pronoun in self.coref_patterns:
                report.append(f"   '{pronoun}' most often refers to:")
                for referent, count in self.coref_patterns[pronoun].most_common(5):
                    report.append(f"      {referent}: {count} times")
        
        report.append(f"\nERRORS FOUND:")
        report.append(f"Pronouns in characters list: {len(self.errors['pronoun_in_characters'])}")
        report.append(f"Invalid coreference mappings: {len(self.errors['invalid_coref'])}")
        report.append(f"Speaker mismatches: {len(self.errors['speaker_mismatch'])}")
        report.append(f"Missing data: {len(self.errors['missing_data'])}")
        report.append(f"Duplicate indices: {len(self.errors['duplicate_indices'])}")
        
        if self.errors['pronoun_in_characters']:
            report.append(f"\n   Sample pronoun errors (first 5):")
            for err in self.errors['pronoun_in_characters'][:5]:
                report.append(f"      Index {err['index']}: '{err['pronoun']}' in {err['text']}")
        
        if self.errors['speaker_mismatch']:
            report.append(f"\n   Sample speaker mismatches (first 5):")
            for err in self.errors['speaker_mismatch'][:5]:
                report.append(f"      Index {err['index']}: {err['text']}")
        
        report.append("\n" + "=" * 80)
        
        for line in report:
            print(line)
        
        with open(ANALYSIS_REPORT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\nFull report saved to: {ANALYSIS_REPORT}")
    
    def fix_data(self, data):
        print("\n🔧 FIXING DATA...\n")
        
        PRONOUNS = {'i', 'me', 'my', 'mine', 'he', 'she', 'it', 'they', 'his', 'her', 
                   'its', 'their', 'him', 'them', 'we', 'us', 'our', 'you', 'your'}
        
        confident_characters = {char for char, count in self.all_characters.items() 
                               if count >= 3}
        
        fixed_count = 0
        
        for entry in data:
            original = json.dumps(entry)
            
            entry['characters'] = [
                c for c in entry.get('characters', [])
                if c.lower() not in PRONOUNS and len(c) > 1
            ]
            
            coref = entry.get('coref', {})
            fixed_coref = {}
            for pronoun, referent in coref.items():
                if (referent not in PRONOUNS and 
                    referent != 'Narrator' and 
                    len(referent) > 1 and
                    referent[0].isupper()):
                    fixed_coref[pronoun.lower()] = referent
            entry['coref'] = fixed_coref
            
            text = entry.get('text', '').lower()
            has_first_person = any(p in text.split() for p in ['i', 'my', 'me'])
            
            if has_first_person:
                for fp in ['i', 'my', 'me']:
                    if fp in entry['coref']:
                        candidate = entry['coref'][fp]
                        if candidate in confident_characters:
                            entry['speaker'] = candidate
                            break
                
                if entry.get('speaker') == 'Narrator' and entry['characters']:
                    entry['speaker'] = entry['characters'][0]
            
            speaker = entry.get('speaker', 'Narrator')
            if speaker.lower() in PRONOUNS:
                entry['speaker'] = 'Narrator'
            
            entities = entry.get('entities', [])
            for char in entry['characters']:
                if char not in entities:
                    entities.append(char)
            entry['entities'] = entities
            
            if json.dumps(entry) != original:
                fixed_count += 1
        
        print(f"Fixed {fixed_count} entries")
        return data
    
    def deduplicate(self, data):
        seen = set()
        deduplicated = []
        duplicates_removed = 0
        
        for entry in data:
            idx = entry.get('index', -1)
            if idx not in seen:
                seen.add(idx)
                deduplicated.append(entry)
            else:
                duplicates_removed += 1
        
        if duplicates_removed:
            print(f"Removed {duplicates_removed} duplicate entries")
        
        return deduplicated
    
    def save_corrected(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Corrected data saved to: {filepath}")

def interactive_sample_review(data, n=10):
    
    print("\n" + "="*80)
    print("MANUAL SAMPLE REVIEW")
    print("="*80)
    print("Review random samples to identify quality issues\n")
    
    sample = random.sample(data, min(n, len(data)))
    issues = []
    
    for i, entry in enumerate(sample, 1):
        print(f"\n--- Sample {i}/{len(sample)} ---")
        print(f"Index: {entry.get('index')}")
        print(f"Text: {entry.get('text', '')[:150]}...")
        print(f"Characters: {entry.get('characters', [])}")
        print(f"Speaker: {entry.get('speaker', 'N/A')}")
        print(f"Coref: {entry.get('coref', {})}")
        
        feedback = input("\nIssues? (press Enter if OK, or describe issue): ").strip()
        if feedback:
            issues.append({
                'index': entry.get('index'),
                'issue': feedback,
                'entry': entry
            })
    
    if issues:
        print(f"\nLogged {len(issues)} issues for review")
        with open('manual_review_issues.json', 'w', encoding='utf-8') as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
    else:
        print("\nNo issues found in sample!")


def suggest_improvements(analyzer):
    print("\n" + "="*80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*80)
    
    suggestions = []
    
    rare_chars = [char for char, count in analyzer.all_characters.items() if count == 1]
    if len(rare_chars) > len(analyzer.all_characters) * 0.3:
        suggestions.append(
            f"{len(rare_chars)} characters appear only once - these might be extraction errors.\n"
            f"Consider: Add character name validation against a known character list."
        )
    
    narrator_pct = (analyzer.all_speakers.get('Narrator', 0) / 
                   sum(analyzer.all_speakers.values()) * 100)
    if narrator_pct > 70:
        suggestions.append(
            f"{narrator_pct:.1f}% of sentences have speaker='Narrator' - this seems high.\n"
            f"Consider: Improve first-person pronoun detection for better speaker inference."
        )
    
    invalid_corefs = len(analyzer.errors['invalid_coref'])
    if invalid_corefs > 0:
        suggestions.append(
            f"{invalid_corefs} invalid coreference mappings (pronouns mapped to pronouns).\n"
            f"Consider: Add post-processing to resolve pronoun chains."
        )
    
    entries_with_pronouns = 0
    entries_with_coref = 0
    
    if not suggestions:
        suggestions.append("Data quality looks reasonable! Minor improvements suggested above.")
    
    for suggestion in suggestions:
        print(f"\n{suggestion}")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("DATA QUALITY ANALYZER & CORRECTOR")
    print("="*80)
    
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file '{INPUT_FILE}' not found!")
        return
    
    analyzer = DataQualityAnalyzer()
    
    data = analyzer.load_data(INPUT_FILE)
    if not data:
        return
    
    analyzer.analyze(data)
    
    suggest_improvements(analyzer)
    
    print("\n" + "="*80)
    response = input("Do you want to automatically fix errors? (y/n): ").strip().lower()
    
    if response == 'y':
        fixed_data = analyzer.fix_data(data)
        
        fixed_data = analyzer.deduplicate(fixed_data)
        
        analyzer.save_corrected(fixed_data, OUTPUT_FILE)
        
        print("\nDone! Review the corrected output and quality report.")
    
    response = input("\nDo you want to manually review samples? (y/n): ").strip().lower()
    if response == 'y':
        n = input("How many samples? (default 10): ").strip()
        n = int(n) if n.isdigit() else 10
        interactive_sample_review(data, n)


if __name__ == "__main__":
    main()
