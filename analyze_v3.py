import pandas as pd
import re
from collections import Counter

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# Get the 17 missed medical rows
print("=== MEDICAL MISSED ANALYSIS ===")
missed_ids = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Medical History'] == 1:
        text = context_df.iloc[idx]['text']
        has_no = any(p in text.lower() for p in ['لا يوجد', 'لا تاريخ', 'لا امراض', 'لا ملاحظات', 'لا شكاوى', 'لا مشاكل'])
        has_pos = any(p in text.lower() for p in ['سجل طبي', 'فحص', 'علاج', 'حساسية', 'ضغط', 'سكري', 'صحة'])
        has_status = 'سليم' in text or 'جيد' in text

        if not (has_no or has_pos or has_status):
            missed_ids.append(idx)

print(f"Missed {len(missed_ids)} medical cases")

# Count brackets in these cases
all_brackets = []
for idx in missed_ids:
    text = context_df.iloc[idx]['text']
    brackets = re.findall(r'\[(.*?)\]', text)
    all_brackets.extend(brackets)

counter = Counter(all_brackets)
print(f"Total bracket items: {len(all_brackets)}")
print(f"Unique bracket items: {len(counter)}")
print("\nTop 20 bracket contents (by frequency):")
for i, (word, count) in enumerate(counter.most_common(20)):
    # Show first few chars as hex to avoid encoding issues
    print(f"  [{i+1}] {count}x: {'/'.join(hex(ord(c)) for c in word[:3])}...")

# FamilyName analysis
print("\n\n=== FAMILY NAME ANALYSIS ===")
fam_missed = 0
no_al = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FamilyName'] == 1:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])
        filtered = [n for n in arabic if n not in name_exclusion]

        if len(filtered) < 4:
            fam_missed += 1
        else:
            potential = filtered[3]
            if not potential.startswith('ال'):
                no_al += 1

print(f"FamilyName missed (< 4 names): {fam_missed}")
print(f"FamilyName missed (doesn't start with ال): {no_al}")

# FirstName FP analysis
print("\n\n=== FIRSTNAME FP ANALYSIS ===")
fp_count = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 0:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])
        filtered = [n for n in arabic if n not in name_exclusion]
        if len(filtered) >= 1:
            fp_count += 1

print(f"Samples with Arabic words but FirstName=0: {fp_count}")

# Check if Medical=1 cases have bracket patterns
print("\n\n=== KEY INSIGHT ===")
# Check if the 17 missed medical cases have any bracket words that could indicate medical
med_brackets = []
for idx in missed_ids:
    text = context_df.iloc[idx]['text']
    brackets = re.findall(r'\[(.*?)\]', text)
    med_brackets.extend([b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)])

med_counter = Counter(med_brackets)
print("Top Arabic words in missed medical brackets:")
for word, count in med_counter.most_common(15):
    print(f"  {count}x: len={len(word)}")

# Check if 'Healthy', 'Normal' etc are present
normal_count = 0
for idx in missed_ids:
    text = context_df.iloc[idx]['text']
    if 'Healthy' in text or 'Normal' in text or 'Good' in text or 'Fine' in text:
        normal_count += 1

print(f"\nMissed cases with English health words: {normal_count}")