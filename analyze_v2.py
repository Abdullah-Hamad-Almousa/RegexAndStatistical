import pandas as pd
import re

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# Get the 17 missed medical rows and show what they have
print("=== THE 17 MISSED MEDICAL CASES ===")
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

# Show what common words appear in these missed cases
from collections import Counter
all_brackets = []
for idx in missed_ids:
    text = context_df.iloc[idx]['text']
    brackets = re.findall(r'\[(.*?)\]', text)
    for b in brackets:
        if re.match(r'^[\u0600-\u06FF]+$', b):
            all_brackets.append(b)

counter = Counter(all_brackets)
print("\nMost common Arabic words in missed medical cases:")
for word, count in counter.most_common(20):
    print(f"  {word}: {count}")

# Check FamilyName missed cases
print("\n\n=== FAMILY NAME MISSED ANALYSIS ===")
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

print(f"FamilyName missed because < 4 names: {fam_missed}")
print(f"FamilyName missed because doesn't start with ال: {no_al}")

# Check FirstName false positives
print("\n\n=== FIRSTNAME FALSE POSITIVE ANALYSIS ===")
# Find what Arabic words are causing FPs
fp_words = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 0:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])
        filtered = [n for n in arabic if n not in name_exclusion]
        for f in filtered[:4]:
            fp_words.append(f)

fp_counter = Counter(fp_words)
print("Words causing FirstName false positives:")
for word, count in fp_counter.most_common(20):
    print(f"  {word}: {count}")

print("\n\n=== SOLUTION DIRECTION ===")
print("1. Medical: 17 cases with no obvious keywords - maybe 'Healthy', 'Normal' pattern?")
print("2. FamilyName: Need to handle cases where 4th name doesn't start with ال")
print("3. FirstName: Need to exclude more non-name words")