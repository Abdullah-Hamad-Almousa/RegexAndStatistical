import pandas as pd
import re

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# Phone patterns missed - collect all unique patterns
print("=== PHONE PATTERNS MISSED ===")
missed_phones = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Phone Number'] == 1:
        text = context_df.iloc[idx]['text']
        potential_phones = re.findall(r'\+[\d\s.\-]{8,}', text)
        for p in potential_phones:
            if p not in missed_phones:
                missed_phones.append(p)

print(f"Found {len(missed_phones)} unique missed phone patterns:")
for p in missed_phones[:20]:
    print(f"  '{p}'")

# Create better phone regex
print("\n=== SUGGESTED PHONE PATTERN ===")
# Pattern: + followed by country code (1-3 digits) then varying formats
# Formats seen:
# +251 11 123 4567
# +370 5 123 4567
# +961 1 123 456 (only 6 digits at end!)
# +500 12345 (very short)
# +290 12345 (very short)
# +687 12.34.56 (dots)
# +386 1 123 45 67 (5 parts)
# +358 9 1234567 (no space after country)
# +45 12 34 56 78 (5 parts)
# +886 2 1234 5678 (4 digit final group)

# New pattern to catch all of these:
new_pattern = r'\+\d{1,3}[\s.\-]?(?:\d{2,4}[\s.\-]?)+'

print(f"New pattern: {new_pattern}")
print("\nTesting new pattern:")
test_count = 0
match_count = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Phone Number'] == 1:
        test_count += 1
        text = context_df.iloc[idx]['text']
        matches = re.findall(new_pattern, text)
        if matches:
            match_count += 1
        else:
            print(f"  Still missed row {idx+2}")

print(f"Matches: {match_count}/{test_count}")

# Now analyze family name issue
print("\n\n=== FAMILY NAME ISSUE ===")
print("Looking at why we miss FamilyName:")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FamilyName'] == 1:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع', 'لا', 'ليس', 'يوجد', 'اكتف', 'غير', 'كبير', 'صغير', 'جيد', 'سيء', 'ممتاز', 'مقبول', 'اول', 'ثاني', 'ثالث', 'رابع', 'خامس', 'لا يوجد', 'ليس هناك', 'ليس فيه'])
        filtered = [n for n in arabic if n not in name_exclusion]
        if len(filtered) < 4:
            print(f"  Row {idx+2}: {len(arabic)} total arabic, {len(filtered)} after filter")
            print(f"    All: {arabic}")
            print(f"    Filtered: {filtered}")
            break

# Analyze FirstName false positives
print("\n\n=== FIRSTNAME FALSE POSITIVES ===")
print("Samples where we flagged but shouldn't:")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 0:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic_names = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        if len(arabic_names) >= 1:
            print(f"  Row {idx+2}: {len(arabic_names)} arabic in brackets: {arabic_names[:4]}")