import pandas as pd
import re

# Set UTF-8 encoding for output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# ============ PHONE FIX ============
# Collect ALL unique phone formats that should be detected
print("=== PHONE NUMBER IMPROVEMENT ===\n")

# New comprehensive phone pattern
new_phone_pattern = r'\+\d{1,3}[\s.\-]*(?:\d{1,4}[\s.\-]*)+'

# Test on all missed phones
missed_phones = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Phone Number'] == 1:
        text = context_df.iloc[idx]['text']
        potential = re.findall(new_phone_pattern, text)
        if not potential:
            # Try to find what looks like a phone
            all_plus = re.findall(r'\+[0-9\s.\-]+', text)
            for p in all_plus:
                if len(p) > 8:  # Minimum length for phone
                    missed_phones.append((idx+2, p))

print(f"Still missing {len(missed_phones)} phones after new pattern:")
for row, phone in missed_phones[:10]:
    print(f"  Row {row}: '{phone}'")

# Even better pattern
best_phone_pattern = r'\+\d{1,3}[\s.\-]*[\d\s.\-]+'

print(f"\n\nBest pattern: {best_phone_pattern}")
match_count = sum(1 for idx in range(len(context_df)) if val_df.iloc[idx]['Phone Number'] == 1 and re.search(best_phone_pattern, context_df.iloc[idx]['text']))
total_phones = sum(1 for idx in range(len(context_df)) if val_df.iloc[idx]['Phone Number'] == 1)
print(f"Matches: {match_count}/{total_phones} = {match_count/total_phones*100:.1f}%")

# ============ FAMILY NAME FIX ============
print("\n\n=== FAMILY NAME IMPROVEMENT ===\n")

# The issue: We need 4 names but only have 3 after filtering
# Solution: Relax the exclusion list OR use position-based heuristics

# Let's analyze what happens in the problematic rows
print("Rows where FamilyName=1 but we detect < 4 names:")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FamilyName'] == 1:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]

        # Current exclusion list (too aggressive)
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع', 'لا', 'ليس', 'يوجد', 'اكتف', 'غير', 'كبير', 'صغير', 'جيد', 'سيء', 'ممتاز', 'مقبول', 'اول', 'ثاني', 'ثالث', 'رابع', 'خامس', 'لا يوجد', 'ليس هناك', 'ليس فيه'])
        filtered = [n for n in arabic if n not in name_exclusion]

        if len(filtered) < 4 and len(arabic) >= 4:
            print(f"  Row {idx+2}: {len(arabic)} arabic, {len(filtered)} after filter")
            # Which words are being excluded?
            excluded = [n for n in arabic if n in name_exclusion]
            if excluded:
                print(f"    Excluded: {excluded}")