import pandas as pd
import re

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# Analyze Medical History failures
print("=== MEDICAL HISTORY ANALYSIS ===")
print("\nWhere Medical=1 but we missed (FN):")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Medical History'] == 1:
        text = context_df.iloc[idx]['text']
        # Check what patterns exist
        has_negation = any(p in text.lower() for p in ['لا يوجد', 'لا تاريخ', 'لا امراض', 'لا ملاحظات'])
        has_positive = any(p in text.lower() for p in ['سجل طبي', 'فحص', 'علاج', 'حساسية', 'ضغط', 'سكري'])
        has_status = 'سليم' in text or 'جيد' in text or 'حالة' in text

        negation_patterns = [p for p in ['لا يوجد', 'لا تاريخ', 'لا امراض', 'لا ملاحظات', 'لا شكاوى', 'لا مشاكل'] if p in text.lower()]
        positive_patterns = [p for p in ['سجل طبي', 'فحص', 'علاج', 'حساسية', 'ضغط', 'سكري', 'صحة'] if p in text.lower()]

        if not (has_negation or has_positive or has_status):
            print(f"  Row {idx+2}: No obvious medical patterns")
            print(f"    Negation: {negation_patterns}")
            print(f"    Positive: {positive_patterns}")
            print(f"    Status: {'سليم' in text}, {'جيد' in text}, {'حالة' in text}")
            print(f"    Text: {text[-150:]}")

print("\n\n=== FAMILY NAME ANALYSIS ===")
print("\nWhere FamilyName=1 but we missed (FN):")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FamilyName'] == 1:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])
        filtered = [n for n in arabic if n not in name_exclusion]

        if len(filtered) < 4:
            print(f"  Row {idx+2}: Only {len(filtered)} filtered names out of {len(arabic)} arabic")
            print(f"    All brackets: {brackets}")
            print(f"    Filtered: {filtered}")
        else:
            potential = filtered[3]
            if not potential.startswith('ال'):
                print(f"  Row {idx+2}: Family '{potential}' doesn't start with ال")
                print(f"    All 4 names: {filtered}")

print("\n\n=== FIRSTNAME FALSE POSITIVES ===")
print("Where FirstName=0 but we detected (FP):")
count = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 0:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])
        filtered = [n for n in arabic if n not in name_exclusion]

        if len(filtered) >= 1:
            count += 1
            if count <= 5:
                print(f"  Row {idx+2}: Detected name(s): {filtered[:4]}")
                print(f"    All brackets: {brackets[:6]}")

print(f"\nTotal samples with Arabic words but FirstName=0: {count}")

print("\n\n=== JOB TITLE FALSE POSITIVES ===")
print("Where JobTitle=0 but we detected (FP):")
count = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Job Title'] == 0:
        text = context_df.iloc[idx]['text']
        # Check what triggered it
        triggers = []
        single_jobs = ['مهندس', 'محاسب', 'طبيب', 'مدير', 'موظف', 'كاتب', 'صحفي', 'معلم', 'دكتور', 'مستشار', 'مراقب', 'مصور', 'مصمم', 'فني', 'عالم', 'باحث', 'اكاديمي', 'طيار', 'قاضي', 'محامي', 'موسيقي', 'جغرافي', 'معالج', 'ممرض', 'دبلوماسي', 'اقتصادي']
        for j in single_jobs:
            if j in text:
                triggers.append(j)

        if triggers:
            count += 1
            if count <= 5:
                print(f"  Row {idx+2}: Triggered by: {triggers}")
                print(f"    Context: {text[-100:]}")

print(f"\nTotal job false positives: {count}")