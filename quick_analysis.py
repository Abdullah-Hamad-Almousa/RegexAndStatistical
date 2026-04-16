import pandas as pd
import re

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# Check Medical History
print("=== MEDICAL HISTORY ANALYSIS ===")
med_fn = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Medical History'] == 1:
        text = context_df.iloc[idx]['text']
        has_no = any(p in text.lower() for p in ['لا يوجد', 'لا تاريخ', 'لا امراض', 'لا ملاحظات', 'لا شكاوى', 'لا مشاكل'])
        has_pos = any(p in text.lower() for p in ['سجل طبي', 'فحص', 'علاج', 'حساسية', 'ضغط', 'سكري', 'صحة'])
        has_status = 'سليم' in text or 'جيد' in text

        if not (has_no or has_pos or has_status):
            med_fn += 1
            print(f"  Row {idx+2}: Medical=1 but no clear patterns")

print(f"\nTotal missed medical (FN): {med_fn}")

# Check FamilyName
print("\n=== FAMILY NAME ANALYSIS ===")
fam_fn = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FamilyName'] == 1:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])
        filtered = [n for n in arabic if n not in name_exclusion]

        if len(filtered) < 4:
            fam_fn += 1
            print(f"  Row {idx+2}: Only {len(arabic)} arabic, {len(filtered)} filtered")
        else:
            potential = filtered[3]
            if not potential.startswith('ال'):
                fam_fn += 1
                # Just print count, not the Arabic
                print(f"  Row {idx+2}: Family doesn't start with ال (len={len(potential)})")

print(f"\nTotal missed family (FN): {fam_fn}")

# Check FirstName FP
print("\n=== FIRSTNAME FALSE POSITIVES ===")
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

print(f"Total samples with Arabic words but FirstName=0: {fp_count}")

# Check Job FP
print("\n=== JOB TITLE FALSE POSITIVES ===")
job_fp = 0
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Job Title'] == 0:
        text = context_df.iloc[idx]['text']
        triggers = []
        single_jobs = ['مهندس', 'محاسب', 'طبيب', 'مدير', 'موظف', 'كاتب', 'صحفي', 'معلم', 'دكتور', 'مستشار', 'مراقب', 'مصور', 'مصمم', 'فني', 'عالم', 'باحث', 'اكاديمي', 'طيار', 'قاضي', 'محامي', 'موسيقي']
        for j in single_jobs:
            if j in text:
                triggers.append(j)
        if triggers:
            job_fp += 1

print(f"Total job false positives: {job_fp}")

print("\n=== SUMMARY OF KEY INSIGHTS ===")
print("1. Medical missed: Need to find what patterns indicate medical=1 without negation/positive words")
print("2. FamilyName missed: Cases where position 4 doesn't start with ال")
print("3. FirstName FP: We flag any Arabic word as name - need better filtering")
print("4. Job FP: Single words like 'طبيب' appear in non-job contexts")