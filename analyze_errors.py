import pandas as pd
import re

context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')

# Analyze Phone Number errors
print("=== PHONE NUMBER ANALYSIS ===")
print("\nSamples where Phone=1 but we detected=0:")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Phone Number'] == 1:
        text = context_df.iloc[idx]['text']
        phone_pattern = r'\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}|\+\d{1,3}\s\d{2,3}\s\d{3,4}|\+\d{1,3}\s\d{3,4}\s\d{3,4}'
        matches = re.findall(phone_pattern, text)
        if not matches:
            # Find all numbers that look like phones
            potential_phones = re.findall(r'\+[\d\s.-]{8,}', text)
            print(f"  Row {idx+2}: {potential_phones[:3]}")

print("\n\n=== FIRSTNAME ERROR ANALYSIS ===")
print("Samples where FirstName=1 but we flagged incorrectly:")
false_positives_first = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 0:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        if len(arabic) >= 1:
            # We flagged but shouldn't have
            false_positives_first.append((idx+2, arabic[:4]))

print(f"  Found {len(false_positives_first)} samples with Arabic words but FirstName=0")
print(f"  Examples:")
for idx, names in false_positives_first[:5]:
    print(f"    Row {idx}: {names}")

print("\n\n=== FAMILY NAME ERROR ANALYSIS ===")
print("Samples where FamilyName=1 but we missed (FN=29):")
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FamilyName'] == 1:
        text = context_df.iloc[idx]['text']
        brackets = re.findall(r'\[(.*?)\]', text)
        arabic = [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]
        name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع', 'لا', 'ليس', 'يوجد', 'اكتف', 'غير', 'كبير', 'صغير', 'جيد', 'سيء', 'ممتاز', 'مقبول', 'اول', 'ثاني', 'ثالث', 'رابع', 'خامس', 'لا يوجد', 'ليس هناك', 'ليس فيه'])
        filtered = [n for n in arabic if n not in name_exclusion]
        if len(filtered) < 4:
            print(f"  Row {idx+2}: Only {len(filtered)} filtered names, brackets: {arabic[:6]}")
            print(f"    Filtered: {filtered}")

print("\n\n=== MEDICAL HISTORY ANALYSIS ===")
print("Samples where Medical=1 but we missed:")
missed_medical = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Medical History'] == 1:
        text = context_df.iloc[idx]['text']
        medical_keywords = ['سجل طبي', 'صحة', 'سليم', 'لا تاريخ', 'لا يوجد', 'حالة صحية', 'فحص', 'حساسية', 'مرضي', 'لا ملاحظات', 'لا امراض', 'لا شكاوى']
        found = any(kw in text.lower() for kw in medical_keywords) or 'سليم' in text or 'لا يوجد سجل' in text or 'لا تاريخ' in text
        if not found:
            missed_medical.append((idx+2, text[-200:]))

print(f"  Missed {len(missed_medical)} medical samples")
for idx, text in missed_medical[:3]:
    print(f"  Row {idx}: ...{text}")

print("\n\n=== JOB TITLE ANALYSIS ===")
print("Samples where JobTitle=1 but we missed:")
missed_job = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['Job Title'] == 1:
        text = context_df.iloc[idx]['text']
        job_titles = ['مهندس', 'محاسب', 'طبيب', 'مدير', 'موظف', 'كاتب', 'صحفي', 'مهندس معماري', 'معلم', 'دكتور', 'مستشار', 'مراقب', 'مصور', 'مصمم', 'فني', 'عالم', 'باحث', 'اكاديمي', 'طيار', 'قاضي', 'محامي', 'موسيقي', 'جغرافي', 'مهندس مدني', 'معالج', 'ممرض', 'دبلوماسي', 'اقتصادي', 'محلل مالي', 'مصمم جرافيك', 'مهندس برمجيات', 'أخصائي', 'مدقق', 'مخطط', 'مرافق', 'مندوب', 'تاجر', 'مقاول', 'مهني', 'سائق', 'حارس', 'امني', 'مشرف', 'قائد', 'سياسي', 'محلل', 'صحفي', 'محرر', 'ناشر', 'مصمم', 'مهندس طيران', 'مراقب جوي', 'مهندس شبكات', 'مهندس انظمة', 'مهندس امن']
        english_jobs = re.findall(r'\b(Engineer|Manager|Doctor|Accountant|Biologist|Teacher|Carpenter|Electrician|Painter|Scientist|Architect|Consultant|Specialist|Analyst|Designer|Developer|Writer|Artist|Musician|Diplomat|Lawyer|Adviser|Coordinator|Supervisor|Executive|Director)\b', text, re.I)
        if not any(t in text for t in job_titles) and not english_jobs:
            missed_job.append((idx+2, text[-200:]))

print(f"  Missed {len(missed_job)} job samples")
for idx, text in missed_job[:5]:
    print(f"  Row {idx}: ...{text}")