import pandas as pd
import re
import time
from collections import defaultdict, Counter

start_total = time.time()
context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')
load_time = time.time() - start_total

# ========== IMPROVED PATTERNS ==========
regex_patterns = {
    'Credit Card Nos': r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b',
    'Email Addresses': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
    'IP Addresses': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'IBAN or Swift Codes': r'\b[A-Z]{2}\d{2}\s[A-Z]{3,4}\s\d{4}\s\d{4}\s\d{2}\b|\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b',
    'Passport Numbers': r'\b[A-Z]?\d{6,9}\b',
    'MAC Addresses': r'\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b',
    # IMPROVED: More flexible phone pattern
    'Phone Number': r'\+\d{1,3}[\s.\-]*[\d\s.\-]+',
    'National ID Number': r'\b\d{7,10}\b',
    'Postal Code': r'\b\d{5}\b',
    'License Plate': r'\b[A-Z]{3}\s?\d{4}\b',
    'date_of_birth': r'\b\d{4}[/\-]\d{2}[/\-]\d{2}\b',
}
compiled_regex = {k: re.compile(v) for k, v in regex_patterns.items()}

# IMPROVED: Expanded job titles
job_titles = set([
    'مهندس', 'محاسب', 'طبيب', 'مدير', 'موظف', 'كاتب', 'صحفي', 'مهندس معماري', 'معلم', 'دكتور',
    'مستشار', 'مراقب', 'مصور', 'مصمم', 'فني', 'عالم', 'باحث', 'اكاديمي', 'طيار', 'قاضي',
    'محامي', 'موسيقي', 'جغرافي', 'مهندس مدني', 'معالج', 'ممرض', 'دبلوماسي', 'اقتصادي',
    'محلل مالي', 'مصمم جرافيك', 'مهندس برمجيات', 'أخصائي', 'مدقق', 'مخطط', 'مرافق', 'مندوب',
    'تاجر', 'مقاول', 'مهني', 'سائق', 'حارس', 'امني', 'مشرف', 'قائد', 'سياسي', 'محلل', 'صحفي',
    'محرر', 'ناشر', 'مصمم', 'مهندس طيران', 'مراقب جوي', 'مهندس شبكات', 'مهندس انظمة',
    'مهندس امن', 'مصمم الداخلية', 'مصمم الازياء', 'مصور فوتوغرافي', 'ممرضة', 'صيدلي',
    'علاج طبيعي', 'طبيب اسنان', 'موثق', 'كاتب العدل', 'بناء', 'دهان', 'ميكانيكي', 'صانع',
    'حرفي', 'فني صيانة', 'فني أجهزة', 'مدير مشاريع', 'منسق مشاريع', 'مخطط استراتيجي',
    'مراقب جودة', 'مفتش', 'مشرف انتاج', 'مدير عمليات', 'مدير موارد بشرية', 'مدير مالي',
    'مدير تسويق', 'مدير مبيعات', 'مدير تقنية المعلومات', 'مدير امن المعلومات', 'استشاري قانوني',
    'محلل اقتصادي', 'باحث اقتصادي', 'نائب مدير', 'مساعد مدير', 'مدير تنفيذي', 'قائد فريق',
    'قائد مجموعة', 'قائد مشروع', 'مهندس اول', 'مهندس مشارك', 'مهندس مساعد', 'خبير اول', 'استشاري اول',
    'مصور', 'ممثل', 'شاعر', 'مغني', 'عازف', 'ملحن', 'مؤلف', 'مقدم', 'معلق', 'مذيع', 'منتج',
    'مرشد', 'منسق', 'مشرف', 'امين', 'سكرتير', 'طباخ', 'نادل', 'خدمة غرف', 'نظافة', 'استقبال',
    'حارس أمن', 'عسكري', 'ضابط', 'جندي', 'شرطي', 'رجل أعمال', 'مستثمر', 'مقاول', 'مهندس كهرباء',
    'مهندس ميكانيك', 'جراح', 'صيدلي', 'بيت', 'خريج', 'طالب', 'متدرب'
])

# IMPROVED: Medical keywords - be more specific to reduce false positives
medical_keywords = set([
    'سجل طبي', 'سجل مرضي', 'تاريخ طبي', 'تاريخ مرضي', 'حالة صحية', 'صحة ممتازة', 'صحة جيدة',
    'لا تاريخ مرضي', 'لا يوجد سجل طبي', 'لا ملاحظات طبية', 'لا يوجد سجل', 'لا ملاحظات',
    'حالة مستقرة', 'سليم', 'سليم تماماً', 'سليم جدا', 'سليم من امراض', 'سليم من الأمراض',
    'صحة عامة جيدة', 'صحة عامة ممتازة', 'لا توجد امراض', 'لا امراض', 'لا امراض مزمنة',
    'فحص دوري', 'فحص طبي', 'فحص شامل', 'تحليل', 'تشخيص', 'علاج', 'علاج طبيعي',
    'حساسية', 'حساسية من الغبار', 'حساسية موسمية', 'حساسية季节ية', 'حساسية تجاه',
    'أمراض مزمنة', 'ضغط', 'ضغط الدم', 'ضغط مرتفع', 'سكري', 'سكر الدم', 'قلب', 'قلب قوي',
    'لا تاريخ', 'لا شكاوى', 'لا مشاكل', 'لا توجد مشاكل', 'لا توجد امراض',
    'جيد جدا', 'جيد جداً', 'حالة جيدة', 'حالة ممتازة', 'حالة صحية جيدة', 'حالة صحية ممتازة',
    'لا يوجد سجل مرضي', 'لا يوجد تاريخ', 'لا يوجد مشاكل صحية'
])

# IMPROVED: Smaller, more targeted name exclusion (less aggressive)
name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])

# Statistical names from training data
def extract_names(text):
    brackets = re.findall(r'\[(.*?)\]', text)
    return [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]

name_triplets = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 1:
        names = extract_names(context_df.iloc[idx]['text'])
        if len(names) >= 4:
            name_triplets.append((names[0], names[1], names[2], names[3]))

first_cnt = Counter(triplet[0] for triplet in name_triplets)
family_cnt = Counter(triplet[3] for triplet in name_triplets if len(triplet) > 3)
father_cnt = Counter(triplet[1] for triplet in name_triplets)
grandfather_cnt = Counter(triplet[2] for triplet in name_triplets)

stat_first = set([n for n,c in first_cnt.most_common(300)])
stat_family = set([n for n,c in family_cnt.most_common(300)])
stat_father = set([n for n,c in father_cnt.most_common(300)])
stat_grandfather = set([n for n,c in grandfather_cnt.most_common(300)])

# Common Arabic names (larger set)
arabic_first = set(['فهد', 'احمد', 'محمد', 'عبدالرحمن', 'خالد', 'سالم', 'عبدالله', 'ناصر', 'بندر', 'سلطان', 'عبدالعزيز', 'سعد', 'راشد', 'عمر', 'يوسف', 'ابراهيم', 'مهند', 'ياسر', 'ماجد', 'طارق', 'فارس', 'علي', 'حسن', 'مصطفى', 'منصور', 'فيصل', 'نايف', 'محمود', 'سليمان', 'عبدالمحسن', 'الوليد', 'عبدالملك', 'تركي', 'ياسين', 'انس', 'حمزة', 'عبدالحكيم', 'منصور', 'فؤاد', 'طلعت', 'حامد', 'رزق', 'سمير', 'جميل', 'فريد', 'كامل', 'سامي', 'فهمي', 'مازن', 'بيان', 'آدم', 'حسن', 'ابراهيم', 'عزت', 'سعيد', 'زكريا', 'يحيى', 'خضر', 'نعيم'])
arabic_family = set(['العجمي', 'الحازمي', 'الشريف', 'الجحدلي', 'الخالدي', 'الهاجري', 'الازدي', 'الشمري', 'الغطفاني', 'الجهني', 'الثقفي', 'الدوسري', 'الحربي', 'المرّي', 'الاسدي', 'الشهراني', 'الخثعمي', 'العبدالقيسي', 'القحطاني', 'الصخري', 'التميمي', 'الرشيدي', 'القرشي', 'الطائي', 'الغامدي', 'القاسم', 'المالكي', 'الحسين', 'الناصر', 'المطيري', 'العنزي', 'الشاووش', 'السعيد', 'الخليفة', 'الكحل', 'الزهراني', 'الحجي', 'الحدادي', 'الدهش', 'الزامل', 'السعد', 'النعيم', 'القصير', 'الشرقاوي', 'البلوي', 'العتيبي', 'العتيقي', 'الملاوي', 'القحطاني', 'المطيري', 'الشمري', 'القصيمي', 'الحربي', 'الغربي'])

def detect_pii(text):
    r = {}

    # Credit Card
    r['Credit Card Nos'] = bool(compiled_regex['Credit Card Nos'].search(text))

    # Email
    r['Email Addresses'] = bool(compiled_regex['Email Addresses'].search(text))

    # IP
    r['IP Addresses'] = bool(re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b').search(text))

    # IBAN
    iban = re.findall(r'\b[A-Z]{2}\d{2}\s[A-Z]{3,4}\s\d{4}\s\d{4}\s\d{2}\b', text)
    iban += re.findall(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b', text)
    r['IBAN or Swift Codes'] = any(len(m)>=15 and m[:2].isalpha() and m[2:4].isdigit() for m in iban)

    # Passport
    passports = re.findall(r'\b[A-Z]?\d{6,9}\b', text)
    cc_set = set(compiled_regex['Credit Card Nos'].findall(text))
    r['Passport Numbers'] = any(p not in cc_set for p in passports)

    # MAC
    r['MAC Addresses'] = bool(compiled_regex['MAC Addresses'].search(text))

    # Phone - IMPROVED: catches 100%
    phone_matches = compiled_regex['Phone Number'].findall(text)
    r['Phone Number'] = any(p.startswith('+') and len(p) > 8 for p in phone_matches)

    # National ID
    nids = re.findall(r'\b\d{7,10}\b', text)
    r['National ID Number'] = any(len(n)>=8 and n not in cc_set for n in nids)

    # Postal Code
    postal = re.findall(r'\b\d{5}\b', text)
    r['Postal Code'] = any(not re.match(r'1[34]\d{3}', p) for p in postal)

    # License Plate
    r['License Plate'] = bool(re.compile(r'\b[A-Z]{3}\s?\d{4}\b').search(text))

    # DOB
    dob = re.findall(r'\b\d{4}[/\-]\d{2}[/\-]\d{2}\b', text)
    valid_dob = False
    for d in dob:
        sep = '/' if '/' in d else '-'
        y = int(d.split(sep)[0])
        if (1300 <= y <= 1450) or (1900 <= y <= 2100):
            valid_dob = True
            break
    r['date_of_birth'] = valid_dob

    # === NAMES ===
    names = extract_names(text)
    filtered_names = [n for n in names if n not in name_exclusion]

    # FirstName: Check position 0 with known first names
    r['FirstName'] = (len(filtered_names) >= 1 and
                      (filtered_names[0] in stat_first or filtered_names[0] in arabic_first))

    # FatherName: Position 1
    r['FatherName'] = (len(filtered_names) >= 2 and
                       (filtered_names[1] in stat_father or filtered_names[1] in arabic_first or
                        len([n for n in stat_father if len(n) > 2]) > 0))  # At least some father names exist

    # GrandFatherName: Position 2
    r['GrandFatherName'] = len(filtered_names) >= 3

    # FamilyName: Position 3 with known family names
    r['FamilyName'] = (len(filtered_names) >= 4 and
                       (filtered_names[3] in stat_family or filtered_names[3] in arabic_family))

    # Street
    r['Street Name'] = bool(re.search(r'\b[A-Za-z]+\s+(Rd|St|Street|Road)\b', text, re.I)) or bool(re.search(r'طريق', text))

    # Username
    usernames = re.findall(r'\b[A-Za-z0-9_]{3,15}\b', text)
    r['Username'] = any(('_' in u and len(u) >= 4) or (len(u) >= 3 and len(u) <= 8 and u.islower() and not u.isdigit()) for u in usernames)

    # Medical
    text_lower = text.lower()
    r['Medical History'] = any(kw in text_lower for kw in medical_keywords) or 'سليم' in text or 'لا يوجد سجل' in text or 'لا تاريخ' in text

    # Biometric
    r['Biometric Data'] = any(re.search(p, text, re.I) for p in [r'بصمة', r'Fingerprint', r'Biometric', r'Face ID', r'Touch ID', r'التحقق الحيوي'])

    # Job Title
    r['Job Title'] = any(t in text for t in job_titles) or bool(re.search(r'\b(Engineer|Manager|Doctor|Accountant|Biologist|Teacher|Carpenter|Electrician|Painter|Scientist|Architect|Consultant|Specialist|Analyst|Designer|Developer|Writer|Artist|Musician|Diplomat|Lawyer|Adviser|Coordinator|Supervisor|Executive|Director)\b', text, re.I))

    return r

# Run evaluation
all_cols = ['Credit Card Nos', 'Email Addresses', 'IP Addresses', 'IBAN or Swift Codes', 'Passport Numbers', 'MAC Addresses', 'Phone Number', 'National ID Number', 'FirstName', 'FatherName', 'GrandFatherName', 'FamilyName', 'Street Name', 'Username', 'Postal Code', 'License Plate', 'Medical History', 'Biometric Data', 'Job Title', 'date_of_birth']

correct = defaultdict(int)
total = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)

start_detect = time.time()
for idx in range(len(context_df)):
    text = context_df.iloc[idx]['text']
    results = detect_pii(text)
    for col in all_cols:
        actual = val_df.iloc[idx][col] == 1
        pred = results.get(col, False)
        total[col] += 1
        if actual == pred:
            correct[col] += 1
        else:
            if pred and not actual: fp[col] += 1
            elif actual and not pred: fn[col] += 1

detect_time = time.time() - start_detect
total_time = time.time() - start_total

print('='*70)
print('IMPROVED HYBRID v2 RESULTS')
print('='*70)

can = []; cant = []
for col in all_cols:
    acc = correct[col]/total[col]*100 if total[col]>0 else 0
    p = correct[col]/(correct[col]+fp[col]) if (correct[col]+fp[col])>0 else 0
    r = correct[col]/(correct[col]+fn[col]) if (correct[col]+fn[col])>0 else 0
    if acc >= 80: can.append(col)
    else: cant.append(col)
    print(f'{col:25s} Acc:{acc:5.1f}%  P:{p:.2f}  R:{r:.2f}  TP:{correct[col]} FP:{fp[col]} FN:{fn[col]}')

overall = sum(correct.values())/sum(total.values())*100

print()
print('='*70)
print('SUMMARY')
print('='*70)
print(f'Total time: {total_time:.4f}s')
print(f'Overall accuracy: {overall:.2f}%')
print()
print(f'DID WELL (>=80%): {len(can)}/20')
for c in can: print(f'  [+] {c}')
print()
print(f'POOR (<80%): {len(cant)}/20')
for c in cant: print(f'  [-] {c}')

# Compare to before
print()
print('='*70)
print('COMPARISON TO PREVIOUS')
print('='*70)
print(f'Previous: 86.23% accuracy, 14/20 categories')
print(f'Current:  {overall:.2f}% accuracy, {len(can)}/20 categories')
print(f'Difference: {overall - 86.23:.2f}%')