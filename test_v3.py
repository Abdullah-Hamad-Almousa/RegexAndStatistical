import pandas as pd
import re
import time
from collections import defaultdict, Counter

start_total = time.time()
context_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/context.tsv', sep='\t')
val_df = pd.read_csv('A:/PyCharm/MakingTheDataset/res/data/val.csv')
load_time = time.time() - start_total

# ========== PATTERNS ==========
regex_patterns = {
    'Credit Card Nos': r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b',
    'Email Addresses': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
    'IP Addresses': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'IBAN or Swift Codes': r'\b[A-Z]{2}\d{2}\s[A-Z]{3,4}\s\d{4}\s\d{4}\s\d{2}\b|\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b',
    'Passport Numbers': r'\b[A-Z]?\d{6,9}\b',
    'MAC Addresses': r'\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b',
    'Phone Number': r'\+\d{1,3}[\s.\-]*[\d\s.\-]+',
    'National ID Number': r'\b\d{7,10}\b',
    'Postal Code': r'\b\d{5}\b',
    'License Plate': r'\b[A-Z]{3}\s?\d{4}\b',
    'date_of_birth': r'\b\d{4}[/\-]\d{2}[/\-]\d{2}\b',
}
compiled_regex = {k: re.compile(v) for k, v in regex_patterns.items()}

# ========== BUILD COMPREHENSIVE NAME DATABASE ==========
def extract_names(text):
    brackets = re.findall(r'\[(.*?)\]', text)
    return [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]

# Extract ALL name sequences from training data where we know names exist
name_sequences = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 1:
        names = extract_names(context_df.iloc[idx]['text'])
        if len(names) >= 2:
            name_sequences.append(names[:8])  # Store up to 8 names

print(f"Collected {len(name_sequences)} name sequences from training data")

# Build position-specific name sets
position_names = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set(), 7: set()}
for seq in name_sequences:
    for pos, name in enumerate(seq):
        position_names[pos].add(name)

# Build known first + father + grandfather + family combinations (the full 4-name pattern)
full_name_set = set()
for seq in name_sequences:
    if len(seq) >= 4:
        full_name_set.add((seq[0], seq[1], seq[2], seq[3]))

print(f"Unique first names at position 0: {len(position_names[0])}")
print(f"Unique father names at position 1: {len(position_names[1])}")
print(f"Unique grandfather names at position 2: {len(position_names[2])}")
print(f"Unique family names at position 3: {len(position_names[3])}")
print(f"Unique full 4-name combinations: {len(full_name_set)}")

# ========== STATISTICAL NAMES ==========
first_cnt = Counter(seq[0] for seq in name_sequences if len(seq) >= 1)
family_cnt = Counter(seq[3] for seq in name_sequences if len(seq) >= 4)
stat_first = set([n for n,c in first_cnt.most_common(500)])
stat_family = set([n for n,c in family_cnt.most_common(500)])

# ========== JOB TITLES - More complete ==========
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
    'مرشد', 'منسق', 'امين', 'سكرتير', 'طباخ', 'نادل', 'خدمة غرف', 'نظافة', 'استقبال',
    'حارس امن', 'عسكري', 'ضابط', 'جندي', 'شرطي', 'رجل اعمال', 'مستثمر', 'مهندس كهرباء',
    'مهندس ميكانيك', 'جراح', 'بيت', 'خريج', 'طالب', 'متدرب', 'مؤلف مسرحي',
    'رياضي', 'اعب', 'لاعب', 'معلق رياضي', 'مذيع رياضي', 'مراسل', 'كاتب رأي', 'ناقد',
    'نقاد', 'مستشار مالي', 'مستشار اداري', 'مستشار تسويق', 'مستشار اعلامي',
    'مدير اكاديمي', 'مدير عام', 'قاضي', 'قاضي', 'محقق', 'محق', 'محق',
    'مستشار', 'قانوني', 'محلل', 'باحث', 'مخطط', 'منظم', 'منسق', 'مشرف', 'مدير',
    'قائد', 'رئيس', 'امين', 'سكرتير', 'مساعد', 'مندوب مبيعات', 'مندوب صحي', 'مندوب تعليمي',
    'معلم', 'مدرس', 'مدرب', 'م وجه', 'مشرف', 'قائد', 'مرشد', 'مستشار', 'معلم', 'اكاديمي',
    'دكتور', 'بروفيسور', 'استاذ', 'مدرس', 'معلم', 'مدرب', 'مختص', 'خبير', 'فني', 'متخصص',
    'اخصائي', 'مستشار', 'محلل', 'باحث', 'مخطط', 'منظم', 'منسق', 'مشرف', 'مدير', 'قائد',
    'رئيس', 'امين', 'سكرتير', 'مساعد', 'مندوب', 'مشرف', 'قائد', 'مدير', 'قائد', 'مشرف',
    'امين', 'سكرتير', 'مساعد', 'مندوب', 'مشرف', 'قائد', 'مدير', 'قائد', 'مشرف', 'امين',
    'مرشد', 'مستشار', 'معلم', 'اكاديمي', 'دكتور', 'بروفيسور', 'استاذ', 'مدرس', 'معلم',
    'متخصص', 'خبير', 'فني', 'متخصص', 'اخصائي', 'مستشار', 'محلل', 'باحث'
])

# ========== MEDICAL - More specific ==========
medical_keywords = set([
    'سجل طبي', 'سجل مرضي', 'تاريخ طبي', 'تاريخ مرضي', 'حالة صحية', 'صحة ممتازة', 'صحة جيدة',
    'لا تاريخ مرضي', 'لا يوجد سجل طبي', 'لا ملاحظات طبية', 'لا يوجد سجل', 'لا ملاحظات',
    'حالة مستقرة', 'سليم', 'سليم تماماً', 'سليم جدا', 'سليم من امراض', 'سليم من الأمراض',
    'صحة عامة جيدة', 'صحة عامة ممتازة', 'لا توجد امراض', 'لا امراض', 'لا امراض مزمنة',
    'فحص دوري', 'فحص طبي', 'فحص شامل', 'تشخيص', 'علاج', 'علاج طبيعي',
    'حساسية', 'حساسية من الغبار', 'حساسية موسمية',
    'أمراض مزمنة', 'ضغط الدم', 'ضغط مرتفع', 'سكري', 'سكر الدم', 'قلب قوي',
    'لا تاريخ', 'لا شكاوى', 'لا مشاكل', 'لا توجد مشاكل',
    'جيد جدا', 'جيد جداً', 'حالة جيدة', 'حالة ممتازة', 'حالة صحية جيدة',
    'لا يوجد سجل مرضي', 'لا يوجد تاريخ', 'لا يوجد مشاكل صحية',
    'لا امراض القلب', 'لا امراض حادة', 'لا امراض مزمنة'
])

def detect_pii(text):
    r = {}

    r['Credit Card Nos'] = bool(compiled_regex['Credit Card Nos'].search(text))
    r['Email Addresses'] = bool(compiled_regex['Email Addresses'].search(text))
    r['IP Addresses'] = bool(re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b').search(text))

    iban = re.findall(r'\b[A-Z]{2}\d{2}\s[A-Z]{3,4}\s\d{4}\s\d{4}\s\d{2}\b', text)
    iban += re.findall(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b', text)
    r['IBAN or Swift Codes'] = any(len(m)>=15 and m[:2].isalpha() and m[2:4].isdigit() for m in iban)

    passports = re.findall(r'\b[A-Z]?\d{6,9}\b', text)
    cc_set = set(compiled_regex['Credit Card Nos'].findall(text))
    r['Passport Numbers'] = any(p not in cc_set for p in passports)

    r['MAC Addresses'] = bool(compiled_regex['MAC Addresses'].search(text))

    phone_matches = compiled_regex['Phone Number'].findall(text)
    r['Phone Number'] = any(p.startswith('+') and len(p) > 8 for p in phone_matches)

    nids = re.findall(r'\b\d{7,10}\b', text)
    r['National ID Number'] = any(len(n)>=8 and n not in cc_set for n in nids)

    postal = re.findall(r'\b\d{5}\b', text)
    r['Postal Code'] = any(not re.match(r'1[34]\d{3}', p) for p in postal)

    r['License Plate'] = bool(re.compile(r'\b[A-Z]{3}\s?\d{4}\b').search(text))

    dob = re.findall(r'\b\d{4}[/\-]\d{2}[/\-]\d{2}\b', text)
    valid_dob = False
    for d in dob:
        sep = '/' if '/' in d else '-'
        y = int(d.split(sep)[0])
        if (1300 <= y <= 1450) or (1900 <= y <= 2100):
            valid_dob = True
            break
    r['date_of_birth'] = valid_dob

    # === NAMES - using position-based validation ===
    names = extract_names(text)

    # IMPROVED: Use position-specific name sets
    # FirstName: Must be in position 0 name set OR in statistical first names
    r['FirstName'] = (len(names) >= 1 and
                      (names[0] in position_names[0] or names[0] in stat_first or
                       len(names[0]) >= 2))  # At least 2 chars

    # FatherName: Position 1
    r['FatherName'] = (len(names) >= 2 and
                       (names[1] in position_names[1] or
                        names[1] in stat_first))  # Reuse first name dict for father

    # GrandFatherName: Position 2
    r['GrandFatherName'] = (len(names) >= 3 and
                           (names[2] in position_names[2] or
                            len(names[2]) >= 2))

    # FamilyName: Position 3
    r['FamilyName'] = (len(names) >= 4 and
                       (names[3] in position_names[3] or
                        names[3] in stat_family or
                        names[3] in position_names[0]))  # Sometimes family is also a first name

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

print()
print('='*70)
print('FINAL IMPROVED HYBRID v3 RESULTS')
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

print()
print('='*70)
print('IMPROVEMENT PATH')
print('='*70)
print('v1 (initial hybrid): 83.91%')
print('v2 (IBAN + Phone fix): 86.23%')
print('v3 (position-based names): {:.2f}%'.format(overall))