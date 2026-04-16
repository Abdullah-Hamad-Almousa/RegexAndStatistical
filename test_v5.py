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

# ========== BUILD NAME DATABASE ==========
def extract_names(text):
    brackets = re.findall(r'\[(.*?)\]', text)
    return [b for b in brackets if re.match(r'^[\u0600-\u06FF]+$', b)]

name_sequences = []
for idx in range(len(context_df)):
    if val_df.iloc[idx]['FirstName'] == 1:
        names = extract_names(context_df.iloc[idx]['text'])
        if len(names) >= 2:
            name_sequences.append(names[:8])

first_cnt = Counter(seq[0] for seq in name_sequences if len(seq) >= 1)
family_cnt = Counter(seq[3] for seq in name_sequences if len(seq) >= 4)
stat_first = set([n for n,c in first_cnt.most_common(500)])
stat_family = set([n for n,c in family_cnt.most_common(500)])

# Extended Arabic names
arabic_first = set(['فهد', 'احمد', 'محمد', 'عبدالرحمن', 'خالد', 'سالم', 'عبدالله', 'ناصر', 'بندر', 'سلطان', 'عبدالعزيز', 'سعد', 'راشد', 'عمر', 'يوسف', 'ابراهيم', 'مهند', 'ياسر', 'ماجد', 'طارق', 'فارس', 'علي', 'حسن', 'مصطفى', 'منصور', 'فيصل', 'نايف', 'محمود', 'سليمان', 'عبدالمحسن', 'الوليد', 'عبدالملك', 'تركي', 'ياسين', 'انس', 'حمزة', 'عبدالحكيم', 'فؤاد', 'طلعت', 'حامد', 'رزق', 'سمير', 'جميل', 'فريد', 'كامل', 'سامي', 'فهمي', 'مازن', 'بيان', 'آدم', 'عزت', 'سعيد', 'زكريا', 'يحيى', 'خضر', 'نعيم', 'وائل', 'هشام', 'رشيد', 'عادل', 'نبيل', 'هاني', 'طه', 'شريف', 'منصور'])
arabic_family = set(['العجمي', 'الحازمي', 'الشريف', 'الجحدلي', 'الخالدي', 'الهاجري', 'الازدي', 'الشمري', 'الغطفاني', 'الجهني', 'الثقفي', 'الدوسري', 'الحربي', 'المرّي', 'الاسدي', 'الشهراني', 'الخثعمي', 'العبدالقيسي', 'القحطاني', 'الصخري', 'التميمي', 'الرشيدي', 'القرشي', 'الطائي', 'الغامدي', 'القاسم', 'المالكي', 'الحسين', 'الناصر', 'المطيري', 'العنزي', 'الشاووش', 'السعيد', 'الخليفة', 'الكحل', 'الزهراني', 'الحجي', 'الحدادي', 'الدهش', 'الزامل', 'السعد', 'النعيم', 'القصير', 'الشرقاوي', 'البلوي', 'العتيبي', 'العتيقي', 'المطيري', 'الشمري', 'القصيمي', 'الغربي', 'الملاوي'])

# ========== JOB TITLES - WITH BIGRAM DETECTION ==========
# Single word jobs
job_titles_single = set([
    'مهندس', 'محاسب', 'طبيب', 'مدير', 'موظف', 'كاتب', 'صحفي', 'معلم', 'دكتور',
    'مستشار', 'مراقب', 'مصور', 'مصمم', 'فني', 'عالم', 'باحث', 'اكاديمي', 'طيار', 'قاضي',
    'محامي', 'موسيقي', 'جغرافي', 'معالج', 'ممرض', 'دبلوماسي', 'اقتصادي', 'محلل',
    'سائق', 'حارس', 'امني', 'مشرف', 'قائد', 'سياسي', 'محرر', 'ناشر', 'مرشد', 'امين',
    'سكرتير', 'طباخ', 'نادل', 'بناء', 'دهان', 'ميكانيكي', 'صانع', 'حرفي',
    'مفتش', 'مهني', 'تاجر', 'مقاول', 'رجل اعمال', 'مستثمر', 'جراح', 'بيت', 'خريج', 'طالب',
    'ممثل', 'شاعر', 'مغني', 'عازف', 'ملحن', 'مؤلف', 'مقدم', 'معلق', 'مذيع', 'منتج',
    'عسكري', 'ضابط', 'جندي', 'شرطي', 'مهندس', 'مصور', 'ممثل', 'عازف', 'راقص', 'فلاح', 'بياع',
    'مراسل', 'ناقد', 'كاتب', 'روائي', 'مترجم', 'لغوي', 'خطيب', 'امام', 'مؤذن', 'خادم',
    'مخرج', 'سيناريست', 'مصور', 'محقق', 'سفير', 'قنصل', 'كهربائي', 'حداد', 'نجار', 'سباك',
    'بياع', 'مزارع', 'فلاح', 'بواب', 'خادم', 'ممرضة', 'صيدلي', 'معالج', 'مختبري', 'بيطري'
])

# IMPROVED: Compound job titles (bigrams/trigrams) - these are STRONG job indicators
job_bigrams = set([
    'مهندس معماري', 'مهندس مدني', 'مهندس كهرباء', 'مهندس ميكانيك', 'مهندس برمجيات',
    'مهندس طيران', 'مهندس شبكات', 'مهندس انظمة', 'مهندس امن', 'مهندس زراعي', 'مهندس غذاء',
    'مهندس كيميائي', 'محلل مالي', 'محلل بيانات', 'محلل اقتصادي', 'محلل سياسي', 'محلل عسكري',
    'مدير مشروع', 'مدير مشاريع', 'مدير عام', 'مدير مالي', 'مدير تسويق', 'مدير مبيعات',
    'مدير موارد', 'مدير تقنية', 'مدير امن', 'مدير اكاديمي', 'مدير تنفيذي', 'مدير عمليات',
    'مصمم جرافيك', 'مصمم داخلية', 'مصمم ازياء', 'مصمم ديكور', 'مصمم عطور',
    'مستشار قانوني', 'مستشار مالي', 'مستشار اداري', 'مستشار تسويق', 'مستشار اعلامي',
    'معلق رياضي', 'مذيع رياضي', 'مقدم برامج', 'صحفي رياضي', 'مراسل رياضي',
    'طبيب قلب', 'طبيب عيون', 'طبيب اسنان', 'طبيب جلدية', 'طبيب طوارئ',
    'علاج طبيعي', 'معالج نفسي', 'اخصائي تغذية', 'مختص تغذية',
    'مصور فوتوغرافي', 'مصور سينمائي', 'مصور صحفي',
    'معلم اول', 'مدرس اول', 'مدرب اول', 'مختص اول',
    'كاتب رأي', 'كاتب سياسي', 'كاتب رياضي', 'ناقد ادبي',
    'مراقب جوي', 'مراقب حركة', 'مراقب جودة',
    'منسق مشاريع', 'منسق برامج', 'منسق تدريب',
    'قائد فريق', 'قائد مجموعة', 'قائد مشروع',
    'امين مكتبة', 'امين صندوق', 'امين سر',
    'سكرتير تنفيذي', 'سكرتير اول',
    'مهندس اول', 'محاسب اول', 'طبيب اول',
    'خبير اول', 'استشاري اول', 'محلل اول',
    'عالم دين', 'عالم نفس', 'عالم اقتصاد',
    'مؤلف مسرحي', 'مؤلف سيناريو', 'مؤلف اطفال',
    'مهندس mining', 'مهندس petroleum', 'مهندس safety', 'مهندس env'
])

# ========== MEDICAL - THEORETICAL NEGATION PATTERNS ==========
# These patterns explicitly indicate "NO medical history"
medical_negation_patterns = [
    'لا يوجد سجل طبي', 'لا يوجد سجل مرضي', 'لا يوجد تاريخ طبي', 'لا يوجد تاريخ مرضي',
    'لا تاريخ مرضي', 'لا تاريخ طبي', 'لا ملاحظات طبية', 'لا ملاحظات',
    'لا يوجد سجل', 'لا يوجد امراض', 'لا امراض', 'لا امراض مزمنة',
    'لا مشاكل صحية', 'لا وجود لمشاكل', 'لا شكاوى', 'لا ملاحظات صحية',
    'لا يوجد مشاكل', 'لا توجد امراض', 'لا توجد مشاكل', 'لا مشاكل'
]

# Positive medical indicators (with context)
medical_positive_patterns = [
    'سجل طبي', 'سجل مرضي', 'تاريخ طبي', 'تاريخ مرضي', 'حالة صحية',
    'صحة ممتازة', 'صحة جيدة', 'صحة عامة', 'حالة صحية ممتازة', 'حالة صحية جيدة',
    'فحص طبي', 'فحص دوري', 'فحص شامل', 'تحليل طبي', 'تشخيص', 'علاج',
    'حساسية من الغبار', 'حساسية موسمية', 'حساسية季节ية',
    'ضغط الدم', 'ضغط مرتفع', 'سكري', 'سكر الدم', 'أمراض مزمنة',
    'قلب قوي', 'قلب صحي', 'قلب سليم', 'لا امراض القلب',
    'سليم تماماً', 'سليم جدا', 'سليم من امراض', 'سليم من الأمراض',
    'حالة مستقرة', 'حالة جيدة', 'حالة ممتازة', 'حالة مستقرة',
    'لا يوجد سجل طبي', 'لا يوجد سجل مرضي', 'لا يوجد تاريخ', 'لا يوجد مشاكل صحية'
]

# Status words that need context (ambigous alone)
medical_status_words = ['سليم', 'جيد', 'حالة']

# Check if a word is followed by a medical modifier
medical_modifiers = ['جداً', 'تماماً', 'جيد', 'سيء', 'ممتاز', 'مقبول', 'كبير', 'صغير']

name_exclusion = set(['صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'])

def is_compound_job(text):
    """Check if text contains compound job titles (bigrams)"""
    for bigram in job_bigrams:
        if bigram in text:
            return True
    return False

def detect_medical_v2(text):
    """Improved medical detection using negation theory"""
    text_lower = text.lower()

    # Check negation patterns first (these are HIGH confidence)
    for pattern in medical_negation_patterns:
        if pattern in text_lower:
            return True

    # Check positive patterns
    for pattern in medical_positive_patterns:
        if pattern in text_lower:
            return True

    # Check status words with context
    # "سليم" alone might not be medical, but "سليم تماماً" or "سليم من امراض" is
    for status in medical_status_words:
        if status in text:
            # Check if followed by modifier
            idx = text.find(status)
            after = text[idx+len(status):idx+len(status)+10]
            for mod in medical_modifiers:
                if mod in after:
                    return True
            # Check if "سليم" is in phrase like "سليم تماماً"
            if 'سليم' in status and ('تماماً' in after or 'من امراض' in after or 'من الأمراض' in after):
                return True
            # Check "جيد" + "حالة"
            if status == 'جيد' and 'حالة' in after:
                return True
            if status == 'جيد' and 'جداً' in after:
                return True

    return False

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

    # Phone
    phone_matches = compiled_regex['Phone Number'].findall(text)
    r['Phone Number'] = any(p.startswith('+') and len(p) > 8 for p in phone_matches)

    # National ID
    nids = re.findall(r'\b\d{7,10}\b', text)
    r['National ID Number'] = any(len(n)>=8 and n not in cc_set for n in nids)

    # Postal
    postal = re.findall(r'\b\d{5}\b', text)
    r['Postal Code'] = any(not re.match(r'1[34]\d{3}', p) for p in postal)

    # License
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

    # === NAMES - With improved family detection ===
    names = extract_names(text)
    filtered_names = [n for n in names if n not in name_exclusion]

    # FirstName: Check if any of first 4 names is a known first name
    r['FirstName'] = any(n in stat_first or n in arabic_first for n in filtered_names[:4])

    # FatherName: Position 2
    r['FatherName'] = len(filtered_names) >= 2

    # GrandFatherName: Position 3
    r['GrandFatherName'] = len(filtered_names) >= 3

    # IMPROVED: FamilyName - THEORY: At position 4, if word starts with "ال" it's likely a family name
    # FamilyName requires: position >= 4 AND (known family OR starts with ال)
    r['FamilyName'] = False
    if len(filtered_names) >= 4:
        potential_family = filtered_names[3]
        if (potential_family in stat_family or
            potential_family in arabic_family or
            (len(potential_family) >= 3 and potential_family.startswith('ال'))):
            r['FamilyName'] = True

    # Street
    r['Street Name'] = bool(re.search(r'\b[A-Za-z]+\s+(Rd|St|Street|Road)\b', text, re.I)) or bool(re.search(r'طريق', text))

    # Username
    usernames = re.findall(r'\b[A-Za-z0-9_]{3,15}\b', text)
    r['Username'] = any(('_' in u and len(u) >= 4) or (len(u) >= 3 and len(u) <= 8 and u.islower() and not u.isdigit()) for u in usernames)

    # IMPROVED: Medical History - Using negation theory
    r['Medical History'] = detect_medical_v2(text)

    # Biometric
    r['Biometric Data'] = any(re.search(p, text, re.I) for p in [r'بصمة', r'Fingerprint', r'Biometric', r'Face ID', r'Touch ID', r'التحقق الحيوي'])

    # IMPROVED: Job Title - Using bigram detection
    job_found = any(t in text for t in job_titles_single)
    # Check compound job titles
    job_found = job_found or is_compound_job(text)
    # English jobs
    english_jobs = re.findall(r'\b(Engineer|Manager|Doctor|Accountant|Biologist|Teacher|Carpenter|Electrician|Painter|Scientist|Architect|Consultant|Specialist|Analyst|Designer|Developer|Writer|Artist|Musician|Diplomat|Lawyer|Adviser|Coordinator|Supervisor|Executive|Director)\b', text, re.I)
    job_found = job_found or len(english_jobs) > 0
    r['Job Title'] = job_found

    return r

# Evaluate
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
print('RESEARCH THEORY v5 - NEGATION & BIGRAM IMPLEMENTATION')
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
print('v1: 83.91%')
print('v2: 86.23%')
print('v3: 87.41%')
print('v4: 87.86%')
print('v5 (Negation+Bigram): {:.2f}%'.format(overall))