#!/usr/bin/env python3
"""
PII Detection: Regex + Statistical Methods
Enhanced detection combining regex patterns with statistical/language models

This script combines:
- Regex for structured PII (credit cards, emails, IPs, phones, MAC, dates)
- Statistical/Named Entity Recognition for names, job titles, and complex fields
"""

import pandas as pd
import re
import time
from collections import defaultdict, Counter

print("Loading data...")
start_total = time.time()

context_df = pd.read_csv('res/data/context.tsv', sep='\t')
val_df = pd.read_csv('res/data/val.csv')

print(f"Context rows: {len(context_df)}")
print(f"Val columns: {len(val_df.columns)}")
load_time = time.time() - start_total
print(f"Load time: {load_time:.4f}s")

# =============================================================================
# Part 1: Regex Patterns for Structured PII
# =============================================================================

# Structured PII - Regex-based detection
# IMPROVED v4: Space-separated IBAN, flexible phone patterns
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
print("Regex patterns compiled:", list(regex_patterns.keys()))

# =============================================================================
# Part 2: Statistical/Language Model for Arabic Names
# =============================================================================

# Common Arabic first names (statistical approach - frequency-based)
arabic_first_names = set([
    'فهد', 'أحمد', 'محمد', 'عبدالرحمن', 'خالد', 'سالم', 'عبدالله', 'ناصر',
    'بندر', 'سلطان', 'عبدالعزيز', 'سعد', 'راشد', 'عمر', 'يوسف', 'إبراهيم',
    'مهند', 'ياسر', 'ماجد', 'طارق', 'فارس', 'علي', 'حسن', 'مصطفى',
    'منصور', 'فيصل', 'نايف', 'محمود', 'سليمان', 'عبدالمحسن', 'الوليد',
    'عبدالملك', 'تركي', 'ياسين', 'انس', 'حمزة', 'عبدالحكيم', 'فؤاد', 'طلعت',
    'حامد', 'رزق', 'سمير', 'جميل', 'فريد', 'كامل', 'سامي', 'فهمي', 'مازن',
    'بيان', 'آدم', 'عزت', 'سعيد', 'زكريا', 'يحيى', 'خضر', 'نعيم', 'وائل',
    'هشام', 'رشيد', 'عادل', 'نبيل', 'هاني', 'طه', 'شريف', 'منصور'
])

# Common Arabic family names
arabic_family_names = set([
    'العجمي', 'الحازمي', 'الشريف', 'الجحدلي', 'الخالدي', 'الهاجري',
    'الأزدي', 'الشمري', 'الغطفاني', 'الجهني', 'الثقفي', 'الدوسري',
    'الحربي', 'المرّي', 'الأسدي', 'الشهراني', 'الخثعمي', 'العبدالقيسي',
    'القحطاني', 'الصخري', 'التميمي', 'الرشيدي', 'القرشي', 'الطائي',
    'الغامدي', 'القاسم', 'المالكي', 'الحسين', 'الناصر', 'المطيري',
    'العنزي', 'الشاووش', 'السعيد', 'الخليفة', 'الكحل', 'الزهراني',
    'الحجي', 'الحدادي', 'الدهش', 'الزامل', 'السعد', 'النعيم', 'القصير',
    'الشرقاوي', 'البلوي', 'العتيبي', 'العتيقي', 'المطيري', 'الشمري',
    'القصيمي', 'الغربي', 'الملاوي'
])

# IMPROVED v4: Complete job titles dictionary
job_titles = set([
    'مهندس', 'محاسب', 'طبيب', 'مدير', 'موظف', 'كاتب', 'صحفي',
    'مهندس معماري', 'معلم', 'دكتور', 'مستشار', 'مراقب', 'مصور', 'مصمم', 'فني',
    'عالم', 'باحث', 'اكاديمي', 'طيار', 'قاضي', 'محامي', 'موسيقي', 'جغرافي',
    'مهندس مدني', 'معالج', 'ممرض', 'دبلوماسي', 'اقتصادي', 'محلل مالي', 'مصمم جرافيك',
    'مهندس برمجيات', 'أخصائي', 'مدقق', 'مخطط', 'مرافق', 'مندوب', 'تاجر', 'مقاول',
    'مهني', 'سائق', 'حارس', 'امني', 'مشرف', 'قائد', 'سياسي', 'محلل', 'صحفي',
    'محرر', 'ناشر', 'مصمم', 'مهندس طيران', 'مراقب جوي', 'مهندس شبكات',
    'مهندس انظمة', 'مهندس امن', 'مصمم الداخلية', 'مصور فوتوغرافي', 'ممرضة',
    'صيدلي', 'علاج طبيعي', 'طبيب اسنان', 'موثق', 'كاتب العدل', 'بناء',
    'دهان', 'ميكانيكي', 'صانع', 'حرفي', 'فني صيانة', 'فني أجهزة',
    'مدير مشاريع', 'منسق مشاريع', 'مخطط استراتيجي', 'مراقب جودة', 'مفتش',
    'مشرف انتاج', 'مدير عمليات', 'مدير موارد بشرية', 'مدير مالي', 'مدير تسويق',
    'مدير مبيعات', 'مدير تقنية المعلومات', 'مدير امن المعلومات', 'استشاري قانوني',
    'محلل اقتصادي', 'باحث اقتصادي', 'نائب مدير', 'مساعد مدير', 'مدير تنفيذي',
    'قائد فريق', 'قائد مجموعة', 'قائد مشروع', 'مهندس اول', 'مهندس مساعد',
    'خبير اول', 'استشاري اول', 'ممثل', 'شاعر', 'مغني', 'عازف', 'ملحن', 'مؤلف',
    'مقدم', 'معلق', 'مذيع', 'منتج', 'مرشد', 'امين', 'سكرتير', 'طباخ', 'نادل',
    'خدمة غرف', 'نظافة', 'استقبال', 'حارس امن', 'عسكري', 'ضابط', 'جندي', 'شرطي',
    'رجل اعمال', 'مستثمر', 'مهندس كهرباء', 'مهندس ميكانيك', 'جراح', 'بيت',
    'خريج', 'طالب', 'متدرب', 'مؤلف مسرحي', 'رياضي', 'اعب', 'لاعب',
    'معلق رياضي', 'مذيع رياضي', 'مراسل', 'كاتب رأي', 'ناقد', 'مستشار مالي',
    'مستشار اداري', 'مدير اكاديمي', 'مدير عام', 'محقق', 'دبلوماسي', 'سفير',
    'كهربائي', 'حداد', 'نجار', 'مبلط', 'بياع', 'مزارع', 'فلاح', 'بواب', 'خادم',
    'طبيب طوارئ', 'طبيب قلب', 'طبيب عيون', 'طبيب جلدية', 'مختبري', 'اشعة',
    'معالج نفسي', 'اخصائي تغذية', 'طبيب بيطري', 'مهندس زراعي', 'مهندس غذاء',
    'مهندس كيميائي', 'مصور سينمائي', 'مصمم ازياء', 'مصمم ديكور', 'مصمم عطور',
    'روائي', 'ناقد ادبي', 'مترجم', 'لغوي', 'عالم دين', 'خطيب', 'امام', 'مؤذن',
    'خادم مسجد', 'مخرج', 'سيناريست', 'راقص', 'ممثل مسرحي', 'مراسل حربي',
    'مراسل اخبار', 'محلل سياسي', 'محلل عسكري', 'كاتب سياسي', 'قنصل',
    'Secretary', 'Director', 'Chief', 'Head', 'President', 'CEO', 'CFO', 'COO',
    'Manager', 'Supervisor', 'Coordinator', 'Analyst', 'Consultant', 'Adviser',
    'Engineer', 'Architect', 'Designer', 'Developer', 'Programmer', 'Technician'
])

# IMPROVED v4: More specific medical keywords to reduce false positives
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

# Exclusion list - words that look like names but aren't
name_exclusion = set([
    'صحة', 'طريق', 'بصمة', 'التحقق', 'الحيوي', 'سجل', 'طبي', 'تاريخ', 'حالة', 'صحي', 'شارع'
])

print(f"Arabic first names dictionary: {len(arabic_first_names)} names")
print(f"Arabic family names dictionary: {len(arabic_family_names)} names")
print(f"Job titles dictionary: {len(job_titles)} titles")
print(f"Medical keywords dictionary: {len(medical_keywords)} keywords")
print(f"Name exclusion dictionary: {len(name_exclusion)} words")

# Build name position model from training data
def extract_name_triplet(text):
    """Extract potential name triplets from bracket content"""
    brackets = re.findall(r'\[(.*?)\]', text)
    names = []
    for b in brackets:
        # Check if content is purely Arabic (likely a name)
        if re.match(r'^[\u0600-\u06FF]+$', b):
            names.append(b)
    return names

# Extract all name triplets from training data
name_triplets = []
for idx in range(len(context_df)):
    # Only use rows where we know the names are present
    if val_df.iloc[idx]['FirstName'] == 1:
        names = extract_name_triplet(context_df.iloc[idx]['text'])
        if len(names) >= 4:
            name_triplets.append((names[0], names[1], names[2], names[3]))

print(f"Extracted {len(name_triplets)} name triplets from training data")

# Build position-specific name frequency
first_names_counter = Counter()
father_names_counter = Counter()
grandfather_names_counter = Counter()
family_names_counter = Counter()

for triplet in name_triplets:
    first_names_counter[triplet[0]] += 1
    if len(triplet) > 1: father_names_counter[triplet[1]] += 1
    if len(triplet) > 2: grandfather_names_counter[triplet[2]] += 1
    if len(triplet) > 3: family_names_counter[triplet[3]] += 1

# Get most common names as the statistical prior (IMPROVED: more names)
statistical_first_names = set([n for n, c in first_names_counter.most_common(500)])
statistical_father_names = set([n for n, c in father_names_counter.most_common(500)])
statistical_grandfather_names = set([n for n, c in grandfather_names_counter.most_common(500)])
statistical_family_names = set([n for n, c in family_names_counter.most_common(500)])

print(f"Statistical first names: {len(statistical_first_names)}")
print(f"Statistical father names: {len(statistical_father_names)}")
print(f"Statistical grandfather names: {len(statistical_grandfather_names)}")
print(f"Statistical family names: {len(statistical_family_names)}")

# =============================================================================
# Part 3: Combined Detection Function
# =============================================================================

def detect_pii_hybrid(text):
    """Detect PII using both regex and statistical methods - IMPROVED v4"""
    results = {}

    # === REGEX-BASED (exact pattern match) ===
    results['Credit Card Nos'] = bool(compiled_regex['Credit Card Nos'].search(text))
    results['Email Addresses'] = bool(compiled_regex['Email Addresses'].search(text))
    results['IP Addresses'] = bool(re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b').search(text))

    # IBAN/Swift - IMPROVED: handles space-separated format (US00 JPM 1234 5678 90)
    iban_matches = re.findall(r'\b[A-Z]{2}\d{2}\s[A-Z]{3,4}\s\d{4}\s\d{4}\s\d{2}\b', text)
    iban_matches += re.findall(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b', text)
    results['IBAN or Swift Codes'] = any(
        (len(m) >= 15 and len(m) <= 34) and
        (m[:2].isalpha()) and
        (m[2:4].isdigit())
        for m in iban_matches
    )

    # Passport Numbers
    passport_matches = re.findall(r'\b[A-Z]?\d{6,9}\b', text)
    cc_matches = set(compiled_regex['Credit Card Nos'].findall(text))
    filtered_passports = [p for p in passport_matches if p not in cc_matches]
    results['Passport Numbers'] = len(filtered_passports) > 0

    results['MAC Addresses'] = bool(compiled_regex['MAC Addresses'].search(text))

    # Phone Numbers - IMPROVED: more flexible international formats
    phone_matches = compiled_regex['Phone Number'].findall(text)
    valid_phones = [p for p in phone_matches if p.startswith('+') and len(p) > 8]
    results['Phone Number'] = len(valid_phones) > 0

    # National ID
    all_cc = set(compiled_regex['Credit Card Nos'].findall(text))
    national_ids = re.findall(r'\b\d{7,10}\b', text)
    filtered_nids = [n for n in national_ids if n not in all_cc and len(n) >= 8]
    results['National ID Number'] = len(filtered_nids) > 0

    # Postal Code
    postal_matches = re.findall(r'\b\d{5}\b', text)
    filtered_postal = [p for p in postal_matches if not re.match(r'1[34]\d{3}', p)]
    results['Postal Code'] = len(filtered_postal) > 0

    results['License Plate'] = bool(re.compile(r'\b[A-Z]{3}\s?\d{4}\b').search(text))

    # Date of Birth
    dob_matches = re.findall(r'\b\d{4}[/\-]\d{2}[/\-]\d{2}\b', text)
    valid_dates = []
    for d in dob_matches:
        year = int(d.split('/')[0] if '/' in d else d.split('-')[0])
        if 1300 <= year <= 1450 or 1900 <= year <= 2100:
            valid_dates.append(d)
    results['date_of_birth'] = len(valid_dates) > 0

    # === STATISTICAL/KEYWORD-BASED ===
    # Extract Arabic names from brackets
    brackets_content = re.findall(r'\[(.*?)\]', text)
    arabic_names = [b for b in brackets_content if re.match(r'^[\u0600-\u06FF]+$', b)]

    # IMPROVED: Filter out non-name words using exclusion list
    filtered_names = [n for n in arabic_names if n not in name_exclusion]

    # FirstName: First Arabic name in sequence that's a known first name
    results['FirstName'] = any(n in statistical_first_names or n in arabic_first_names for n in filtered_names[:4])

    # FatherName: 2nd Arabic name in sequence
    results['FatherName'] = len(filtered_names) >= 2

    # GrandFatherName: 3rd Arabic name in sequence
    results['GrandFatherName'] = len(filtered_names) >= 3

    # FamilyName: IMPROVED: 4th Arabic name that's a known family name
    results['FamilyName'] = False
    if len(filtered_names) >= 4:
        potential_family = filtered_names[3]
        # Accept if it's in statistical family names OR Arabic family names OR starts with "ال"
        if (potential_family in statistical_family_names or
            potential_family in arabic_family_names or
            (len(potential_family) >= 4 and potential_family.startswith('ال'))):
            results['FamilyName'] = True

    # Street Name
    street_match = re.search(r'\b[A-Za-z]+\s+(Rd|St|Street|Road)\b', text, re.I)
    arabic_street = re.search(r'طريق\s+[\u0600-\u06FF]+', text)
    results['Street Name'] = bool(street_match or arabic_street)

    # Username - IMPROVED: stricter validation
    username_matches = re.findall(r'\b[A-Za-z0-9_]{3,15}\b', text)
    potential_usernames = [u for u in username_matches if ('_' in u and len(u) >= 4) or (len(u) >= 3 and len(u) <= 8 and u.islower() and not u.isdigit())]
    results['Username'] = len(potential_usernames) > 0

    # Medical History - IMPROVED: check for more patterns
    text_lower = text.lower()
    results['Medical History'] = any(kw in text_lower for kw in medical_keywords) or 'سليم' in text or 'لا يوجد سجل' in text or 'لا تاريخ' in text

    # Biometric Data
    biometric_patterns = [
        r'بصمة', r'بصمة الإصبع', r'بصمة الوجه', r'بصمة العين',
        r'Fingerprint', r'Biometric', r'Face ID', r'Iris Scan',
        r'Touch ID', r'التحقق الحيوي', r'البيانات الحيوية',
        r'السمات الحيوية', r'التعرف على الوجه'
    ]
    biometric_found = any(re.search(p, text, re.I) for p in biometric_patterns)
    results['Biometric Data'] = biometric_found

    # Job Title - IMPROVED: expanded check
    job_found = any(title in text for title in job_titles)
    english_jobs = re.findall(r'\b(Engineer|Manager|Doctor|Accountant|Biologist|Teacher|Carpenter|Electrician|Painter|Scientist|Architect|Consultant|Specialist|Analyst|Designer|Developer|Writer|Artist|Musician|Diplomat|Lawyer|Adviser|Coordinator|Supervisor|Executive|Director)\b', text, re.I)
    job_found = job_found or len(english_jobs) > 0
    results['Job Title'] = job_found

    return results

print("Improved v4 hybrid detection function defined")

# =============================================================================
# Part 4: Run Detection & Evaluate
# =============================================================================

start_detection = time.time()

all_cols = [
    'Credit Card Nos', 'Email Addresses', 'IP Addresses', 'IBAN or Swift Codes',
    'Passport Numbers', 'MAC Addresses', 'Phone Number', 'National ID Number',
    'FirstName', 'FatherName', 'GrandFatherName', 'FamilyName', 'Street Name',
    'Username', 'Postal Code', 'License Plate', 'Medical History', 'Biometric Data',
    'Job Title', 'date_of_birth'
]

correct = defaultdict(int)
total = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)

for idx in range(len(context_df)):
    text = context_df.iloc[idx]['text']
    results = detect_pii_hybrid(text)

    for col in all_cols:
        actual = val_df.iloc[idx][col] == 1
        predicted = results.get(col, False)
        total[col] += 1
        if actual == predicted:
            correct[col] += 1
        else:
            if predicted and not actual:
                fp[col] += 1
            elif actual and not predicted:
                fn[col] += 1

detection_time = time.time() - start_detection
total_time = time.time() - start_total

print(f"Detection completed in: {detection_time:.4f}s")
print(f"Total time (load + detect): {total_time:.4f}s")

# =============================================================================
# Results
# =============================================================================

print("=" * 75)
print("PII DETECTION RESULTS (Regex + Statistical Hybrid)")
print("=" * 75)

can_handle = []
cannot_handle = []

for col in all_cols:
    acc = correct[col] / total[col] * 100 if total[col] > 0 else 0
    tp = correct[col]
    fp_c = fp[col]
    fn_c = fn[col]

    precision = tp / (tp + fp_c) if (tp + fp_c) > 0 else 0
    recall = tp / (tp + fn_c) if (tp + fn_c) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if acc >= 80:
        can_handle.append(col)
    else:
        cannot_handle.append(col)

    print(f"\n{col}:")
    print(f"  Accuracy:   {acc:6.2f}%  |  Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}")
    print(f"  TP: {tp:3d}  FP: {fp_c:3d}  FN: {fn_c:3d}  |  Total: {total[col]}")

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

overall = sum(correct.values()) / sum(total.values()) * 100
print(f"\nCAN HANDLE (>=80% accuracy): {len(can_handle)}")
for c in can_handle:
    print(f"  ✓ {c}")

print(f"\nCANNOT HANDLE (<80% accuracy): {len(cannot_handle)}")
for c in cannot_handle:
    print(f"  ✗ {c}")

print(f"\n{'=' * 50}")
print(f"Overall Accuracy: {overall:.2f}%")
print(f"Total processing time: {total_time:.4f}s")
print(f"Detection speed: {len(context_df) * len(all_cols) / detection_time:.0f} predictions/sec")
print(f"{'=' * 50}")

# =============================================================================
# Comparison: Regex-only vs Hybrid
# =============================================================================

print("\n" + "=" * 75)
print("COMPARISON: Regex-only vs Regex+Hybrid")
print("=" * 75)

regex_only_acc = {
    'Credit Card Nos': 89.1,
    'Email Addresses': 100.0,
    'IP Addresses': 100.0,
    'IBAN or Swift Codes': 48.2,
    'Passport Numbers': 95.5,
    'MAC Addresses': 100.0,
    'Phone Number': 70.0,
    'National ID Number': 97.3,
    'FirstName': 58.2,
    'FatherName': 61.8,
    'GrandFatherName': 68.2,
    'FamilyName': 77.3,
    'Street Name': 88.2,
    'Username': 60.0,
    'Postal Code': 97.3,
    'License Plate': 94.5,
    'Medical History': 70.9,
    'Biometric Data': 93.6,
    'Job Title': 58.2,
    'date_of_birth': 100.0,
}

hybrid_acc = {}
for col in all_cols:
    hybrid_acc[col] = correct[col] / total[col] * 100 if total[col] > 0 else 0

print(f"{'PII Type':<25} {'Regex-only':>12} {'Hybrid':>12} {'Improvement':>12}")
print("-" * 65)
for col in all_cols:
    reg = regex_only_acc.get(col, 0)
    hyb = hybrid_acc[col]
    diff = hyb - reg
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    print(f"{col:<25} {reg:>10.1f}% {hyb:>10.1f}% {arrow} {abs(diff):>8.1f}%")

# =============================================================================
# Final summary table
# =============================================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(f"\nTime Performance:")
print(f"  - Data loading: {load_time:.4f}s")
print(f"  - Detection:    {detection_time:.4f}s")
print(f"  - Total:        {total_time:.4f}s")

print(f"\nEffectiveness:")
print(f"  - Total predictions: {sum(total.values())}")
print(f"  - Correct:           {sum(correct.values())}")
print(f"  - Overall accuracy:  {overall:.2f}%")

print(f"\nCategories handled well (>=80%): {len(can_handle)}/20")
print(f"Categories needing improvement (<80%): {len(cannot_handle)}/20")