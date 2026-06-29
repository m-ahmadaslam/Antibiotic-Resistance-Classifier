"""
Generate synthetic test data for the Healthcare Antibiotic Classifier.

Creates an .xlsx file with the 9 required columns + realistic organism / antibiotic
patterns that the data_processor's regex feature extractors will pick up.
"""

import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)
N = 500  # rows

ORGANISMS = [
    ("Escherichia coli", "ESBL producing E. coli isolated. Susceptible to meropenem, resistant to ceftriaxone."),
    ("Klebsiella pneumoniae", "Klebsiella pneumoniae detected. Multi-drug resistant. Sensitive to colistin."),
    ("Pseudomonas aeruginosa", "Pseudomonas aeruginosa isolated from blood culture. Resistant to piperacillin."),
    ("Staphylococcus aureus", "MRSA - Staphylococcus aureus detected. Susceptible to vancomycin and linezolid."),
    ("Staphylococcus epidermidis", "Staphylococcus epidermidis isolated. Coagulase negative. Sensitive to vancomycin."),
    ("Streptococcus pneumoniae", "Streptococcus pneumoniae detected. Susceptible to ceftriaxone and penicillin."),
    ("Enterococcus faecalis", "Enterococcus faecalis isolated. VRE - resistant to vancomycin."),
    ("Candida albicans", "Candida albicans isolated from urine. Susceptible to fluconazole."),
    ("Acinetobacter baumannii", "Acinetobacter baumannii detected. Carbapenem resistant. Sensitive to colistin only."),
    ("Proteus mirabilis", "Proteus mirabilis isolated. Susceptible to ciprofloxacin and ceftriaxone."),
    ("Enterobacter cloacae", "Enterobacter cloacae detected. AmpC producer. Susceptible to meropenem."),
    ("Stenotrophomonas maltophilia", "Stenotrophomonas maltophilia isolated. Susceptible to trimethoprim-sulfamethoxazole."),
    ("Haemophilus influenzae", "Haemophilus influenzae detected. Beta-lactamase positive. Susceptible to amoxicillin-clavulanate."),
    ("Salmonella", "Salmonella organism isolated from stool culture. Susceptible to ciprofloxacin."),
    ("No growth", "No organism isolated. No growth after 48 hours of incubation."),
]

ANTIBIOTICS = [
    "Ceftriaxone 1g IV", "Meropenem 500mg IV", "Vancomycin 1g IV", "Piperacillin-tazobactam 4.5g IV",
    "Ciprofloxacin 400mg IV", "Amoxicillin 500mg PO", "Amikacin 500mg IV", "Levofloxacin 750mg IV",
    "Cefepime 2g IV", "Imipenem 500mg IV", "Linezolid 600mg IV", "Colistin 150mg IV",
    "Fluconazole 200mg PO", "Caspofungin 70mg IV", "Azithromycin 500mg PO", "Doxycycline 100mg PO",
    "Trimethoprim-sulfamethoxazole 960mg PO", "Metronidazole 500mg IV", "Gentamicin 240mg IV",
    "Cefuroxime 1.5g IV", "Cefazolin 1g IV", "Ampicillin 1g IV", "Tigecycline 100mg IV",
    "Daptomycin 350mg IV", "Ertapenem 1g IV", "Tobramycin 240mg IV", "Clindamycin 600mg IV",
    "Cefotaxime 1g IV", "Moxifloxacin 400mg IV", "Teicoplanin 400mg IV",
]

WARDS = ["ICU", "ICU-CARDIAC", "ICU-NEURO", "MED-A", "MED-B", "SURG-1", "SURG-2", "EMERGENCY", "ONCOLOGY", "PEDIATRICS", "MATERNITY", "ORTHO", "NEPHRO"]
WARD_CODES = ["W001", "W002", "W003", "W004", "W005", "W006", "W007", "W008", "W009", "W010"]
DEPT_CODES = ["D-MED", "D-SURG", "D-ICU", "D-ED", "D-ONC", "D-NEPH", "D-CARD", "D-NEURO", "D-PED"]

# A small "ground truth" mapping the model can learn:
# Some antibiotics are well-matched to certain organisms (M = Match), others not (N = Mismatch)
MATCHES = {
    "Escherichia coli": ["meropenem", "ciprofloxacin", "amikacin", "ertapenem"],
    "Klebsiella pneumoniae": ["meropenem", "colistin", "tigecycline"],
    "Pseudomonas aeruginosa": ["piperacillin", "cefepime", "meropenem", "amikacin", "ciprofloxacin"],
    "Staphylococcus aureus": ["vancomycin", "linezolid", "daptomycin", "cefazolin"],
    "Staphylococcus epidermidis": ["vancomycin", "linezolid"],
    "Streptococcus pneumoniae": ["ceftriaxone", "penicillin", "amoxicillin"],
    "Enterococcus faecalis": ["linezolid", "daptomycin", "ampicillin"],
    "Candida albicans": ["fluconazole", "caspofungin"],
    "Acinetobacter baumannii": ["colistin", "tigecycline"],
    "Proteus mirabilis": ["ciprofloxacin", "ceftriaxone", "cefotaxime"],
    "Enterobacter cloacae": ["meropenem", "ertapenem", "cefepime"],
    "Stenotrophomonas maltophilia": ["trimethoprim-sulfamethoxazole", "levofloxacin"],
    "Haemophilus influenzae": ["amoxicillin", "ceftriaxone", "azithromycin"],
    "Salmonella": ["ciprofloxacin", "ceftriaxone", "azithromycin"],
}

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days), hours=random.randint(0, 23))

rows = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)

for _ in range(N):
    org_name, exam_result = random.choice(ORGANISMS)
    antibiotic = random.choice(ANTIBIOTICS)
    abx_lower = antibiotic.lower()

    # Compute "Result" label (Match / Mismatch / Unknown) for variety
    match_keywords = MATCHES.get(org_name, [])
    if org_name == "No growth":
        result_label = "N"  # No abnormal finding
    elif any(k in abx_lower for k in match_keywords):
        result_label = "A"  # Abnormal -> meaningful antibiotic decision case
    else:
        result_label = random.choice(["A", "N"])

    exam_order_date = random_date(start_date, end_date)
    exam_report_date = exam_order_date + timedelta(days=random.randint(1, 4))
    antibiotic_order_date = exam_report_date + timedelta(hours=random.randint(0, 48))

    rows.append({
        "EXAM_ORDER_DATE": exam_order_date,
        "EXAM_REPORT_DATE": exam_report_date,
        "EXAM_RESULT": exam_result,
        "EXAM_ABNORMAL_RESULT": result_label,
        "ANTIBIOTIC_ORDER_DATE": antibiotic_order_date,
        "ANTIBIOTIC_ORDER_NAME": antibiotic,
        "EXAM_ORDER_WARD_CD": random.choice(WARD_CODES),
        "ANTIBIOTIC_ORDER_WARD": random.choice(WARDS),
        "ORDERING_DEPARTMENT_CD": random.choice(DEPT_CODES),
    })

df = pd.DataFrame(rows)
out = "data/sample_antibiotic_data.xlsx"
import os
os.makedirs("data", exist_ok=True)
df.to_excel(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
print(df.head(3).to_string())
