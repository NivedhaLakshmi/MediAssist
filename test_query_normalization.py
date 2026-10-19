from services.query_utils import normalize_health_query


tests = [
   "What is my heamoglobin level?",
    "What is my haemoglogin level?",
    "What is my creatininee level?",
    "What is my thrombocytopenai?",
    "What is my leukocitosis?",
    "What is my glomerular filtraton rate?",
    "What is my aminotransferace?",
    "What is my lymphocite count?",
    "What is my neutrofil count?",
    "What is my hematocryt level?",
    "What is my bilrubin level?",
    "What is my albumine level?",
    "What is my xyzabc?",
    "What is my hemoglobinx?",
    "What is my creatinor?",
    "What is my thromboflexia?",
    "What is my glomerulofilter?",
    "What is my leukozom?",
    "What is my aminotransferon?",
    "What is my lymphozite?",
    "What is my rate?",
    "What is my level?",
    "What is my concentration?",
    "What is my result?",
    "What is my value?",
    "What is my report?",
    "What is my blood pressure?",
    "What is my heart rate?",
    "What is my temperature?",
    "What is my heamoglobin level?",
    "What is my creatininee and glomerular filtraton rate?",
    "What are my lymphocite and neutrofil counts?",
    "What is my bilrubin level and albumine?",
    "What is my erythrocyte sedimantation rate?",
    "What is my mean corpusclar hemoglobin concentraton?",
]



for question in tests:

    result = normalize_health_query(question)

    print("\nOriginal:")
    print(result["original"])

    print("Normalized:")
    print(result["normalized"])

    print("Corrections:")
    print(result["corrections"])

    print("-" * 50)