# MediAssist

**A local multimodal AI assistant for healthcare guidance, using retrieval-augmented generation (RAG) and the Model Context Protocol (MCP).**

MediAssist helps nurses get first-level answers about a patient's lab reports, medicines and guidelines. It runs entirely on hospital-controlled hardware. Patient records are only read and searched on-premises, and only medical keywords are ever sent to public sources.

- Project page and demo: [nivedhalakshmi.github.io/mediassist-site](https://nivedhalakshmi.github.io/mediassist-site)
- iAIM 2026 poster (AG03): [PDF](https://nivedhalakshmi.github.io/mediassist-site/files/MediAssist-poster.pdf)
- Final Year Project, Nanyang Technological University. Authors: Nivedha Lakshmi Magesh, Chee Wei Tan.

> Research prototype tested on synthetic reports. Not a medical device and not for clinical use.

## What it does

| Feature | How |
|---|---|
| Ask in any form | Typed questions (spelling checked against MeSH terms), voice (Faster Whisper, transcript shown for the nurse to check), photos, digital and scanned PDFs |
| Read lab reports | Digital PDFs use the text layer. Scans and photos are read by MedGemma 4B, with Tesseract OCR as the fallback |
| Answer from the patient's records | RAG over the patient's own reports (6 closest chunks, this patient only) and a local knowledge base (3 closest chunks) |
| Check lab numbers in code | Python decides whether each value is within, above or below the report's range. The model is given the verdict and must repeat it |
| Reach trusted public sources | An MCP server with DailyMed, ClinicalTrials.gov, MedlinePlus and PubMed. The model picks one source; only keywords leave the machine |
| Keep patients separate | One profile per patient, with saved conversations and attachments |

## Architecture

```
1 Input        Typed | Voice | Photo | Digital PDF | Scanned PDF
                         |
2 Understand   Prepare the question (MeSH)    Read documents (text layer, MedGemma 4B, OCR)
                         |
3 Context      Patient's reports (RAG)   Knowledge base (RAG)   MCP tools ---> Public sources
                         |                                       (keywords only)
4 Check        Lab numbers checked in code   Build the prompt
                         |
5 Answer       Qwen 2.5 7B writes the answer ---> Nurse reviews
```

Everything except the public-source search runs locally through [Ollama](https://ollama.com).

| Model | Role | Size |
|---|---|---|
| Qwen 2.5 7B Instruct | Answers questions, picks the MCP source | 7B |
| MedGemma 4B | Reads photos and scanned reports | 4B |
| all-MiniLM-L6-v2 | Embeddings for search | 22M |
| Faster Whisper (base) | Speech to text | 74M |
| Tesseract | OCR fallback | n/a |

## Project layout

```
app.py                     Streamlit app (the nurse-facing interface)
services/                  Core logic: prompts, lab checks, report reading, profiles, speech, embeddings
medical_mcp/               MCP server and tools (DailyMed, ClinicalTrials, MedlinePlus, PubMed, linked records)
components/chat_input/     Custom chat input with voice and attachments
ingest.py                  Builds the knowledge base from PDFs in data/
rag.py                     Knowledge base retrieval
privacy_audit.py           Checks what leaves the machine
tests/                     Unit tests (277)
experiments/eval/          Evaluation scripts, synthetic dataset and saved results
```

## Setup

Tested on macOS (Apple silicon, 32 GB RAM) with Python 3.12.

**1. System tools**

```bash
# Ollama: https://ollama.com/download
ollama pull qwen2.5:7b-instruct
ollama pull medgemma:4b   # if this name is not found, pull a MedGemma 4B build and set MEDGEMMA_MODEL

# Tesseract OCR (fallback reader)
brew install tesseract
```

**2. Python environment**

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. MeSH vocabulary**

Download the MeSH descriptor XML (for example `desc2026.xml`) from the [NLM MeSH download page](https://www.nlm.nih.gov/databases/download/mesh.html) and place it in `data/`. It is about 300 MB, so it is not in the repository. The vocabulary is cached to `data/mesh_vocabulary.pkl` on first run.

**4. Knowledge base**

Put reference PDFs in `data/` and build the vector store:

```bash
python ingest.py
```

## Running

Start the MCP server, then the app, in two terminals:

```bash
# Terminal 1: MCP server on 127.0.0.1:8000
cd medical_mcp && python server.py

# Terminal 2: the app
streamlit run app.py
```

Optional settings (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Answering model |
| `OLLAMA_NUM_CTX` | `8192` | Context window |
| `MEDGEMMA_MODEL` | `medgemma:4b` | Report reader for photos and scans |
| `MCP_ROUTER_MODEL` | `qwen2.5:7b-instruct` | Model that picks the public source |
| `MCP_ROUTING` | `model` | `model` (falls back to keyword rules) or `rules` |

Linked records (a read-only prototype that stands in for a hospital record system) need a one-time Google Cloud setup, described in [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md).

## Tests

```bash
python -m pytest -q
```

277 tests cover saving and deleting data (100), question handling (62), reading reports (47), range checks (42), record sign-in (19) and photo privacy (7).

## Evaluation

All evaluation data is synthetic. `make_dataset.py` writes 5 lab reports from a ground-truth table (40 labelled values) and renders each one 4 ways: digital PDF, scanned PDF, clean image and phone photo. Run every script from the project root.

| What was tested | Script | Result |
|---|---|---|
| Reading reports (160 values) | `eval_documents.py` | MediAssist vs OCR only: digital 100% vs 98%, scanned 90% vs 98%, clean image 80% vs 68%, phone photo 55% vs 43% |
| Range judgement (40 values) | `eval_models.py` | Correct verdicts: Llama 3.2 3B 15, Qwen 2.5 7B 29, Llama 3 8B 36, code 40 of 40 |
| Source selection (40 questions) | `eval_routing.py`, `eval_models.py` | Correct source: Qwen 39, Llama 3.2 3B 35, keyword rules 20, Llama 3 8B 0 (no usable answer) of 40 |
| Answers (20 scenarios) | `run_per_model.py` | Passed: Qwen 18, Llama 3 8B 14, Llama 3.2 3B 13 of 20 |
| Unanswerable questions (20) | `run_per_model.py`, then reviewed by hand | Answered honestly: Qwen 17, Llama 3 8B 16, Llama 3.2 3B 10 of 20 (`results/hallucination_reviewed.json`) |
| Invented numbers | `eval_numbers.py` | 0 answers with a number not in the prompt, for all three models |
| Warm answer time | `eval_timing.py` | Mean: Llama 3.2 3B 1.85 s, Llama 3 8B 3.63 s, Qwen 3.89 s |
| Retrieval (16 scenarios) | `eval_retrieval.py` | 15 of 16 |
| Privacy (20 checks) | `eval_privacy.py` | 19 of 20 (a name that is also a common word can pass as a keyword) |
| Model footprint | `eval_footprint.py` | Disk, memory, load time and speed per model |

Across the 116 scenarios (16 retrieval, 20 privacy, 40 source selection, 20 answers, 20 unanswerable), 108 pass (93%).

Saved outputs are in `experiments/eval/results/`, with a `.log` of each run next to the scripts. Earlier versions of the scenario sets are kept in `results/v1_13scenarios/` and `results/v1_28q/`.

## Limitations

- Prompt rules can be ignored: all three models broke a safety rule in at least one scenario. Advice requests should be blocked in code.
- The best reader depends on the input: OCR is better on scans, MediAssist on photos.
- A patient name that is also a common word may pass the keyword filter.
- Synthetic data only. Clinical usefulness has not been evaluated with nurses.
- Record access is a prototype, not an authorised hospital system integration.

## References

1. Singhal et al., *Nature Medicine*, 2025
2. Chen et al., MEDITRON-70B, arXiv:2311.16079, 2023
3. Sellergren et al., MedGemma technical report, arXiv:2507.05201, 2025
4. Amugongo et al., *PLOS Digital Health*, 2025
5. Zhao et al., MedRAG, WWW '25, 2025
6. Model Context Protocol specification, 2026-07-28
7. PDPC, Advisory guidelines for the healthcare sector, 2023
8. Health Information Act, Singapore, 2026
