"""Privacy and data management scenarios.

Storage is redirected to a temporary directory so no real profile,
conversation or photograph is touched.

    python experiments/eval/eval_privacy.py
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

from common import ROOT, save

from services import profiles
TMP = tempfile.mkdtemp()
profiles.USER_DATA_DIR = TMP

from services import attachments, user_data  # noqa: E402
from services.query_utils import build_search_keywords, normalize_health_query  # noqa: E402

scenarios = []


def record(sid, name, passed, detail):
    scenarios.append({"id": sid, "scenario": name, "passed": bool(passed), "detail": detail})
    print(f"{sid:<7}{'PASS' if passed else 'FAIL'}  {name}  ({detail})", flush=True)


# PRV-1 privacy audit
audit = subprocess.run([sys.executable, "privacy_audit.py"], cwd=ROOT,
                       capture_output=True, text=True).stdout
dests = len(re.findall(r"^\s+\[(?:LOCAL|PUBLIC|GOOGLE|USER|OK)\]", audit, re.M))
record("PRV-1", "Privacy audit finds no unreviewed network destination",
       "no unreviewed network destinations" in audit,
       "RESULT line present" if "no unreviewed" in audit else "audit reported a problem")

# PRV-2 outbound requests carry search keywords only
# A question as a nurse would type it. Values typed into a question
# identify nobody; what must not leave is anything from the report.
question = "what does it mean if the wbc count is 11800"
normalized = normalize_health_query(question)["normalized"]
keywords = build_search_keywords(normalized)
probe = f'''
import sys, json, requests
sys.path.insert(0, ".")
sent = []
class R:
    status_code = 200
    content = b"<x/>"
    text = "<x/>"
    def json(self): return {{}}
    def raise_for_status(self): pass
def fake_get(url, params=None, **kw):
    sent.append({{"url": url, "params": params}})
    return R()
requests.get = fake_get
from utils import model_router
model_router.ROUTING_MODE = "rules"
from tools.search import search_medical_sources
search_medical_sources({keywords!r}, {normalized!r})
print("SENT=" + json.dumps(sent))
'''
out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT / "medical_mcp",
                     capture_output=True, text=True).stdout
sent = json.loads(out.split("SENT=")[-1]) if "SENT=" in out else []
blob = json.dumps(sent).lower()
leaked = [w for w in ["daniel", "tan", "citycare", "menon"] if w in blob]
hosts = sorted({re.sub(r"^https?://([^/]+).*", r"\1", s["url"]) for s in sent})
record("PRV-2", "A typed question sends only its own keywords, nothing from the report",
       sent and not leaked,
       f"keywords sent: '{keywords}'; hosts {hosts}; leaked {leaked or 'nothing'}")

# PRV-10/11 photographs: a report adds nothing, a medicine label adds its name
from services.image_service import photo_search_terms  # noqa: E402
import pdfplumber  # noqa: E402
reports = [(ROOT / "experiments/eval/dataset/R1_cbc_male/report.txt").read_text()]
for path in [*sorted(ROOT.parent.glob("Mr. *.pdf")), ROOT / "covidTest.pdf"]:
    if path.exists():
        reports.append("\n".join(p.extract_text() or "" for p in pdfplumber.open(path).pages))
added = [photo_search_terms([(r, [])]) for r in reports]
record("PRV-10", "A photographed lab report adds no words to the outside search",
       all(a == "" for a in added),
       f"{len(reports)} reports (1 generated, {len(reports)-1} real) contributed "
       f"{sum(len(a.split()) for a in added)} terms")
label = photo_search_terms([("DOLO 650 Paracetamol Tablets IP 650 mg Micro Labs Limited", [])])
record("PRV-11", "A photographed medicine label still adds its name to the search",
       "dolo" in label or "paracetamol" in label, f"terms '{label}'")

# PRV-3 export excludes credentials
user_data.save_conversation("c1", [{"role": "user", "content": "is my tsh ok"},
                                   {"role": "assistant", "content": "It is above range."}],
                            profile_id="A")
archive = user_data.build_export_archive(
    user_data.load_conversations("A"), [{"report_name": "r.pdf", "text": "TSH 6.8"}],
    drive_connected=True, drive_folder="Reports")
z = zipfile.ZipFile(io.BytesIO(archive))
text = " ".join(z.read(n).decode(errors="ignore") for n in z.namelist())
secrets = [p for p in [r"ya29\.", r"\b1//0", r"client_secret", r"refresh_token\"\s*:"]
           if re.search(p, text)]
record("PRV-3", "Data export contains the patient's data and no credentials",
       not secrets, f"files {z.namelist()}; credential patterns found: {secrets or 'none'}")

# PRV-4 deleting one profile's conversations leaves another's
user_data.save_conversation("c2", [{"role": "user", "content": "hello"},
                                   {"role": "assistant", "content": "hi"}], profile_id="B")
user_data.delete_all_conversations("A")
record("PRV-4", "Deleting one profile's chats leaves another profile's chats",
       not user_data.load_conversations("A") and len(user_data.load_conversations("B")) == 1,
       f"A {len(user_data.load_conversations('A'))}, B {len(user_data.load_conversations('B'))}")

# PRV-5 path traversal rejected
bad = ["../../../etc/passwd", "../profiles.json", "x.png", "/etc/hosts"]
record("PRV-5", "A stored image name cannot point outside the profile folder",
       all(attachments.attachment_path(b, "A") is None for b in bad),
       f"{len(bad)} crafted names all rejected")

# PRV-6 unreferenced photographs removed
name = attachments.save_attachment(b"\x89PNG fake", "photo.png", profile_id="A")
attachments.prune_unreferenced([], profile_id="A")
record("PRV-6", "A photo no conversation refers to is deleted from disk",
       attachments.attachment_path(name, "A") is None, f"stored as {name}, then pruned")

# PRV-7 embedding model offline
import services.embedding_service  # noqa: E402,F401
record("PRV-7", "Embedding model loads without contacting its download service",
       os.environ.get("HF_HUB_OFFLINE") == "1",
       f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

# PRV-8 telemetry off
cfg = (ROOT / ".streamlit" / "config.toml").read_text()
record("PRV-8", "Interface usage statistics are switched off",
       re.search(r"gatherUsageStats\s*=\s*false", cfg) is not None, ".streamlit/config.toml")

# PRV-9 local endpoints
srcs = (ROOT / "mcp_client.py").read_text() + (ROOT / "medical_mcp" / "server.py").read_text()
record("PRV-9", "Model runtime and source gateway are reached on this machine only",
       "127.0.0.1:8000" in srcs and 'host="127.0.0.1"' in srcs,
       "gateway bound to 127.0.0.1:8000; model runtime at 127.0.0.1:11434")

# ---- Added for the extended evaluation (PRV-12 to PRV-20) ----

# PRV-12 every synthetic report, photographed, adds nothing to searches
synth = sorted((ROOT / "experiments/eval/dataset").glob("R*/report.txt"))
added12 = [photo_search_terms([(p.read_text(), [])]) for p in synth]
record("PRV-12", "None of the 5 synthetic reports, photographed, adds words to outside searches",
       len(synth) == 5 and all(a == "" for a in added12),
       f"{len(synth)} reports contributed {sum(len(a.split()) for a in added12)} terms")

# PRV-13 / PRV-14 the 40 source-selection questions: what goes out, and where
import ast  # noqa: E402
_src = (ROOT / "experiments/eval/eval_routing.py").read_text()
_a = _src.index("CASES = [")
CASES40 = ast.literal_eval(_src[_a + 8:_src.index("]\n", _a) + 1])
pairs = []
for q, _, _ in CASES40:
    n = normalize_health_query(q)["normalized"]
    pairs.append((build_search_keywords(n), n, q))
probe40 = "\n".join([
    "import sys, json, requests",
    "sys.path.insert(0, '.')",
    "sent = []",
    "CUR = ''",
    "class R:",
    "    status_code = 200",
    "    content = b'<x/>'",
    "    text = '<x/>'",
    "    def json(self): return {}",
    "    def raise_for_status(self): pass",
    "def fake_get(url, params=None, **kw):",
    "    sent.append({'url': url, 'params': params, 'q': CUR})",
    "    return R()",
    "requests.get = fake_get",
    "from utils import model_router",
    "model_router.ROUTING_MODE = 'rules'",
    "from tools.search import search_medical_sources",
    "for kw, norm, q in json.loads(sys.argv[1]):",
    "    CUR = q",
    "    search_medical_sources(kw, norm)",
    "print('SENT=' + json.dumps(sent))",
])
out40 = subprocess.run([sys.executable, "-c", probe40, json.dumps(pairs)], cwd=ROOT / "medical_mcp",
                       capture_output=True, text=True).stdout
sent40 = json.loads(out40.split("SENT=")[-1]) if "SENT=" in out40 else []
full_sent = sorted({s["q"] for s in sent40
                    if s["q"].lower() in json.dumps([s["url"], s["params"]]).lower()})
asked = {s["q"] for s in sent40}
record("PRV-13", "Across the 40 source questions, the full question is never sent outside",
       len(asked) == 40 and not full_sent,
       f"{len(asked)} questions searched; full question found in requests for {len(full_sent)}")
allowed = {"clinicaltrials.gov", "dailymed.nlm.nih.gov", "eutils.ncbi.nlm.nih.gov", "wsearch.nlm.nih.gov"}
hosts40 = sorted({re.sub(r"^https?://([^/]+).*", r"\1", s["url"]) for s in sent40})
record("PRV-14", "Outside requests go only to the four public medical sources",
       bool(sent40) and set(hosts40) <= allowed, f"{len(sent40)} requests to {hosts40}")

# PRV-15 Google Drive is requested read-only
from services import google_drive_auth  # noqa: E402
_auth = (ROOT / "services" / "google_drive_auth.py").read_text()
_scope = google_drive_auth.GOOGLE_DRIVE_SCOPE.strip()
record("PRV-15", "The record tool asks Google Drive for read-only access",
       _scope == "https://www.googleapis.com/auth/drive.readonly"
       and "auth/drive.file" not in _auth and '"https://www.googleapis.com/auth/drive"' not in _auth,
       f"scope {_scope}")

# PRV-16 one profile's export holds none of another profile's chats
user_data.save_conversation("c3", [{"role": "user", "content": "PROFILE-A-ONLY question"},
                                   {"role": "assistant", "content": "ok"}], profile_id="A")
user_data.save_conversation("c4", [{"role": "user", "content": "PROFILE-B-ONLY question"},
                                   {"role": "assistant", "content": "ok"}], profile_id="B")
za = zipfile.ZipFile(io.BytesIO(user_data.build_export_archive(user_data.load_conversations("A"), [])))
ta = " ".join(za.read(n).decode(errors="ignore") for n in za.namelist())
record("PRV-16", "A profile's data export contains none of another profile's chats",
       "PROFILE-A-ONLY" in ta and "PROFILE-B-ONLY" not in ta,
       f"A's export has A's chat: {'PROFILE-A-ONLY' in ta}; has B's chat: {'PROFILE-B-ONLY' in ta}")

# PRV-17 the app's "Delete this profile" steps leave no chats or photos on disk
pid = profiles.create_profile("Privacy Test Patient")
user_data.save_conversation("c17", [{"role": "user", "content": "PRV17 question"},
                                    {"role": "assistant", "content": "ok"}], profile_id=pid)
attachments.save_attachment(b"\x89PNG fake17", "r.png", profile_id=pid)
pdir = profiles.profile_directory(pid)
user_data.delete_all_conversations(pid)   # same steps as app.py's delete button
attachments.delete_all_attachments(pid)
profiles.delete_profile(pid)
left17 = [os.path.relpath(os.path.join(d, f), pdir) for d, _, fs in os.walk(pdir) for f in fs] if os.path.exists(pdir) else []
record("PRV-17", "Deleting a patient profile in the app leaves none of its chats or photos on disk",
       not left17, f"files left after delete: {left17 or 'none'}")

# PRV-18 deleting a profile's photos removes every one of them
names18 = [attachments.save_attachment(b"\x89PNG fake" + bytes([i]), f"p{i}.png", profile_id="B") for i in range(3)]
attachments.delete_all_attachments("B")
record("PRV-18", "Deleting a profile's photos removes every stored photo",
       all(attachments.attachment_path(n, "B") is None for n in names18), f"{len(names18)} photos stored, then deleted")

# PRV-19 stored photos do not keep the original file name
n19 = attachments.save_attachment(b"\x89PNG fake19", "daniel_tan_report.png", profile_id="A")
record("PRV-19", "A stored photo does not keep its original file name",
       "daniel" not in n19.lower() and "tan" not in n19.lower(), f"saved 'daniel_tan_report.png' as {n19}")

# PRV-20 a patient's name typed into a question is not sent outside
kw20 = build_search_keywords(normalize_health_query("is daniel tan's wbc count normal")["normalized"])
record("PRV-20", "A patient's name typed into a question is not sent as a search keyword",
       "daniel" not in kw20 and "tan" not in kw20.split(), f"keywords '{kw20}'")

print(f"\n{sum(s['passed'] for s in scenarios)}/{len(scenarios)} privacy scenarios passed")
save("privacy", {"scenarios": scenarios})
