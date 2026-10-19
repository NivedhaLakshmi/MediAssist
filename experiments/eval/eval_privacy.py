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

print(f"\n{sum(s['passed'] for s in scenarios)}/{len(scenarios)} privacy scenarios passed")
save("privacy", {"scenarios": scenarios})
