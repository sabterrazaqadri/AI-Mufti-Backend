"""Tone/grounding eval for the answer modes (HUKM / NASEEHAT / gentle refusal).

Posts a fixed set of prompts to a running backend's /chat endpoint and checks each
reply against what its category must (and must not) do:

  hukm     — an obscure ruling question with no source in the library: must use the
             gentle refusal (a refusal marker) and must carry no citation.
  counsel  — a personal / akhlaqi problem: must NOT be a refusal and must NOT quote
             an ayat/hadith or name a book from memory.
  grounded — a well-covered mas'ala: must come back with sources.
  scope    — non-Islamic: the out-of-scope reply.
  greet    — salam: answered normally, no refusal.

The automatic checks are a floor, not the verdict — read every answer by eye too.

Usage:  python -I eval_tone.py [base_url]     (default http://127.0.0.1:8000)
"""
import base64
import json
import re
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

CASES = [
    ("hukm", "Kya chaand par namaz parhte waqt qibla ka rukh zameen ki taraf hoga ya Makkah ki taraf?"),
    ("hukm", "Agar koi drone se qurbani ka janwar zibah kare to kya qurbani ho jayegi?"),
    ("hukm", "کیا ورچوئل رئیلٹی میں حج کی مشق کرنے سے کوئی ثواب ملتا ہے؟"),
    ("hukm", "Is cryptocurrency staking reward halal according to Hanafi fiqh?"),
    ("counsel", "Mujhe bohat ghussa aata hai, ghar walon par chilla deta hun phir pachtata hun. Kya karun?"),
    ("counsel", "Meri saas aur biwi me roz jhagra hota hai, main beech me pis raha hun"),
    ("counsel", "میرا دل بہت اداس رہتا ہے، کسی کام میں دل نہیں لگتا"),
    ("counsel", "My brother betrayed me in a business deal. How do I forgive him?"),
    ("counsel", "Mera beta namaz nahi parhta, daantne se aur door ho jata hai"),
    ("counsel", "Future ki bohat fikr rehti hai, naukri nahi mil rahi, himmat toot rahi hai"),
    ("counsel", "Main ek acha insaan banna chahta hun, kahan se shuru karun?"),
    ("counsel", "Dost ne meri ghibat ki aur mujhe pata chal gaya, ab us se kaise milun?"),
    ("grounded", "Wuzu ke faraiz kitne hain?"),
    ("grounded", "Namaz-e-musafir me qasr kab hota hai?"),
    ("scope", "NASA ne mars par kya bheja hai?"),
    ("greet", "Assalam o alaikum"),
]

REFUSAL_MARKERS = (
    "مستند حوالہ نہیں", "mustanad hawala nahi", "authentic reference on this matter",
)
SCOPE_MARKERS = (
    "صرف اسلامی مسائل", "sirf islami masail", "only have knowledge about islamic matters",
    "only cover islamic matters",
)
# A specific scripture/book attribution — forbidden in counsel and in refusals.
CITATION_RE = re.compile(
    r"jild|hissa|safha|bukhari|muslim shareef|sahih muslim|tirmizi|surah|sura\b|"
    r"آیت|حدیث|جلد|صفحہ|بخاری|سورۃ|سورہ|\(\d+:\d+\)|\b\d+:\d+\b|"
    r"bahar-e-shariat|fatawa|فتاوی|بہار شریعت",
    re.IGNORECASE,
)


def ask(text: str):
    req = urllib.request.Request(
        f"{BASE}/chat",
        data=json.dumps({"content": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw_sources = resp.headers.get("X-Sources")
        body = resp.read().decode("utf-8", errors="replace")
    # The stream ends with "\x1e" + base64(JSON) of the sources actually cited;
    # the header (every retrieved passage) is only the fallback.
    body, _, trailer = body.partition("\x1e")
    raw = trailer or raw_sources
    sources = json.loads(base64.b64decode(raw)) if raw else []
    return body.strip(), sources


def has(text: str, markers) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in markers)


def arabic_share(text: str) -> float:
    """Fraction of letters in Arabic script (Urdu/Arabic). A few du'a lines in an
    English answer are expected; a mostly-Urdu answer to English is the bug."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters)


ENGLISH_HINTS = {"the", "how", "do", "i", "my", "is", "what", "him", "me"}


def is_english(text: str) -> bool:
    return len(ENGLISH_HINTS & set(re.findall(r"[a-z]+", text.lower()))) >= 3


def check(cat: str, prompt: str, answer: str, sources) -> list:
    problems = []
    refused, cited = has(answer, REFUSAL_MARKERS), bool(CITATION_RE.search(answer))
    if cat == "hukm":
        # Retrieval sometimes does find a passage; then a grounded answer is correct,
        # and only an unsourced answer must be the refusal.
        if not sources:
            if not refused:
                problems.append("expected the gentle refusal (no marker found)")
            if cited:
                problems.append("refusal carries a citation")
    elif cat == "counsel":
        if refused or has(answer, SCOPE_MARKERS):
            problems.append("counsel answered with a refusal/out-of-scope reply")
        if cited and not sources:
            problems.append("cites ayat/hadith/book with no source supplied")
        if len(answer) < 200:
            problems.append("suspiciously short for counsel")
        if is_english(prompt) and arabic_share(answer) > 0.4:
            problems.append("English question answered in Urdu/Arabic script")
    elif cat == "grounded":
        if not sources:
            problems.append("no sources returned")
        if refused:
            problems.append("refused a grounded mas'ala")
    elif cat == "scope":
        if not has(answer, SCOPE_MARKERS):
            problems.append("expected the out-of-scope reply")
    elif cat == "greet":
        if refused or has(answer, SCOPE_MARKERS):
            problems.append("refused a greeting")
    return problems


def main() -> int:
    # Windows consoles default to cp1252, which cannot print Urdu.
    sys.stdout.reconfigure(encoding="utf-8")
    failures = 0
    for i, (cat, prompt) in enumerate(CASES, 1):
        try:
            answer, sources = ask(prompt)
        except Exception as exc:  # network / 5xx — report and carry on
            print(f"\n[{i}] {cat.upper()} — ERROR: {exc}")
            failures += 1
            continue
        problems = check(cat, prompt, answer, sources)
        failures += bool(problems)
        status = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"\n{'=' * 78}\n[{i}] {cat.upper()} — {status}\nQ: {prompt}\n"
              f"sources: {len(sources)} | chars: {len(answer)}\n{'-' * 78}\n{answer}")
        time.sleep(4)  # stay under the /chat 20-per-minute limit
    print(f"\n{'=' * 78}\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
