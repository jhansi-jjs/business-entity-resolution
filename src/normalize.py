"""Normalization module for Business Entity Resolution Challenge.

Provides robust, deterministic normalization for business names and addresses,
handling multilingual scripts (English, Indic scripts, French), legal suffixes,
abbreviations, noise, and addresses without external lookup.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional
import unidecode


LEGAL_SUFFIX_MAP = {
    # English variations
    "pvt ltd": "private limited",
    "pvt limited": "private limited",
    "private ltd": "private limited",
    "pvtltd": "private limited",
    "p limited": "private limited",
    "ltd": "limited",
    "limited": "limited",
    "ltd liability co": "limited liability company",
    "llc": "limited liability company",
    "l l c": "limited liability company",
    "llp": "limited liability partnership",
    "l l p": "limited liability partnership",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "company": "company",
    "cie": "company",
    "pub ltd": "public limited",
    "public ltd": "public limited",
    "public limited": "public limited",

    # French legal entities
    "sarl": "sarl",
    "s a r l": "sarl",
    "sas": "sas",
    "s a s": "sas",
    "sasu": "sasu",
    "sci": "sci",
    "s c i": "sci",
    "sa": "sa",
    "s a": "sa",
    "snc": "snc",
    "eurl": "eurl",
    "gie": "gie",

    # Indic script legal terms (Hindi / Devanagari)
    "प्राइवेट लिमिटेड": "private limited",
    "प्राइवेट": "private",
    "लिमिटेड": "limited",
    "एलएलपी": "llp",
    "कंपनी": "company",
    "कॉर्पोरेशन": "corporation",
}

LEGAL_TERMS_PATTERN = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|public\s+limited|pub\s+ltd|"
    r"limited\s+liability\s+company|limited\s+liability\s+partnership|limited|ltd|"
    r"incorporated|inc|corporation|corp|company|llc|llp|sarl|sasu|sas|sci|eurl|snc|"
    r"प्राइवेट\s+लिमिटेड|प्राइवेट|लिमिटेड|एलएलपी|कंपनी|कॉर्पोरेशन"
    r")\b",
    re.IGNORECASE
)

ADDRESS_ABBREV_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bh\.no\b": "house number",
    r"\bhno\b": "house number",
    r"\bno\b": "number",
    r"\bplot\b": "plot",
    r"\bsec\b": "sector",
    r"\brue\b": "rue",
    r"\bbvd\b": "boulevard",
    r"\ball\b": "allee",
    r"\ballee\b": "allee",
}


def clean_unicode(text: str) -> str:
    """Normalize unicode to NFC to preserve combining marks in Indic and accented text."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text))
    return "".join(c for c in norm if unicodedata.category(c) != "Cs")


def normalize_text_general(text: str) -> str:
    """Basic lowercasing, unicode cleaning, noise stripping, and whitespace compaction."""
    if not text or str(text).lower() in ("nan", "none", "null"):
        return ""
    text = clean_unicode(str(text))
    text = text.lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"^[<\-#@*!]+\s*", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\bdba\b.*", "", text)
    
    # Strip punctuation while preserving letters, numbers, whitespace, and combining vowel marks (Mn, Mc)
    chars = [c if (c.isalnum() or unicodedata.category(c) in ("Mn", "Mc") or c.isspace()) else " " for c in text]
    text = "".join(chars)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(raw_name: str) -> Tuple[str, str, str]:
    """Normalize a business name.

    Returns:
        (name_normalized, name_core, name_ascii)
    """
    clean = normalize_text_general(raw_name)
    if not clean:
        return "", "", ""

    # Transliterated ASCII for cross-script comparison
    name_ascii = unidecode.unidecode(clean).strip()
    name_ascii = re.sub(r"\s+", " ", name_ascii)

    full_norm = clean
    for k, v in LEGAL_SUFFIX_MAP.items():
        pattern = r"\b" + re.escape(k) + r"\b"
        full_norm = re.sub(pattern, v, full_norm)
    full_norm = re.sub(r"\s+", " ", full_norm).strip()

    core_norm = LEGAL_TERMS_PATTERN.sub(" ", full_norm)
    core_norm = re.sub(r"\s+", " ", core_norm).strip()
    if not core_norm:
        core_norm = full_norm

    return full_norm, core_norm, name_ascii


def normalize_address(raw_address: str) -> Tuple[str, List[str]]:
    """Normalize a business address.

    Returns:
        (addr_normalized, numeric_tokens)
    """
    clean = normalize_text_general(raw_address)
    if not clean:
        return "", []

    addr_norm = clean
    for pat, rep in ADDRESS_ABBREV_MAP.items():
        addr_norm = re.sub(pat, rep, addr_norm)
    addr_norm = re.sub(r"\s+", " ", addr_norm).strip()

    numeric_tokens = re.findall(r"\b\d+\b", addr_norm)
    return addr_norm, numeric_tokens


def extract_postal_code(tokens: List[str], country: str) -> Optional[str]:
    """Extract candidate postal/PIN code based on country-specific length."""
    country = str(country).upper()
    if country in ("INDIA", "IN"):
        for t in tokens:
            if len(t) == 6:
                return t
    elif country in ("US", "USA"):
        for t in tokens:
            if len(t) == 5:
                return t
    elif country in ("FRANCE", "FR"):
        for t in tokens:
            if len(t) == 5:
                return t
    return None


if __name__ == "__main__":
    test_cases = [
        ("ABC Technologies Pvt. Ltd.", "Plot 12, MG Road, Bengaluru", "India"),
        ("ABC Technologies Private Limited", "12 M.G. Rd Bangalore", "India"),
        ("<< Team Ecole", "175 Boulevard du Président Franklin Roosevelt", "France"),
        ("Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE", "France"),
        ("-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC", "US"),
        ("wilfordhancock.com", "Mack Rd, Haltom City, Texas", "US"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI", "India"),
        ("आदित्य प्रॉपर्टीज एलएलपी", "G-3/571, GULMOHAR COLONY, BHOPAL", "India")
    ]

    print("=== Normalization Unit / Sanity Verification ===")
    for name, addr, cntry in test_cases:
        norm_n, core_n, ascii_n = normalize_name(name)
        norm_a, nums = normalize_address(addr)
        postal = extract_postal_code(nums, cntry)
        print(f"Original Name: {name}")
        print(f"  -> Norm:  '{norm_n}'")
        print(f"  -> Core:  '{core_n}'")
        print(f"  -> ASCII: '{ascii_n}'")
        print(f"Original Addr: {addr}")
        print(f"  -> Norm:  '{norm_a}'")
        print(f"  -> Nums:  {nums}, Postal: {postal}")
        print("-" * 50)
