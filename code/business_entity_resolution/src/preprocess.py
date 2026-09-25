"""
ML Challenge 2026 - Business Entity Resolution
Data Preprocessing and Text Normalization Module

Handles:
- Legal entity suffix normalization (Corp/Corporation, Pvt/Private, Ltd/Limited, etc.)
- Punctuation & symbol standardization (& vs 'and', hyphens, etc.)
- Street and address abbreviation expansion (Rd -> Road, St -> Street, etc.)
- Unicode transliteration & accent stripping (NFKD)
- Postal / PIN code extraction (India, US, France)
- Phonetic encoding (Soundex and simplified Metaphone)
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set


# Legal business suffix normalization mapping
# Map common variations to canonical short forms or remove
LEGAL_SUFFIX_MAP = {
    r"\bprivate\s+limited\b": "pvt ltd",
    r"\bpvt\s*\.?\s*ltd\b": "pvt ltd",
    r"\blimited\s+liability\s+company\b": "llc",
    r"\blimited\s+liability\s+partnership\b": "llp",
    r"\bincorporated\b": "inc",
    r"\bcorporation\b": "corp",
    r"\bcorp\b": "corp",
    r"\blimited\b": "ltd",
    r"\bltd\b": "ltd",
    r"\bcompany\b": "co",
    r"\bco\b": "co",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bgmbh\b": "gmbh",
    r"\bs\.?a\.?r\.?l\.?\b": "sarl",
    r"\bs\.?a\.?s\.?\b": "sas",
    r"\bs\.?a\.?\b": "sa",
    r"\bpty\s+ltd\b": "pty ltd",
    r"\benterprises\b": "ent",
    r"\bservices\b": "svc",
}

# Legal suffixes set for stripping when isolating core business name
LEGAL_SUFFIXES_SET = {
    "inc", "corp", "corporation", "incorporated", "llc", "llp", "ltd", "limited",
    "pvt", "private", "co", "company", "gmbh", "sarl", "sas", "sa", "plc",
    "ent", "enterprises", "svc", "services", "center", "centre", "group"
}

# Address abbreviation mapping
ADDRESS_ABBR_MAP = {
    r"\brd\b": "road",
    r"\brd\.\b": "road",
    r"\bst\b": "street",
    r"\bst\.\b": "street",
    r"\bstr\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bave\.\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bdr\.\b": "drive",
    r"\bln\b": "lane",
    r"\bln\.\b": "lane",
    r"\bpkwy\b": "parkway",
    r"\bpky\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bterr\b": "terrace",
    r"\bter\b": "terrace",
    r"\bapt\b": "apartment",
    r"\bapts\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bdept\b": "department",
    r"\bbldg\b": "building",
    r"\bctr\b": "center",
    r"\bno\b": "number",
    r"\bno\.\b": "number",
    r"\bopp\b": "opposite",
    r"\bopp\.\b": "opposite",
    r"\bnr\b": "near",
    r"\bnr\.\b": "near",
}

# Common US state abbreviation map for normalization
US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"
}

# Compiled regex patterns for speed
RE_AMPERSAND = re.compile(r"&")
RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
RE_EXTRA_SPACES = re.compile(r"\s+")
RE_DOMAIN = re.compile(r"^(https?:\/\/)?(www\.)?([a-zA-Z0-9_\-\.]+)\.(com|org|net|in|us|fr|biz|info|co|io)$")

# Postal code patterns
# India: 6 digits (e.g. 560001, 700048)
# US: 5 digits, optionally followed by 4 digits (e.g. 90210, 10001-1234)
# France: 5 digits (e.g. 75008, 69001)
RE_POSTAL_IN = re.compile(r"\b([1-9][0-9]{5})\b")
RE_POSTAL_US_FR = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")


def strip_accents(text: str) -> str:
    """Normalize unicode characters and strip accents/diacritics."""
    if not text:
        return ""
    # NFKD decomposition separates base characters from combining diacritical marks
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_domain(text: str) -> str:
    """If text looks like a domain name (e.g. applecorp.com), extract words."""
    m = RE_DOMAIN.match(text.strip().lower())
    if m:
        domain_body = m.group(3)
        # replace dots, hyphens with spaces
        return re.sub(r"[\.\-_]", " ", domain_body)
    return text


def normalize_business_name(name: Optional[str]) -> str:
    """
    Standardize business names:
    - Unicode accent stripping and lowercasing
    - Domain name unpacking
    - Ampersand -> 'and'
    - Legal suffix normalization
    - Punctuation removal and whitespace collapsing
    """
    if not name or not isinstance(name, str):
        return ""

    text = strip_accents(name).lower()
    text = clean_domain(text)
    text = RE_AMPERSAND.sub(" and ", text)

    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_EXTRA_SPACES.sub(" ", text).strip()
    return text


def get_core_name_tokens(cleaned_name: str) -> List[str]:
    """Return name tokens excluding legal suffixes."""
    tokens = cleaned_name.split()
    return [t for t in tokens if t not in LEGAL_SUFFIXES_SET and len(t) > 1]


def normalize_address(address: Optional[str]) -> str:
    """
    Standardize business addresses:
    - Unicode accent stripping and lowercasing
    - Street abbreviation expansion (rd -> road, st -> street)
    - Punctuation removal
    - Whitespace collapsing
    """
    if not address or not isinstance(address, str):
        return ""

    text = strip_accents(address).lower()
    text = RE_AMPERSAND.sub(" and ", text)

    # Expand abbreviations
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_EXTRA_SPACES.sub(" ", text).strip()
    return text


def extract_postal_code(address: Optional[str], country: Optional[str] = None) -> Optional[str]:
    """
    Extract postal / PIN code from address text.
    Handles India 6-digit, US 5-digit, and France 5-digit codes.
    """
    if not address or not isinstance(address, str):
        return None

    c = (country or "").upper().strip()
    if c == "INDIA":
        m = RE_POSTAL_IN.search(address)
        if m:
            return m.group(1)
    elif c in ("US", "FRANCE"):
        m = RE_POSTAL_US_FR.search(address)
        if m:
            return m.group(1)
    else:
        # Generic fallback
        m_in = RE_POSTAL_IN.search(address)
        if m_in:
            return m_in.group(1)
        m_us = RE_POSTAL_US_FR.search(address)
        if m_us:
            return m_us.group(1)
    return None


def soundex(token: str) -> str:
    """
    Compute classic Soundex phonetic code for a token.
    Maps phonetic equivalents to the same code (e.g. Smith -> S530).
    """
    token = re.sub(r"[^a-zA-Z]", "", token).upper()
    if not token:
        return "0000"

    soundex_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }

    first = token[0]
    encoded = [first]
    prev_code = soundex_map.get(first, "0")

    for char in token[1:]:
        code = soundex_map.get(char, "0")
        if code != "0" and code != prev_code:
            encoded.append(code)
            prev_code = code
        elif code == "0":
            prev_code = "0"
        if len(encoded) == 4:
            break

    while len(encoded) < 4:
        encoded.append("0")

    return "".join(encoded[:4])


def simplified_metaphone(token: str) -> str:
    """
    Compute a fast simplified phonetic metaphone code.
    Normalizes silent letters, soft consonants, and vowels.
    """
    token = re.sub(r"[^a-zA-Z]", "", token).upper()
    if not token:
        return ""

    # Drop silent initials
    if token.startswith(("KN", "GN", "PN", "WR", "PS")):
        token = token[1:]
    elif token.startswith("X"):
        token = "S" + token[1:]

    # Initial sound
    res = [token[0]]

    # Phonetic substitutions
    t = token
    t = re.sub(r"PH", "F", t)
    t = re.sub(r"CK", "K", t)
    t = re.sub(r"SCH", "SK", t)
    t = re.sub(r"TH", "0", t)
    t = re.sub(r"TCH", "CH", t)
    t = re.sub(r"SH", "X", t)
    t = re.sub(r"CH", "X", t)
    t = re.sub(r"C([EIY])", r"S\1", t)
    t = re.sub(r"C", "K", t)
    t = re.sub(r"Q", "K", t)
    t = re.sub(r"Z", "S", t)
    t = re.sub(r"DG([EIY])", r"J\1", t)
    t = re.sub(r"G([EIY])", r"J\1", t)
    t = re.sub(r"GH", "", t)
    t = re.sub(r"W([AEIOU])", r"W\1", t)
    t = re.sub(r"W", "", t)
    t = re.sub(r"Y([AEIOU])", r"Y\1", t)
    t = re.sub(r"Y", "", t)

    # Remove consecutive duplicate letters
    dedup = []
    for c in t:
        if not dedup or c != dedup[-1]:
            dedup.append(c)
    cleaned = "".join(dedup)

    # Keep first vowel if initial, drop subsequent vowels
    vowels = {"A", "E", "I", "O", "U"}
    final = [cleaned[0]] if cleaned else []
    for c in cleaned[1:]:
        if c not in vowels:
            final.append(c)

    return "".join(final[:6])
