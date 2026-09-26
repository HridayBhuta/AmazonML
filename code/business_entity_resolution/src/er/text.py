"""Offline text normalisation for business names and addresses (polars expressions only).

Everything here is rule-based and local: no geocoder, no registry, no hosted model.
Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
Malayalam) are romanised with a small hand-written table so that, e.g.,
"राम मार्केटिंग प्राइवेट लिमिटेड" can be compared with "Ram Marketing Private Limited".
A consonant "skeleton" (vowels and h removed, similar sounds merged) makes the two
spellings meet: both become "rm mrktng prvt lmtd".
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Indic -> Latin romanisation tables (offsets from the start of each Unicode block;
# the nine Brahmic blocks share the same ISCII-derived layout).
# ---------------------------------------------------------------------------
INDIC_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f", 0x5F: "y",
}
_VOWELS = {
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
    0x14: "au", 0x60: "ri", 0x61: "li",
}
_MATRAS = {
    0x3A: "e", 0x3B: "e", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4E: "e", 0x4F: "aw", 0x55: "e", 0x56: "ai", 0x57: "au", 0x62: "li", 0x63: "li",
}
_SIGNS = {0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA, _NUKTA, _AVAGRAHA = 0x4D, 0x3C, 0x3D
INDIC_RANGE = r"[\x{0900}-\x{0D7F}]"


def _dedupe(pats, reps):
    seen, p2, r2 = set(), [], []
    for p, r in zip(pats, reps):
        if p not in seen:
            seen.add(p)
            p2.append(p)
            r2.append(r)
    return p2, r2


def _indic_tables():
    removals = ["‌", "‍", " "]
    removals += [chr(b + _NUKTA) for b in INDIC_BASES] + [chr(b + _AVAGRAHA) for b in INDIC_BASES]
    # Phase 1: consonant + virama / consonant + vowel sign (two-character patterns; they never
    # overlap each other, so standard Aho-Corasick semantics are safe).
    p1, r1 = [], []
    for b in INDIC_BASES:
        for c, cv in _CONS.items():
            p1.append(chr(b + c) + chr(b + _VIRAMA))
            r1.append(cv)
            for m, mv in _MATRAS.items():
                p1.append(chr(b + c) + chr(b + m))
                r1.append(cv + mv)
    # Phase 2: single characters (bare consonant carries the inherent "a").
    p2, r2 = [], []
    for b in INDIC_BASES:
        for c, cv in _CONS.items():
            p2.append(chr(b + c))
            r2.append(cv + "a")
        for v, vv in _VOWELS.items():
            p2.append(chr(b + v))
            r2.append(vv)
        for m, mv in _MATRAS.items():
            p2.append(chr(b + m))
            r2.append(mv)
        for s, sv in _SIGNS.items():
            p2.append(chr(b + s))
            r2.append(sv)
        p2.append(chr(b + _VIRAMA))
        r2.append("")
        for d in range(10):
            p2.append(chr(b + 0x66 + d))
            r2.append(str(d))
    p2 += ["।", "॥"]
    r2 += [" ", " "]
    return removals, _dedupe(p1, r1), _dedupe(p2, r2)


_REMOVALS, (_P1, _R1), (_P2, _R2) = _indic_tables()


def romanise(e: pl.Expr) -> pl.Expr:
    """Romanise Indic characters; Latin text passes through unchanged."""
    return (e.str.replace_many(_REMOVALS, [""] * len(_REMOVALS))
             .str.replace_many(_P1, _R1)
             .str.replace_many(_P2, _R2))


def fold(e: pl.Expr) -> pl.Expr:
    """Romanise, strip accents (NFKD + drop combining marks) and lowercase."""
    return (romanise(e.fill_null(""))
            .str.normalize("NFKD")
            .str.replace_all(r"\p{M}+", "")
            .str.to_lowercase())


# ---------------------------------------------------------------------------
# Consonant skeleton: phonetic-ish key shared by romanised Indic and English spellings.
# ---------------------------------------------------------------------------
_DOUBLE = [c + c for c in "bcdfgjklmnprstvxz0123456789"]
_SINGLE = [c for c in "bcdfgjklmnprstvxz0123456789"]


def skeleton(e: pl.Expr) -> pl.Expr:
    return (e.str.replace_all("h", "")
             .str.replace_many(["ck", "c", "q", "x", "w", "f", "z", "y"],
                               ["k", "k", "k", "ks", "v", "p", "j", ""])
             .str.replace_all(r"[aeiou]", "")
             .str.replace_many(_DOUBLE, _SINGLE)
             .str.replace_many(_DOUBLE, _SINGLE)
             .str.replace_all(r"\s+", " ")
             .str.strip_chars())


# ---------------------------------------------------------------------------
# Hand-written vocabularies (generic domain knowledge, not business data).
# ---------------------------------------------------------------------------
NAME_MAP = {
    "incorporated": "inc", "incorporation": "inc", "limited": "ltd", "ltd": "ltd", "private": "pvt",
    "pvt": "pvt", "prv": "pvt", "corporation": "corp", "corpn": "corp", "company": "co", "compan": "co",
    "companies": "co", "cos": "co", "llc": "llc", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "service", "mgmt": "management", "bros": "brothers", "ctr": "center",
    "centre": "center", "grp": "group", "assoc": "associates", "assocs": "associates", "tech": "tech",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "limted": "ltd", "limitd": "ltd",
}
LEGAL = {
    "inc", "ltd", "pvt", "corp", "co", "llc", "llp", "lp", "pllc", "plc", "pc", "pa", "sa", "sas",
    "sasu", "sarl", "eurl", "sci", "snc", "selarl", "gmbh", "ag", "bv", "nv", "pty", "public", "the",
    "and", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d", "dba", "ms", "m", "s", "shri",
    "sri", "shree", "opc",
}
# Legal words as they appear after romanising Indic script (matched on the romanised token, plus a
# few long skeletons). Short skeletons are deliberately NOT used: they would delete real name
# tokens such as "sai", "om", "priya" or "lal".
INDIC_LEGAL_TOK = {"pra", "praiveta", "praivet", "privet", "praivheta", "li", "lim", "limiteda", "limited",
                   "kampani", "kampanee", "kanpani", "kampni", "karporeshana", "karporeshan", "inka", "inkaa",
                   "elaelapi", "elelapi", "elaelasi", "elelasi", "enda", "end", "aind", "emesa", "ema", "esa",
                   "shri", "shree", "shrii", "sri", "sree", "shrimati", "da", "tha"}
INDIC_LEGAL_SKEL = {"prvt", "pvt", "lmtd", "ltd", "kmpn", "knpn", "krprsn", "krprtn", "prlk", "pblk"}
STOP_TOKENS = ["null", "none", "nan", "n/a"]
STREET_TYPES = ["street", "road", "avenue", "boulevard", "drive", "lane", "court", "circle", "place", "parkway",
                "highway", "trail", "terrace", "square", "way", "rue", "chemin", "allee", "impasse", "route",
                "quai", "cours", "faubourg", "marg", "path", "north", "south", "east", "west", "northeast",
                "northwest", "southeast", "southwest", "no", "number", "unit", "suite", "apartment", "bis", "ter"]

_ADDR_GENERIC = {
    "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "cir": "circle", "pkwy": "parkway", "hwy": "highway", "trl": "trail", "ter": "terrace",
    "sq": "square", "apt": "apartment", "apts": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "opp": "opposite", "nr": "near", "sec": "sector", "sect": "sector",
    "ngr": "nagar", "dist": "district", "dt": "district", "tq": "taluk", "tal": "taluk",
    "po": "po", "pob": "po", "hno": "house", "mkt": "market", "ext": "extension", "extn": "extension",
    "indl": "industrial", "ind": "industrial", "estt": "estate", "cplx": "complex", "stn": "station",
    "univ": "university", "hosp": "hospital",
}
_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
_US_MULTI = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "district of columbia": "dc",
}
_US = {"st": "street", "str": "street", "ste": "suite", "n": "north", "s": "south", "e": "east",
       "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
       "mt": "mount", "ft": "fort", "av": "avenue", "bd": "boulevard", "pl": "place", **_US_STATES}
_IN = {
    "st": "street", "av": "avenue", "pl": "place", "bd": "boulevard", "hr": "haryana", "dl": "delhi",
    "mh": "maharashtra", "ka": "karnataka", "tn": "tamil nadu", "up": "uttar pradesh",
    "mp": "madhya pradesh", "ap": "andhra pradesh", "wb": "west bengal", "gj": "gujarat",
    "rj": "rajasthan", "pb": "punjab", "br": "bihar", "kl": "kerala", "od": "odisha",
    "orissa": "odisha", "ts": "telangana", "tg": "telangana", "jh": "jharkhand", "ch": "chandigarh",
    "ga": "goa", "hp": "himachal pradesh", "uk": "uttarakhand", "ua": "uttarakhand",
    "jk": "jammu kashmir", "as": "assam", "cg": "chhattisgarh", "ct": "chhattisgarh",
    "bangalore": "bengaluru", "bombay": "mumbai", "gurgaon": "gurugram", "calcutta": "kolkata",
    "madras": "chennai", "poona": "pune", "baroda": "vadodara", "mysore": "mysuru",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "pondicherry": "puducherry",
}
_FR = {"st": "saint", "ste": "sainte", "r": "rue", "av": "avenue", "ave": "avenue",
       "bd": "boulevard", "bld": "boulevard", "bvd": "boulevard", "pl": "place", "ch": "chemin",
       "che": "chemin", "chem": "chemin", "imp": "impasse", "all": "allee", "rte": "route",
       "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "crs": "cours", "res": "residence",
       "sq": "square"}
# Country-specific maps are keyed by the normalised country label; any other label (the
# country field is an open set) simply gets the generic map.
ADDR_BY_COUNTRY = {"us": _US, "india": _IN, "france": _FR}
COUNTRY_ALIASES = {"usa": "us", "united states": "us", "united states of america": "us",
                   "u s": "us", "u s a": "us", "in": "india", "ind": "india", "bharat": "india",
                   "fr": "france", "fra": "france", "republique francaise": "france"}
ADDR_STOP_SKEL = {"strt", "rd", "avn", "drv", "ln", "krt", "blvrd", "plk", "krkl", "prkv", "gv",
                  "trl", "trrk", "st", "nt", "prtmnt", "plr", "bldng", "nr", "ppst", "bnd",
                  "sktr", "ngr", "klny", "dstrkt", "tlk", "nd", "nrt", "st", "vst", "mn", "krs",
                  "blk", "ps", "plt", "nmbr", "s", "lt", "sp", "pk", "kmplks", "tvr", "r",
                  "kmn", "l", "mpss", "prns", "nl", "nn", "p", "rt", "mrg", "vr", "ps", "gl"}


def norm_country(e: pl.Expr) -> pl.Expr:
    c = (fold(e).str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())
    return c.replace(COUNTRY_ALIASES)


def _tokens(e: pl.Expr) -> pl.Expr:
    return (e.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
             .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(STOP_TOKENS))))


def normalise_frame(df: pl.DataFrame) -> pl.DataFrame:
    """df: entity_id, business_name, business_address, country (strings). Adds normalised columns."""
    name0, addr0 = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
    out = df.with_columns(
        country_n=norm_country(pl.col("country")),
        name_indic=name0.str.contains(INDIC_RANGE),
        addr_indic=addr0.str.contains(INDIC_RANGE),
        name_f=fold(name0)
        .str.replace_all(r"^\s*www\.", "")
        .str.replace_all(r"\.(com|net|org|co\.in|co\.uk|in|co|biz|info|io|fr|us)\b", " ")
        .str.replace_all(r"[.'’`]", "")
        .str.replace_all("&", " and ").str.replace_all(r"\+", " plus "),
        addr_f=fold(addr0).str.replace_all(r"['’`]", "").str.replace_all(r"[.]", " ")
        .str.replace_all(r"\b(\d+)(st|nd|rd|th)\b", "$1")
        .str.replace_all(r"\b(\d+)\s+(?:st|nd|rd|th)\s+(cross|main|floor|flr|fl|block|stage|phase|sector|lane)\b", "$1 $2")
        .str.replace_all(r"(\d)([a-z])", "$1 $2").str.replace_all(r"([a-z])(\d)", "$1 $2"),
    )
    # multi-word US state names -> codes (string level, before tokenising)
    us = pl.col("country_n") == "us"
    padded = pl.lit(" ") + pl.col("addr_f").str.replace_all(r"[^\p{L}\p{N}]+", " ") + pl.lit(" ")
    out = out.with_columns(addr_f=pl.when(us).then(
        padded.str.replace_many([f" {k} " for k in _US_MULTI], [f" {v} " for v in _US_MULTI.values()]))
        .otherwise(pl.col("addr_f")))
    name_tok = _tokens(pl.col("name_f")).list.eval(pl.element().replace(NAME_MAP))
    addr_tok = _tokens(pl.col("addr_f")).list.eval(pl.element().replace(_ADDR_GENERIC))
    for country, mapping in ADDR_BY_COUNTRY.items():
        addr_tok = pl.when(pl.col("country_n") == country).then(
            addr_tok.list.eval(pl.element().replace(mapping))).otherwise(addr_tok)
    out = out.with_columns(name_tok=name_tok, addr_tok=addr_tok)
    core_latin = pl.col("name_tok").list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL))))
    core_indic = core_latin.list.eval(pl.element().filter(~pl.element().is_in(list(INDIC_LEGAL_TOK))
                                                          & ~skeleton(pl.element()).is_in(list(INDIC_LEGAL_SKEL))))
    core = pl.when(pl.col("name_indic")).then(core_indic).otherwise(core_latin)
    out = out.with_columns(core_tok=core)
    out = out.with_columns(core_tok=pl.when(pl.col("core_tok").list.len() == 0)
                           .then(pl.col("name_tok")).otherwise(pl.col("core_tok")))
    out = out.with_columns(
        name_norm=pl.col("name_tok").list.join(" "),
        core=pl.col("core_tok").list.join(" "),
        core_sorted=pl.col("core_tok").list.unique().list.sort().list.join(" "),
        core_compact=pl.col("core_tok").list.join(""),
        addr_norm=pl.col("addr_tok").list.join(" "),
    )
    out = out.with_columns(
        addr_tok=pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        core_tok=pl.col("core_tok").list.unique(),
        name_skel=skeleton(pl.col("core")),
        addr_skel=skeleton(pl.col("addr_norm")),
    )
    out = out.with_columns(
        addr_sorted=pl.col("addr_tok").list.unique().list.sort().list.join(" "),
        nums=pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))),
        house=pl.concat_str([pl.col("addr_norm").str.extract(r"\b(\d{1,6})\s+(?:[a-z]\s+)?[a-z]{3}", 1),
                             pl.col("addr_norm").str.extract(r"\b\d{1,6}\s+(?:[a-z]\s+)?([a-z]{3})", 1)],
                            separator=" "),
        house2=pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(STREET_TYPES)))
        .list.join(" ").str.extract(r"\b(\d{1,6} [a-z]{3,})\b", 1),
        name_skel_tok=pl.col("name_skel").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(),
        addr_skel_tok=pl.col("addr_skel").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(),
    )
    out = out.with_columns(
        postcodes=pl.col("nums").list.eval(pl.element().filter(pl.element().str.len_chars().is_between(5, 6))).list.unique(),
        first_num=pl.col("nums").list.first(),
        nums=pl.col("nums").list.unique(),
        addr_tok=pl.col("addr_tok").list.unique(),
    )
    return out.select(
        "entity_id", "country_n", "name_indic", "addr_indic", "name_norm", "core", "core_sorted",
        "core_compact", "core_tok", "name_skel", "name_skel_tok", "addr_norm", "addr_sorted", "addr_tok",
        "addr_skel", "addr_skel_tok", "nums", "first_num", "postcodes", "house", "house2",
    )
