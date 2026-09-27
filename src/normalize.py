"""Text normalisation for business names and addresses.

All work is done with vectorised polars expressions; the only per-row Python call is
anyascii transliteration, applied only to rows containing non-ASCII characters
(Indic scripts, accented Latin).

Output columns per record:
    name_n   normalised full name (tokens joined by spaces, legal forms canonicalised)
    core_n   name with legal forms / generic filler words removed
    addr_n   normalised address (abbreviations canonicalised, NULL tokens removed)
    nums     list of numeric tokens found in the address (leading zeros stripped)
    skel     consonant skeleton of core_n (bridges vowel-less transliterations)
"""
import polars as pl
from anyascii import anyascii

# --- canonical forms -------------------------------------------------------------
# Legal / organisational forms (US, India, France and generic). Mapped to one canonical
# token; these tokens are then dropped to build the "core" name.
LEGAL = {
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "limited": "ltd", "ltd": "ltd", "lt": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "llc": "llc", "lllc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "public": "public", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "ei": "ei", "ets": "ets", "etablissement": "ets",
    "etablissements": "ets", "gmbh": "gmbh", "ag": "ag", "oyj": "oyj", "nv": "nv", "bv": "bv",
    "ms": "ms",  # "M/s" prefix common in Indian names
    # anyascii transliterations of Indic legal forms
    "praivet": "pvt", "privet": "pvt", "pra": "pvt", "li": "ltd", "limitid": "ltd",
    "elelpi": "llp", "kampani": "co", "kampni": "co", "kmpni": "co", "inkarporeted": "inc",
}
# Generic words that carry little identity (dropped from core name, kept in name_n).
FILLER = {
    "the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "formerly", "dba", "aka",
    "india", "france", "usa", "group", "groupe", "holdings", "holding", "center", "centre",
    "services", "service", "partners", "associates", "enterprises", "enterprise",
    "international", "global", "industries", "solutions", "ventures", "company",
}
# Address abbreviations -> canonical token.
ADDR = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave",
    "av": "ave", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "cr": "cir",
    "boulevard": "blvd", "bd": "blvd", "bvd": "blvd", "place": "pl", "highway": "hwy",
    "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq", "sqr": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt", "suite": "ste",
    "number": "no", "nr": "near", "opp": "opposite", "rue": "rue", "r": "rue",
    "impasse": "imp", "allee": "all", "chemin": "ch", "route": "rte", "quai": "qu",
    "floor": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "sector": "sec",
    "nagar": "ngr", "ngr": "ngr", "marg": "mg", "mg": "mg", "colony": "col",
    "mount": "mt", "fort": "ft", "township": "twp", "village": "vil", "vill": "vil",
}
# Address tokens that are pure noise.
ADDR_DROP = {"null", "none", "nan", "na", "unit", "no", "hno", "pmb", "po", "box", "house",
             "plot", "flat", "door", "shop", "office", "near", "opposite", "at", "and", "the"}
# US state names -> USPS codes (so "Indiana" == "IN").
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
US_STATES_2W = {  # two-word states, replaced on the joined string
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "district of columbia": "dc",
}


def _translit(col: str) -> pl.Expr:
    """Transliterate non-ASCII rows with anyascii, leave ASCII rows untouched."""
    c = pl.col(col).fill_null("")
    return (pl.when(c.str.contains(r"[^\x00-\x7F]"))
            .then(c.map_elements(anyascii, return_dtype=pl.String))
            .otherwise(c))


def _basic(e: pl.Expr) -> pl.Expr:
    """Lowercase, fix dotted acronyms (l.l.c. -> llc), strip punctuation to spaces."""
    e = e.str.to_lowercase()
    e = e.str.replace_all(r"<\s*null\s*>|\bnull\b|\bn/a\b", " ")
    e = e.str.replace_all("&", " and ").str.replace_all(r"\bm/s\b", " ms ")
    e = e.str.replace_all(r"www\.|\.(com|net|org|in|co\.in|fr|biz|us|info)\b", " ")
    # drop dots (s.a.s.u. -> sasu, l.l.c. -> llc, h.no -> hno), then re-split "no22" -> "no 22"
    e = e.str.replace_all(".", "", literal=True)
    e = e.str.replace_all(r"\b(no|hno|plot|flat|door|house|unit|apt|sec|ph|gat)(\d)", "$1 $2")
    e = e.str.replace_all(r"[^a-z0-9]+", " ")
    return e.str.strip_chars()


def _map_tokens(e: pl.Expr, mapping: dict) -> pl.Expr:
    """Replace whole tokens using a dict (vectorised via list.eval + replace)."""
    return (e.str.split(" ")
            .list.eval(pl.element().replace(mapping))
            .list.join(" "))


def _drop_tokens(e: pl.Expr, drop: set) -> pl.Expr:
    return (e.str.split(" ")
            .list.eval(pl.element().filter(~pl.element().is_in(list(drop)) & (pl.element() != "")))
            .list.join(" "))


def _deleet(e: pl.Expr) -> pl.Expr:
    """Undo leetspeak inside alphabetic words (Y0der -> yoder, 5mith -> smith)."""
    leet = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}

    def fix(tok: pl.Expr) -> pl.Expr:
        out = tok
        for d, ch in leet.items():
            out = out.str.replace_all(d, ch, literal=True)
        # only fix tokens that are mostly letters (keep '12th', '2d', '10c' etc.)
        n_alpha = tok.str.count_matches(r"[a-z]")
        return pl.when((n_alpha >= 2) & (n_alpha > tok.str.len_chars() / 2)).then(out).otherwise(tok)

    return e.str.split(" ").list.eval(fix(pl.element())).list.join(" ")


def _skeleton(e: pl.Expr) -> pl.Expr:
    """Consonant skeleton: collapse phonetic variants and drop vowels, so that
    'tek krietiv imdstrij' ~ 'tech creative industries'."""
    e = e.str.replace_all("ph", "f", literal=True)
    for a, b in [("ch", "k"), ("sh", "s"), ("th", "t"), ("kh", "k"), ("gh", "g"),
                 ("bh", "b"), ("dh", "d"), ("jh", "j"), ("q", "k"), ("c", "k"),
                 ("z", "j"), ("w", "v"), ("x", "ks"), ("h", ""), ("y", "")]:
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"m([bcdfgjklnpqrstvz])", "n$1")   # anusvara m -> n before consonant
    e = e.str.replace_all(r"[aeiou]", "")
    e = e.str.replace_all(r"j\b", "s")                         # Hindi plural: imdstrij ~ industries
    for ch in "bdfgjklmnprstv":                                # collapse doubles
        e = e.str.replace_all(ch + "{2,}", ch)
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _skel_py(t: str) -> str:
    return pl.select(_skeleton(pl.lit(t))).item()


DROP_SKEL = None
DROP_WORDS = set(LEGAL) | set(LEGAL.values()) | FILLER


def normalize(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add normalised columns to a frame with entity_id, business_name, business_address, country."""
    global DROP_SKEL
    if DROP_SKEL is None:  # skeletons of legal/filler words, so transliterated forms match too
        DROP_SKEL = {k for k in (_skel_py(w) for w in DROP_WORDS) if len(k) >= 4}
    df = df.with_columns(_translit("business_name").alias("_n"), _translit("business_address").alias("_a"))
    # accents left after anyascii on mixed rows are already gone; NFKD guard for safety
    df = df.with_columns(
        _basic(pl.col("_n")).alias("_n"),
        _basic(pl.col("_a")).alias("_a"),
    )
    name = _map_tokens(_deleet(pl.col("_n")), LEGAL)
    addr = pl.col("_a").str.replace_all(r"\b0+(\d)", "$1")  # 00209 -> 209
    for k, v in US_STATES_2W.items():
        addr = addr.str.replace_all(r"\b" + k + r"\b", v)
    addr = _map_tokens(_map_tokens(addr, ADDR), US_STATES)
    df = df.with_columns(name.alias("name_n"), addr.alias("_a"))
    df = df.with_columns(
        pl.col("name_n").str.split(" ").list.eval(
            pl.element().filter(
                (pl.element() != "")
                & ~pl.element().is_in(list(DROP_WORDS))
                # skeleton match only for long skeletons: short ones collide ('hail' -> 'l')
                & ~_skeleton(pl.element()).is_in(list(DROP_SKEL)))
        ).list.join(" ").alias("core_n"),
        pl.col("_a").str.extract_all(r"\b\d+[a-z]?\b")
          .list.eval(pl.element().str.replace(r"^0+(\d)", "$1")).alias("nums"),
        _drop_tokens(pl.col("_a"), ADDR_DROP).alias("addr_n"),
        pl.col("name_n").str.split(" ")
          .list.eval(pl.element().filter(pl.element().is_in(list(set(LEGAL.values()))))).alias("legal"),
    )
    df = df.with_columns(
        pl.when(pl.col("core_n") == "").then(pl.col("name_n")).otherwise(pl.col("core_n")).alias("core_n"))
    df = df.with_columns(_skeleton(pl.col("core_n")).alias("skel"))
    return df.drop("_n", "_a")
