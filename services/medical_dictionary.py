import functools
import os
import pickle
import time
import xml.etree.ElementTree as ET


# ============================================================
# MeSH Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data"
)


# ============================================================
# Find MeSH XML File``
# ============================================================

def find_mesh_file():
    """
    Find the MeSH XML file inside the data directory.

    The function does not hardcode a particular production year.
    This allows the MeSH file to be updated without changing
    the Python code.
    """

    if not os.path.exists(DATA_DIRECTORY):
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIRECTORY}"
        )

    xml_files = [
        filename
        for filename in os.listdir(DATA_DIRECTORY)
        if filename.lower().endswith(".xml")
    ]

    if not xml_files:
        raise FileNotFoundError(
            "No MeSH XML file was found in the data directory."
        )

    # If multiple XML files exist, use the first one.
    # We can refine this later if necessary.
    xml_files.sort()

    return os.path.join(
        DATA_DIRECTORY,
        xml_files[0]
    )


# ============================================================
# Load MeSH Vocabulary
# ============================================================

CACHE_FILE = os.path.join(
    DATA_DIRECTORY,
    "mesh_vocabulary.pkl",
)

CACHE_VERSION = 3


def _source_fingerprint(mesh_file):
    """Identify the source file so a changed MeSH release is noticed."""

    stat = os.stat(mesh_file)

    return {
        "version": CACHE_VERSION,
        "name": os.path.basename(mesh_file),
        "size": stat.st_size,
        "modified": int(stat.st_mtime),
    }


def _read_cache(fingerprint):
    """Return the cached vocabulary, or None if it cannot be used."""

    if not os.path.exists(CACHE_FILE):
        return None

    try:
        with open(CACHE_FILE, "rb") as handle:
            payload = pickle.load(handle)

    except Exception as error:  # noqa: BLE001
        print("[MESH] Ignoring unreadable cache:", error)
        return None

    if payload.get("fingerprint") != fingerprint:
        print("[MESH] Source file changed - rebuilding the cache.")
        return None

    vocabulary = payload.get("vocabulary")
    terms = payload.get("terms")

    if not isinstance(vocabulary, set) or not vocabulary:
        return None

    if not isinstance(terms, set) or not terms:
        return None

    return vocabulary, terms


def _write_cache(fingerprint, vocabulary, terms):
    try:
        with open(CACHE_FILE, "wb") as handle:
            pickle.dump(
                {
                    "fingerprint": fingerprint,
                    "vocabulary": vocabulary,
                    "terms": terms,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    except OSError as error:
        print("[MESH] Could not write the cache:", error)


MIN_INDEXED_WORD_LENGTH = 4


def _index_words_of(term):
    """Yield the individual words of a multi-word MeSH term.

    MeSH stores descriptors in full, so "Lupus Erythematosus,
    Systemic" leaves the everyday word "lupus" absent from the
    vocabulary. The same is true of "diabetes" and "autoimmune".
    Indexing the parts makes those recognisable on their own.
    """

    if " " not in term:
        return

    for word in term.replace(",", " ").split():

        word = word.strip("()[].;:'\"")

        if len(word) >= MIN_INDEXED_WORD_LENGTH and word.isalpha():
            yield word


def _parse_mesh_file(mesh_file):
    """Stream the MeSH XML and collect descriptor and entry terms.

    iterparse is used rather than ET.parse because the descriptor
    file is several hundred megabytes; building the whole tree in
    memory costs far more than the terms themselves.
    """

    vocabulary = set()

    try:
        context = ET.iterparse(
            mesh_file,
            events=("start", "end"),
        )

        _, root = next(context)

        for event, element in context:

            if (
                event != "end"
                or element.tag != "DescriptorRecord"
            ):
                continue

            descriptor_name = element.find(
                "./DescriptorName/String"
            )

            if (
                descriptor_name is not None
                and descriptor_name.text
            ):
                term = descriptor_name.text.strip().lower()

                if term:
                    vocabulary.add(term)

            # Entry terms / synonyms
            for term_element in element.findall(
                ".//TermList/Term/String"
            ):

                if term_element.text:

                    term = term_element.text.strip().lower()

                    if term:
                        vocabulary.add(term)

            # Release this record and the siblings the parser has
            # accumulated, so memory stays flat across the file.
            element.clear()
            root.clear()

    except ET.ParseError as error:

        raise ValueError(
            f"Unable to parse MeSH XML file: {error}"
        )

    return vocabulary


# query_utils asks for both sets, which used to unpickle the cache
# twice and keep two copies of roughly 550,000 strings alive - about
# 39 MB of duplicate vocabulary for no benefit.
@functools.lru_cache(maxsize=1)
def _load_both():
    """Return (vocabulary, terms) from the MeSH file.

    ``terms`` are the descriptors and entry terms exactly as NLM
    publishes them. ``vocabulary`` additionally contains the
    individual words of multi-word terms, so "lupus" and "diabetes"
    are recognised on their own.

    The two are cached together, so only the first run after a new
    MeSH release pays the parsing cost.
    """

    mesh_file = find_mesh_file()

    fingerprint = _source_fingerprint(mesh_file)

    started = time.perf_counter()

    cached = _read_cache(fingerprint)

    if cached is not None:

        print(
            f"Loaded {len(cached[0])} MeSH terms from cache "
            f"in {time.perf_counter() - started:.2f}s."
        )

        return cached

    print(
        f"Parsing MeSH vocabulary from: {mesh_file}"
    )

    terms = _parse_mesh_file(mesh_file)

    vocabulary = set(terms)

    vocabulary.update(
        word
        for term in terms
        for word in _index_words_of(term)
    )

    _write_cache(fingerprint, vocabulary, terms)

    print(
        f"Loaded {len(vocabulary)} MeSH terms in "
        f"{time.perf_counter() - started:.2f}s "
        f"(cached to {os.path.basename(CACHE_FILE)})."
    )

    return vocabulary, terms


def load_mesh_vocabulary():
    """Terms plus the individual words of multi-word terms.

    Used for spelling correction, where a broad set of recognisable
    words means fewer false corrections and more valid targets.
    """

    return _load_both()[0]


def load_mesh_terms():
    """Only terms MeSH publishes in their own right.

    Used where a word must genuinely name a clinical concept, such
    as picking search keywords. The broader vocabulary is wrong
    there: "normal" and "count" appear inside MeSH descriptors but
    are filler in a question.
    """

    return _load_both()[1]
