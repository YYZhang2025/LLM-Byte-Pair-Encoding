import unicodedata


def get_stats(ids, counts=None):
    """
    Count the number of pair occurrences in a list of ids.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_idx):
    """Merge a pair of ids into a new id."""
    new_ids = []

    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_idx)
            i += 2  # skip the next id since it's part of the pair
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


# Helper functions
def replace_control_characters(s: str) -> str:
    """
    Replace control characters, such as \n, \t, etc., with their Unicode names.
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch).startswith("C"):
            chars.append(f"\\u{ord(ch):04x}")
        else:
            chars.append(ch)

    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    Render a token as a string, replacing control characters with their Unicode names.
    """
    s = t.decode("utf-8", errors="replace")
    return replace_control_characters(s)
