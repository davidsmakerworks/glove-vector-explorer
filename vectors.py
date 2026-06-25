"""Pure functions for loading GloVe vectors and computing cosine similarity."""

import numpy as np


def load_vectors(
    path: str,
    cache_path: str | None = None,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Load GloVe vectors from a text file or a previously saved .npz cache.

    Args:
        path:       Path to a GloVe text file (.txt) or an .npz cache file.
        cache_path: If given and *path* is a text file, save a cache here so
                    subsequent runs can pass this path as *path* directly.

    Returns:
        (words, mat, index)  —  words list, float32 matrix, word→row dict.
    """
    if path.endswith(".npz"):
        return _load_npz(path)

    words, mat = _parse_text(path)
    index = {w: i for i, w in enumerate(words)}

    if cache_path:
        np.savez_compressed(
            cache_path,
            words=np.array(words, dtype=object),
            mat=mat,
        )

    return words, mat, index


def _load_npz(path: str) -> tuple[list[str], np.ndarray, dict[str, int]]:
    data = np.load(path, allow_pickle=True)
    words = list(data["words"])
    mat = data["mat"].astype(np.float32)
    index = {w: i for i, w in enumerate(words)}
    return words, mat, index


def _parse_text(path: str) -> tuple[list[str], np.ndarray]:
    words: list[str] = []
    vecs: list[np.ndarray] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            # Skip word2vec-style header (two integers)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if vec.size == 0:
                continue
            words.append(word)
            vecs.append(vec)

    mat = np.vstack(vecs).astype(np.float32)
    return words, mat


def unit_matrix(mat: np.ndarray) -> np.ndarray:
    """Return row-normalized copy of mat (unit vectors for cosine similarity)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


def nearest(
    vec: np.ndarray,
    unit_mat: np.ndarray,
    words: list[str],
    exclude: set[int],
    top_k: int = 1,
) -> list[tuple[str, float]]:
    """Return top_k (word, similarity) pairs closest to vec, excluding given indices."""
    norm = np.linalg.norm(vec)
    unit_vec = (vec / norm).astype(np.float32) if norm != 0 else vec.astype(np.float32)

    sims = unit_mat @ unit_vec

    if exclude:
        excl = np.array(list(exclude), dtype=np.intp)
        sims[excl] = -2.0  # below any valid cosine value

    if top_k == 1:
        idx = int(np.argmax(sims))
        return [(words[idx], float(sims[idx]))]

    k = min(top_k, len(words) - len(exclude))
    part = np.argpartition(sims, -k)[-k:]
    part = part[np.argsort(sims[part])[::-1]]
    return [(words[i], float(sims[i])) for i in part]


def parse_line(line: str) -> tuple[str, str] | None:
    """Parse an operator expression or a bare word.

    Returns:
        ('+', word)  — add word's vector to current
        ('-', word)  — subtract word's vector from current
        ('=', word)  — bare word: jump to this word as new starting point
        None         — unrecognized input
    """
    line = line.strip()
    if not line:
        return None
    if line[0] in ("+", "-"):
        op = line[0]
        word = line[1:].strip().lower()
        if word and " " not in word:
            return op, word
        return None
    # Bare word with no spaces → jump
    if " " not in line:
        return "=", line.lower()
    return None
