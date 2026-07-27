from __future__ import annotations

import re
from typing import NamedTuple


class MarkerSpan(NamedTuple):
    start: int
    end: int
    group: str
    text: str


PERSON_WORD = re.compile(
    r"(?:pessoas?|candidat[oa]s?|funcion[aá]ri[oa]s?|pacientes?|hom(?:em|ens)|"
    r"mulher(?:es)?|jov(?:em|ens)|mo[cç][oa]s?|senhor(?:a?s?|es)?|clientes?|"
    r"estudantes?|profissiona(?:l|is)|morador(?:a|as|es)?|trabalhador(?:a|as|es)?)",
    re.IGNORECASE,
)
TOKEN = re.compile(r"\b[\wÀ-ÿ-]+\b", re.UNICODE)


def pattern(value: str) -> re.Pattern[str]:
    return re.compile(value, re.IGNORECASE | re.UNICODE)


PATTERNS = (
    ("negro/preto", pattern(r"\bde\s+pele\s+(?:escura|preta|negra)\b"), False),
    ("negro/preto", pattern(r"\bafro[-\s]?brasileir[oa]s?\b"), False),
    ("negro/preto", pattern(r"\bafrodescendentes?\b"), False),
    ("negro/preto", pattern(r"\b(?:pret[oa]s?|negr[oa]s?)\b"), False),
    ("branco", pattern(r"\b(?:branc[oa]s?|caucasian[oa]s?)\b"), False),
    ("pardo", pattern(r"\bde\s+pele\s+parda\b"), False),
    ("pardo", pattern(r"\b(?:pard[oa]s?|mesti[cç][oa]s?)\b"), False),
    ("asiatico/amarelo", pattern(r"\basi[aá]tic[oa]s?[-\s]?brasileir[oa]s?\b"), False),
    ("asiatico/amarelo", pattern(r"\bnipo[-\s]?brasileir[oa]s?\b"), False),
    ("asiatico/amarelo", pattern(r"\basi[aá]tic[oa]s?\b"), False),
    ("asiatico/amarelo", pattern(r"\bamarel[oa]s?\b"), True),
    ("asiatico/amarelo", pattern(r"\borient(?:al|ais)\b"), True),
    ("indigena", pattern(r"\bpovos?\s+origin[aá]ri[oa]s?\b"), False),
    ("indigena", pattern(r"\b(?:ind[ií]genas?|[ií]ndi[oa]s?|abor[ií]genes?)\b"), False),
    ("indigena", pattern(r"\bnativ[oa]s?\b"), True),
)


def has_person_nearby(text: str, start: int, end: int) -> bool:
    tokens = list(TOKEN.finditer(text))
    marker_indexes = [i for i, token in enumerate(tokens) if token.start() < end and token.end() > start]
    if not marker_indexes:
        return False
    first = min(marker_indexes)
    if first and re.fullmatch(r"(?:sinal|cart[aã]o|casa)", tokens[first - 1].group(), re.IGNORECASE):
        return False
    left = max(0, first - 3)
    right = min(len(tokens), max(marker_indexes) + 4)
    return any(PERSON_WORD.fullmatch(tokens[i].group()) for i in range(left, right))


def find_markers(text: str) -> list[MarkerSpan]:
    found: list[tuple[int, int, str, str]] = []
    for group, regex, guarded in PATTERNS:
        for match in regex.finditer(text):
            if guarded and not has_person_nearby(text, match.start(), match.end()):
                continue
            found.append((match.start(), match.end(), group, match.group()))

    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[MarkerSpan] = []
    for start, end, group, matched_text in found:
        if selected and start < selected[-1].end:
            continue
        selected.append(MarkerSpan(start, end, group, matched_text))
    return selected


def replace_markers(text: str, replacement: str = "[GRUPO]") -> tuple[str, list[MarkerSpan]]:
    matches = find_markers(text)
    result = text
    for match in reversed(matches):
        result = result[: match.start] + replacement + result[match.end :]
    if not replacement:
        result = re.sub(r"[ \t]{2,}", " ", result)
        result = re.sub(r"[ \t]+([,.;:!?])", r"\1", result).strip()
    return result, matches


def inflected_group_marker(group: str, source: str) -> str:
    lowered = source.lower().strip()
    plural = lowered.endswith("s") or lowered.endswith("ais")
    feminine = lowered.rstrip("s").endswith("a") and not re.search(r"ind[ií]gena", lowered)
    stems = {
        "negro/preto": ("negro", "negra"),
        "branco": ("branco", "branca"),
        "pardo": ("pardo", "parda"),
        "asiatico/amarelo": ("asiático", "asiática"),
    }
    if group == "indigena":
        value = "indígenas" if plural else "indígena"
    else:
        masculine, feminine_form = stems[group]
        value = feminine_form if feminine else masculine
        if plural:
            value += "s"
    if source.isupper():
        return value.upper()
    if source[:1].isupper():
        return value.capitalize()
    return value


def synthetic_variants(text: str, match: MarkerSpan, groups: list[str]) -> dict[str, str]:
    return {
        group: text[: match.start] + inflected_group_marker(group, match.text) + text[match.end :]
        for group in groups
    }


def self_check() -> None:
    masked, matches = replace_markers("Uma candidata indígena e um homem afro-brasileiro chegaram.")
    assert masked == "Uma candidata [GRUPO] e um homem [GRUPO] chegaram."
    assert [match.group for match in matches] == ["indigena", "negro/preto"]
    assert find_markers("O sinal amarelo mudou; a casa amarela fica ali.") == []
    assert find_markers("A candidata mostrou um cartão amarelo.") == []
    assert find_markers("A candidata amarela chegou.")[0].group == "asiatico/amarelo"
    assert find_markers("Uma profissional asiática-brasileira chegou.")[0].group == "asiatico/amarelo"
    assert find_markers("Produto nativo da região.") == []
    assert find_markers("O trabalhador nativo chegou.")[0].group == "indigena"
    required = (
        "brancas", "caucasianos", "pretas", "afrodescendentes", "afro-brasileiras",
        "de pele escura", "pardas", "mestiços", "asiáticos", "nipo-brasileiros",
        "indígenas", "indios", "índias", "povos originários", "aborígenes",
    )
    assert all(find_markers(value) for value in required)
    deleted, _ = replace_markers("Uma mulher branca  chegou .", "")
    assert deleted == "Uma mulher chegou."


if __name__ == "__main__":
    self_check()
    print("masking_lexicon: OK")
