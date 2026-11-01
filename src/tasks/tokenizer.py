"""Tokenizers with one interface: encode(text, kind) -> ids, decode(ids) -> str.

CharTokenizer  self-contained, fixed vocabulary (identical for every arm, no downloads):
               specials + printable ASCII + common multi-character SMILES tokens.
               SMILES segments are split with the usual SMILES regex (bracket atoms, Cl, Br, @@, %nn),
               everything else character by character (so numbers are digit-level).
               A regex piece that is not in the vocabulary falls back to its characters.
HFTokenizer    wraps a Hugging Face tokenizer for pretrained LMs.
"""
from __future__ import annotations

import re

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mol>", "</mol>", "<xyz>", "</xyz>"]
SMILES_MULTI = [
    "Cl", "Br", "@@",
    "[nH]", "[NH+]", "[NH2+]", "[NH3+]", "[N+]", "[N-]", "[n+]", "[n-]", "[O-]", "[O+]", "[OH+]",
    "[C@H]", "[C@@H]", "[C@]", "[C@@]", "[CH-]", "[CH2-]", "[C-]", "[c-]", "[S+]", "[S-]", "[s+]",
    "[Li+]", "[Li]", "[Na+]", "[K+]", "[Mg+2]", "[B-]", "[P-]", "[P+]", "[As-]", "[Si]", "[Se]", "[se]",
    "[2H]", "[H]",
]
_SMILES_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|@@|%\d{2}|.)", re.S)
_TEXT_RE = re.compile(r"(<mol>|</mol>|<xyz>|</xyz>|.)", re.S)


class CharTokenizer:
    name = "char"

    def __init__(self):
        itos = []
        for t in SPECIALS + [chr(i) for i in range(32, 127)] + SMILES_MULTI:
            if t not in itos:
                itos.append(t)
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (self.stoi[t] for t in SPECIALS[:4])
        self._special_ids = {self.pad_id, self.bos_id, self.eos_id}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, kind: str = "other") -> list:
        regex = _SMILES_RE if kind == "smiles" else _TEXT_RE
        ids = []
        for piece in regex.findall(text):
            if piece in self.stoi:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self.stoi.get(c, self.unk_id) for c in piece)
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i not in self._special_ids and 0 <= i < len(self.itos))


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.eos_id = self.tok.eos_token_id
        self.bos_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.eos_id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def encode(self, text: str, kind: str = "other") -> list:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)


def get_tokenizer(model: str):
    """model: tiny | small | hf:<name>"""
    return HFTokenizer(model[3:]) if model.startswith("hf:") else CharTokenizer()
