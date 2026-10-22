"""HF `tokenizers`' ids for prompts holding NUL and other control bytes.

usage: python nul_cases.py <tokenizer.json> <cases.txt>
Writes one case a line: the text's UTF-8 bytes in hex, a tab, the ids
(add_special_tokens=False, as the engine's tok_encode). tok_nul_check.c reads it."""
import sys
from tokenizers import Tokenizer

CASES = [
    "\x00", "\x00\x00\x00", "a\x00b", " \x00", "\x00 ", "\x00abc", "abc\x00", "\x00123", "1\x002",
    "\x00\n", "\n\x00", "\x00\r\n\x00", "\t\x00\t", "x \x00 y", "\x00'll", "'\x00s",
    "\x01\x02\x03\x04\x07\x08\x0b\x0c\x0e\x1f\x7f", "\x00￿￿\x80", "À肈\x00\x00\x04",
    "<|im_start|>tool\n\x00\x00|\x00<|im_end|>", "\x00<|im_end|>\x00", "Привет\x00мир", "日本\x00語",
    "%.6X\x00\x00.\x00\x00\x00\\\x00\x00\x00fCoder\x00\x00-flag 15",
]

tok = Tokenizer.from_file(sys.argv[1])
with open(sys.argv[2], "w", encoding="ascii", newline="\n") as out:
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        out.write(text.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)) + "\n")
print(len(CASES), "cases")
