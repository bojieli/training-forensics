"""Token categories for fine-tune token records (multi-label; decoded token text)."""
from __future__ import annotations

import re
import unicodedata

SCRIPTS = ["LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "HEBREW", "GREEK",
           "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "KANNADA", "MALAYALAM", "GURMUKHI",
           "THAI", "LAO", "KHMER", "MYANMAR", "GEORGIAN", "ARMENIAN", "ETHIOPIC", "SINHALA", "TIBETAN"]
ALIAS = {"HANGUL": "HANGUL", "CJK": "CJK", "ETHIOPIC": "ETHIOPIC"}

CHAT = re.compile(r"im_start|im_end|start_header|end_header|eot_id|end_of_turn|start_of_turn|<\|user\|>|<\|assistant\|>|"
                  r"<\|system\|>|<\|end\|>|\[INST\]|\[/INST\]|<\|endofprompt|<\|turn|<\|role|<\|message|<\|channel|"
                  r"<\|im_sep|<\|observation|<\|eom_id|bos|eos|endoftext|<s>|</s>|<\|begin_of_text|<\|end_of_text|"
                  r"<\|beginoftext|<\|endoftext|<\|return|<\|start|<\|constrain|<\|end_of_sentence|<\|Assistant|<\|User", re.I)
TOOL = re.compile(r"tool|function|python_tag|<\|call|api_|plugin|code_interpreter|arguments|\[CALL|\[TOOL", re.I)
REASON = re.compile(r"think|reason|reflect|<\|analysis|<\|final|scratch|solution|<\|begin_of_thought|_of_thought", re.I)
MM = re.compile(r"image|img|vision|video|audio|vid_|<\|box|<\|quad|object_ref|<\|ref|patch|frame|speech|pixel|<\|vq|"
                r"<\|AUDIO|<\|IMAGE|<\|VIDEO|<\|mm|<\|media|<\|point|boi|eoi|<\|placeholder", re.I)
RESERVED = re.compile(r"reserved|unused|<\|extra_|\[control_|<unused|placeholder|<\|dummy|<\|pad|\[PAD|<pad|<\|fim|"
                      r"<\|repo_name|<\|file_sep|<SPECIAL_|<filler|_SPECIAL|<\|special", re.I)
FIM = re.compile(r"fim|repo_name|file_sep", re.I)
CODE_KW = {"def", "return", "import", "class", "function", "const", "var", "let", "if", "else", "elif", "for", "while",
           "include", "public", "private", "static", "void", "int", "self", "this", "null", "None", "True", "False",
           "struct", "fn", "impl", "package", "namespace", "async", "await", "lambda", "printf", "println", "std",
           "nullptr", "typedef", "extends", "try", "catch", "except", "finally", "throw", "raise", "yield", "unsigned"}
CODE_PUNCT = re.compile(r"[{}\[\];]|==|!=|->|=>|::|&&|\|\||\+\+|--|<=|>=|\(\)|#include|//|/\*|\*/|\+=|-=")
MATH = re.compile(r"[∑∫√≤≥±×÷∞∂∇∈∉⊂⊆∪∩≈≠πθαβγδλμσφωΩΣΔ]|\\(frac|sqrt|sum|int|alpha|beta|cdot|times|left|right|mathbb|mathrm|"
                  r"begin|end|leq|geq|infty|pi|theta|lambda|partial|text|boxed|dfrac)")


def script_of(text: str):
    counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            n = unicodedata.name(ch)
        except ValueError:
            continue
        first = n.split()[0]
        s = first if first in SCRIPTS else ("CJK" if "CJK" in n or "IDEOGRAPH" in n else "OTHER")
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def is_emoji(ch):
    o = ord(ch)
    return (0x1F300 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or (0x1F000 <= o <= 0x1F2FF) or o == 0xFE0F


def categories(text: str, is_added: bool, is_special: bool):
    """Return a set of category labels."""
    c = set()
    if is_added or is_special:
        c.add("added/special")
        if CHAT.search(text):
            c.add("chat-template")
        if TOOL.search(text):
            c.add("tool-call")
        if REASON.search(text):
            c.add("reasoning-tag")
        if MM.search(text):
            c.add("multimodal")
        if RESERVED.search(text) and not FIM.search(text):
            c.add("reserved")
        if FIM.search(text):
            c.add("code-FIM")
        if len(c) == 1:
            c.add("other-special")
        return c
    if "�" in text:
        c.add("byte-fragment")
        return c
    s = script_of(text)
    if s:
        c.add("script:" + s)
    t = text.strip()
    if t == "" and len(text) >= 1:
        c.add("whitespace" if len(text) < 2 else "code:indent")
    if t.isdigit():
        c.add("digits")
    if MATH.search(text):
        c.add("math")
    if any(is_emoji(ch) for ch in text):
        c.add("emoji")
    if t in CODE_KW or CODE_PUNCT.search(text):
        c.add("code")
    if not c and t and all(not ch.isalnum() for ch in t):
        c.add("punct/symbol")
    if not c:
        c.add("other")
    return c
