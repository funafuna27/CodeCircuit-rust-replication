"""A second cleaning pass for sampled Rust, candidate for Phase 3 (not used by Phase 2).

At temperature 0.8 the model often wraps or repeats its answer: prose before the
function, a second complete copy of it (rustc E0428), stray tokens between
items. The Phase 2 cleaner -- fences as separators, comment and blank lines
dropped, a stray leading `;` dropped, a cut-off restart trimmed -- was built on
greedy output and lets these through, so the sample is recorded as a compile
failure for something that is not the code.

tidy() keeps only well-formed top-level items, in order:
  - a top-level line that does not start an item (prose, a stray `{`, a
    backtick) is dropped -- unless it continues an item header that has not yet
    opened its body or ended with `;` (a signature split over lines);
  - a second top-level definition of the target function is dropped whole;
  - everything else -- helpers before or after, `use`, structs, `fn main` -- is
    kept, so it never removes code a solution needs.

Brace depth ignores braces inside strings, char literals and // comments, and
tells a lifetime ('a) from a char literal ('{').
"""
import re

ITEM = re.compile(r'^\s*(#!?\[|(pub(\([^)]*\))?\s+)?(unsafe\s+)?'
                  r'(fn|use|struct|enum|impl|const|static|trait|mod|type|extern)\b|macro_rules!)')


def _scan(line, st):
    """Advance st = [depth, in_str] over one line; return True if a brace
    outside strings, chars and comments was seen."""
    depth, in_str = st
    saw = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if in_str:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "'":
            if i + 2 < n and line[i + 1] != '\\' and line[i + 2] == "'":
                i += 3                              # 'x'
                continue
            if i + 1 < n and line[i + 1] == '\\':
                j = line.find("'", i + 2)           # '\n', '\u{7f}'
                if j != -1:
                    i = j + 1
                    continue
            # otherwise a lifetime: skip just the quote
        elif ch == '/' and i + 1 < n and line[i + 1] == '/':
            break
        elif ch == '{':
            depth += 1
            saw = True
        elif ch == '}':
            depth -= 1
            saw = True
        i += 1
    st[0], st[1] = max(depth, 0), in_str
    return saw


def tidy(code: str, name: str) -> str:
    target = re.compile(rf'^\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?fn\s+{re.escape(name)}\s*[<(]')
    out, st = [], [0, False]
    keep, in_header, seen_target = True, False, False
    for line in code.split('\n'):
        top = st[0] == 0 and not st[1] and not in_header
        is_attr = False
        if top:
            if not ITEM.match(line):
                continue
            is_target = bool(target.match(line))
            keep = not (is_target and seen_target)
            seen_target = seen_target or is_target
            is_attr = line.lstrip().startswith('#')
        if keep:
            out.append(line)
        saw = _scan(line, st)
        ends = saw or line.rstrip().endswith(';')
        if top and not is_attr:
            in_header = not ends
        elif in_header and ends:
            in_header = False
    return '\n'.join(out).strip()


def _selftest():
    name = 'f'
    cases = {
        'duplicate definition': ("fn f(a: i64) -> i64 {\n    a + 1\n}\nfn f(a: i64) -> i64 {\n    a + 2\n}",
                                 "fn f(a: i64) -> i64 {\n    a + 1\n}"),
        'prose preamble': ("Here is the code:\nfn f(a: i64) -> i64 {\n    a\n}",
                           "fn f(a: i64) -> i64 {\n    a\n}"),
        'helper after kept': ("fn f(a: i64) -> i64 {\n    g(a)\n}\nfn g(a: i64) -> i64 {\n    a\n}",
                              "fn f(a: i64) -> i64 {\n    g(a)\n}\nfn g(a: i64) -> i64 {\n    a\n}"),
        'multi-line signature': ("fn f(a: i64,\n     b: i64) -> i64 {\n    a + b\n}",
                                 "fn f(a: i64,\n     b: i64) -> i64 {\n    a + b\n}"),
        'use and attribute': ("use std::collections::HashMap;\n#[inline]\nfn f(a: i64) -> i64 {\n    a\n}\nfn f(a: i64) -> i64 {\n    0\n}",
                              "use std::collections::HashMap;\n#[inline]\nfn f(a: i64) -> i64 {\n    a\n}"),
        'stray brace and backtick': ("fn f(a: i64) -> i64 {\n    a\n}\n{\n```",
                                     "fn f(a: i64) -> i64 {\n    a\n}"),
        'braces in strings': ("fn f(a: i64) -> String {\n    format!(\"{}}}\", a)\n}\nfn f(a: i64) -> String {\n    String::new()\n}",
                              "fn f(a: i64) -> String {\n    format!(\"{}}}\", a)\n}"),
        'brace char literals': ("fn f(s: &str) -> bool {\n    let mut d = 0;\n    for c in s.chars() {\n        if c == '{' { d += 1; }\n        if c == '}' { d -= 1; }\n    }\n    d == 0\n}\nThat's it.",
                                "fn f(s: &str) -> bool {\n    let mut d = 0;\n    for c in s.chars() {\n        if c == '{' { d += 1; }\n        if c == '}' { d -= 1; }\n    }\n    d == 0\n}"),
        'lifetime': ("fn f<'a>(a: &'a str) -> &'a str {\n    a\n}\nok then",
                     "fn f<'a>(a: &'a str) -> &'a str {\n    a\n}"),
        'one-line item': ("fn g() -> i64 { 1 }\nfn f(a: i64) -> i64 {\n    a + g()\n}",
                          "fn g() -> i64 { 1 }\nfn f(a: i64) -> i64 {\n    a + g()\n}"),
        'main kept': ("fn f(a: i64) -> i64 {\n    a\n}\nfn main() {\n    println!(\"{}\", f(1));\n}",
                      "fn f(a: i64) -> i64 {\n    a\n}\nfn main() {\n    println!(\"{}\", f(1));\n}"),
        'unchanged when clean': ("fn f(a: Vec<i64>) -> i64 {\n    a.iter().sum()\n}",
                                 "fn f(a: Vec<i64>) -> i64 {\n    a.iter().sum()\n}"),
    }
    bad = 0
    for k, (src, want) in cases.items():
        got = tidy(src, name)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {k}" + ('' if ok else f"\n---got---\n{got}\n---want---\n{want}"))
    return bad


if __name__ == '__main__':
    raise SystemExit(_selftest())
