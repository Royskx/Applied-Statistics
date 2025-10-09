#!/usr/bin/env python3
"""Sanitize .tex files for common leading-backslash/tab insertion issues.

This script fixes these issues (only at start of line after indentation):
- leading tab characters replaced by two spaces
- lines that start with double backslash (e.g. "\\titlepage") -> single backslash ("\titlepage")
- lines where a common LaTeX command lost its leading backslash (e.g. "titlepage") -> restore leading backslash

Run without args to scan `lessons/` recursively. Pass file paths to process specific files.
"""
import argparse
import pathlib
import re
import sys

WHITELIST = {
    "titlepage",
    "tableofcontents",
    "maketitle",
    "title",
    "author",
    "date",
    "institute",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
    "frametitle",
    "frame",
    "centering",
    "vspace",
    "includegraphics",
    "caption",
}



def process_text(text: str) -> (str, bool):
    changed = False
    out_lines = []
    dbl_bs_re = re.compile(r'^(\s*)\\\\([A-Za-z@]+)')
    missing_bs_re = re.compile(r'^(\s*)(' + r'|'.join(re.escape(w) for w in WHITELIST) + r')\b')
    for line in text.splitlines(True):
        orig = line
        # Replace leading tabs with two spaces per tab
        # (only leading tabs to avoid changing content elsewhere)
        line = re.sub(r'^\t+', lambda m: '  ' * len(m.group(0)), line)
        # Convert leading double-backslash to single (only at line start after indentation)
        line = dbl_bs_re.sub(r"\1\\\2", line)
        # If a whitelisted command lost its backslash at line-start, restore it
        line = missing_bs_re.sub(r"\1\\\2", line)
        if line != orig:
            changed = True
        out_lines.append(line)
    return ''.join(out_lines), changed


def process_file(path: pathlib.Path) -> bool:
    try:
        text = path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"[skip] Cannot read {path}: {e}")
        return False
    new_text, changed = process_text(text)
    if changed:
        path.write_text(new_text, encoding='utf-8')
        print(f"fixed: {path}")
    return changed


def gather_paths(args):
    paths = []
    if args.paths:
        for p in args.paths:
            ppath = pathlib.Path(p)
            if ppath.is_file() and ppath.suffix in ('.tex',):
                paths.append(ppath)
            elif ppath.is_dir():
                paths.extend(ppath.rglob('*.tex'))
            else:
                # allow globs
                for m in pathlib.Path('.').glob(p):
                    if m.is_file() and m.suffix == '.tex':
                        paths.append(m)
    else:
        # default: scan lessons/ recursively
        paths.extend(pathlib.Path('lessons').rglob('*.tex'))
    # unique
    seen = set()
    uniq = []
    for p in paths:
        pp = p.resolve()
        if pp not in seen:
            seen.add(pp)
            uniq.append(pp)
    return uniq


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='Files or directories or globs to sanitize (default: lessons/**/*.tex)')
    args = parser.parse_args(argv)
    paths = gather_paths(args)
    if not paths:
        print('No .tex files found to process.')
        return 0
    any_changed = False
    for p in paths:
        changed = process_file(p)
        any_changed = any_changed or changed
    if any_changed:
        print('One or more files were modified. Please review and (re-)stage changes if using git.')
        return 1
    print('No changes necessary.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
