#!/bin/bash
# Quiet compilation for development - suppresses warnings but shows important messages
# Compile with XeLaTeX (first run)
xelatex -interaction=nonstopmode -halt-on-error "$@" 2>&1 | grep -v "^warning:" | grep -v "you may need to load" | grep -v "Missing character:" | grep -v "could not represent character" | grep -E "(Writing|Running|Error|error|Fatal|fatal|Output written|Document Class|^note: Running|^note: Writing|^note: Skipped)" > /dev/null
# Second run for TOC/bookmarks
xelatex -interaction=nonstopmode -halt-on-error "$@" 2>&1 | grep -v "^warning:" | grep -v "you may need to load" | grep -v "Missing character:" | grep -v "could not represent character" | grep -E "(Writing|Running|Error|error|Fatal|fatal|Output written|Document Class|^note: Running|^note: Writing|^note: Skipped)"