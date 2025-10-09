#!/usr/bin/env python3
"""
Wrap figure generation modules in proper functions.

This script takes modules that execute code at module level and wraps
them in a generate() function with proper indentation.
"""

import re
from pathlib import Path

def wrap_module(filepath):
    """Wrap module code in a generate() function."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find where to start wrapping (after imports)
    lines = content.split('\n')

    # Find the last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            last_import_idx = i

    # Split into header (docstring + imports) and body (code to wrap)
    header_lines = lines[:last_import_idx + 1]
    body_lines = lines[last_import_idx + 1:]

    # Skip empty lines at the start of body
    while body_lines and not body_lines[0].strip():
        header_lines.append(body_lines.pop(0))

    # Check if already wrapped
    if any('def generate()' in line for line in body_lines[:10]):
        print(f"  ⏭️  Already wrapped: {filepath.name}")
        return False

    # Add function definition
    function_lines = [
        '',
        '',
        'def generate():',
        '    """Generate all figures for this module."""'
    ]

    # Indent all body lines
    indented_body = []
    for line in body_lines:
        if line.strip():  # Non-empty line
            indented_body.append('    ' + line)
        else:  # Empty line
            indented_body.append('')

    # Combine everything
    new_content = '\n'.join(header_lines + function_lines + indented_body)

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Wrapped: {filepath.name}")
    return True


def main():
    """Wrap all figure generation modules."""
    script_dir = Path(__file__).parent / 'slides' / 'figure_scripts'

    modules = [
        'bias_variance.py',
        'consistency.py',
        'efficiency.py',
        'confidence_intervals.py',
        'bootstrap.py',
        'delta_method.py'
    ]

    print("=" * 70)
    print("WRAPPING MODULES IN FUNCTIONS")
    print("=" * 70)

    wrapped_count = 0
    for module in modules:
        filepath = script_dir / module
        if wrap_module(filepath):
            wrapped_count += 1

    print("\n" + "=" * 70)
    print(f"✓ COMPLETED: Wrapped {wrapped_count}/{len(modules)} modules")
    print("=" * 70)


if __name__ == '__main__':
    main()
