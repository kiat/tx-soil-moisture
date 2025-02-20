#!/usr/bin/env python3
# setup/tree/tree_cmd.py
import os
import sys
from pathlib import Path

def main():
    # Adjust as needed if tree_ignore.txt is in a different folder
    ignore_file = Path(__file__).resolve().parent.joinpath("tree_ignore.txt")

    if not ignore_file.exists():
        print("Error: 'tree_ignore.txt' does not exist at:", ignore_file, file=sys.stderr)
        sys.exit(1)

    # Read patterns: one per line or space separated
    patterns = ignore_file.read_text().strip().split()
    # Join them with a pipe for the tree -I option
    ignore_pattern = "|".join(patterns)

    # Use double quotes around the pattern to avoid shell issues
    cmd = f'tree -I "{ignore_pattern}"'
    exit_code = os.system(cmd)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
