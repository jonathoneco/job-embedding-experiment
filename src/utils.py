"""Shared utility functions for JSON I/O."""

import json


def load_json(path):
    """Load JSON data from file."""
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    """Save data as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
