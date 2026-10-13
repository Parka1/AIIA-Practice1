"""Azure credentials for the local serverless scripts.

Each value is read from its environment variable or, if it is not set, from
credentials.env at the repository root (git-ignored), which has this layout:

    # Storage account
    Key=...
    ConnectionString=...

    # Computer Vision resource
    Key 1=...
    Endpoint=...

The Azure Function does not use this module: there the values are app settings,
which reach the function as environment variables.
"""

import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "credentials.env"

# name -> (environment variable, (section, key) in credentials.env)
SOURCES = {
    "vision_endpoint": ("AZURE_VISION_ENDPOINT", ("computer vision resource", "endpoint")),
    "vision_key": ("AZURE_VISION_KEY", ("computer vision resource", "key 1")),
    "storage_connection_string": ("AZURE_STORAGE_CONNECTION_STRING", ("storage account", "connectionstring")),
}


def load_credentials(*names: str, path: Path = DEFAULT_PATH) -> tuple[str, ...]:
    """Return the requested values, e.g. load_credentials("vision_endpoint", "vision_key")."""
    from_file = parse_credentials_file(path.read_text(encoding="utf-8")) if path.is_file() else {}
    values = []
    for name in names:
        env_var, file_key = SOURCES[name]
        value = os.getenv(env_var) or from_file.get(file_key)
        if not value:
            raise ValueError(f"{name} not found: set {env_var} or add it to {path}")
        values.append(value)
    return tuple(values)


def parse_credentials_file(text: str) -> dict[tuple[str, str], str]:
    """Parse '# Section' headers and 'Key=value' lines; names are lower-cased."""
    values = {}
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            section = line.lstrip("#").strip().lower()
        elif "=" in line:
            key, value = line.split("=", 1)
            values[(section, key.strip().lower())] = value.strip()
    return values
