"""Central runtime configuration loaded from the project ``.env`` file.

Environment variables supplied by the operating system take precedence, so the
same project can use local development settings without changing deployment
configuration.  Import this module before reading any DDI_* or Gemini setting.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

# ``override=False`` means a CI secret or a variable set by the host wins over
# a local file.  The .env file is deliberately ignored by Git.
load_dotenv(ENV_FILE, override=False)
