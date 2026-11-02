import json
import re
from pathlib import Path


def account_id():
    value = json.loads(Path('submission_settings.json').read_text())['account_id']
    if value != 'ID未設定' and not re.fullmatch(r'[A-Za-z0-9_.-]{1,40}', value):
        raise ValueError('Use an anonymous account ID of 1-40 ASCII letters/digits/._-')
    return value
