import json
import re

def safe_json_parse(text, fallback={'action': 'fold'}):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON-like substring from the LLM output
        matches = re.findall(r"\{.*?\}", text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return fallback
