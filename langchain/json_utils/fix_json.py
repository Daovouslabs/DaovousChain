import json
import re


def maybe_fix_json(text):
	text = text.strip()
	try:
		data = json.loads(text)
	except json.JSONDecodeError as e:
		json_match = re.search(r"\{(.*?)\}", text, re.DOTALL)
		text = json_match.group(0).strip()
		try:
			data = json.loads(text)
		except json.JSONDecodeError as e2:
			raise e2
	return data