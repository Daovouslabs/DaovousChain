import json

def maybe_fix_json(text):
	text = text.strip()
	try:
		data = json.loads(text)
	except json.JSONDecodeError as e:
		if text.startswith('```'):
			text = text[3:]
		if text.endswith('```'):
			text = text[:-3]
		try:
			data = json.loads(text)
		except json.JSONDecodeError as e2:
			raise e2
	return data