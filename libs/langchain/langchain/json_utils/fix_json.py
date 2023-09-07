import json
import re


def maybe_fix_json(text):
	try:
		text = text.strip()
		text = re.sub(",+\s*}$", "}", text)
		action_match1 = re.search(r"```(.*?)```?", text, re.DOTALL) or re.search(r"```json(.*?)```?", text, re.DOTALL) or re.search(r"```(.*?)$", text, re.DOTALL)
		# match {*}
		action_match2 = re.search(r"\{(.*?)\}", text, re.DOTALL) or re.search(r"\{(.*?)$", text, re.DOTALL)
		if action_match1 is not None:
			json_text = action_match1.group(1).strip()
		elif action_match2 is not None:
			json_text = action_match2.group(0).strip()
		else:
			json_text = None

		if json_text is not None:
			if json_text[-1] != "}":
				json_text += "}"
			json_text = re.sub(",+\s*}$", "}", text)
			try:
				data = json.loads(json_text)
			except json.JSONDecodeError as e:
				raise e
		else:
			return {}
		return data
	except Exception as e:
		raise e