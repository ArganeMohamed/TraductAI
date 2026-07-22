import json

with open("data/vocabulary.json", "r", encoding="utf-8") as f:
    vocab_to_id = json.load(f)

id_to_token = {}

for token, idx in vocab_to_id.items():
    id_to_token[str(idx)] = token

with open("data/id_to_token.json", "w", encoding="utf-8") as f:
    json.dump(id_to_token, f, ensure_ascii=False, indent=4)

print(f"ID to token size: {len(id_to_token)}")

print("\nFirst 20 mappings:")
for i, (idx, token) in enumerate(id_to_token.items()):
    print(f"{idx} -> {token}")

    if i == 19:
        break