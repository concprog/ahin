import json
import base64
tokenizer = "models/vocab.json"
with open(tokenizer, "r") as f:
    contents: dict[str, int] = json.load(f)
    #  tokens = {
    #      base64.b64decode(token): int(rank)
    #      for token, rank in (line.split() for line in contents.splitlines() if line)
    #  }
    tokens = {
        base64.b64encode(bytes(token, encoding="utf-8")): int(rank)
        for token, rank in contents.items()
    }
name = "hindi-vocab"
with open(f"models/{name}-tokens.txt", "w") as f:
    for t, i in tokens.items():
        f.write(f"{t.decode("utf-8")} {i}\n")