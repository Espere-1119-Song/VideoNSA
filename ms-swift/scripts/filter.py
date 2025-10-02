import json
import os
from glob import glob

folder = "datasets/jsonl/keye"
for path in glob(os.path.join(folder, "*.jsonl")):
    tmp_path = path + ".tmp"
    with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            if "images" in data and data["images"]:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)