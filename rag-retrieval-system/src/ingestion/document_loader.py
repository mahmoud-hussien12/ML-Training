from pathlib import Path
from typing import List, Dict


def load_text_files(data_dir: str) -> List[Dict]:

    documents = []

    for path in Path(data_dir).glob("*.txt"):
        text = path.read_text(encoding="utf-8")

        documents.append({
            "text": text,
            "source": path.name,
        })

    return documents
