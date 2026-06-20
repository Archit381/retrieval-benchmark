import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SetInfo:
    name: str
    model_type: str
    role: str
    is_multivector: bool
    dim: int
    count: int
    dtype: str
    filename: str
    ids: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    repo_id: str
    sets: dict[str, SetInfo]
    query_ids: list[str]
    doc_ids: list[str]
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {
            "repo_id": self.repo_id,
            "query_ids": self.query_ids,
            "doc_ids": self.doc_ids,
            "notes": self.notes,
            "sets": {k: asdict(v) for k, v in self.sets.items()},
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Manifest":
        d = json.loads(s)
        sets = {k: SetInfo(**v) for k, v in d["sets"].items()}
        return cls(
            repo_id=d["repo_id"],
            sets=sets,
            query_ids=d["query_ids"],
            doc_ids=d["doc_ids"],
            notes=d.get("notes", {}),
        )
