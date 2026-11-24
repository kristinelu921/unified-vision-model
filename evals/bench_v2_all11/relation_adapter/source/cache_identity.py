"""Content-addressed conditioning identity; query targets cannot enter this API."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConditioningIdentity:
    weights_sha256: str
    official_source_sha256: str
    runner_source_sha256: str
    environment_sha256: str
    config_sha256: str
    preprocessing_sha256: str
    prompt_tokenization_sha256: str
    precision_backend: str
    ordered_demo_input_sha256: tuple[str, ...]
    ordered_demo_output_sha256: tuple[str, ...]
    query_input_sha256: str
    seed: int

    def digest(self) -> str:
        obj = asdict(self)
        for key, value in obj.items():
            if key.endswith('sha256'):
                values = value if isinstance(value, (tuple, list)) else [value]
                if not values or any(len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in values):
                    raise ValueError('missing or invalid SHA-256 dependency: ' + key)
        if len(self.ordered_demo_input_sha256) != len(self.ordered_demo_output_sha256):
            raise ValueError('demonstration input/output count differs')
        if type(self.seed) is not int or not self.precision_backend:
            raise ValueError('seed and precision/backend are mandatory')
        return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
