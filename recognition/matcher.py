"""Visitor identity matching backed by the persistent embedding gallery."""

import numpy as np

from database.db import VisitorDatabase


class IdentityMatcher:
    def __init__(self, database: VisitorDatabase, threshold: float) -> None:
        self.database = database
        self.threshold = threshold

    def identify(self, embedding: np.ndarray) -> tuple[str, bool, float]:
        return self.database.identify_or_register(embedding, self.threshold)
