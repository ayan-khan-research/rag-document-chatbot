from typing import List, Tuple


class AnalyticsTracker:
    """Tracks query performance and document statistics for the dashboard."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.documents: List[Tuple[str, int]] = []
        self.query_log: List[Tuple[str, float, float]] = []
        self.response_times: List[float] = []
        self.confidences: List[float] = []
        self.total_queries: int = 0

    def add_document(self, name: str, chunks: int):
        self.documents.append((name, chunks))

    def add_query(self, query: str, response_time: float, confidence: float):
        self.total_queries += 1
        short_query = query[:60] + "..." if len(query) > 60 else query
        self.query_log.append((short_query, round(response_time, 2), confidence))
        self.response_times.append(response_time)
        self.confidences.append(confidence)

    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)
