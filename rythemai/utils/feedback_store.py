import sqlite3
from datetime import datetime

class FeedbackStore:
    """
    Stores user feedback for generated tracks for model improvement.
    """
    def __init__(self, db_path="feedback.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    track_id TEXT,
                    rating INTEGER,
                    comments TEXT,
                    timestamp TEXT
                )
            ''')

    def add_feedback(self, track_id: str, rating: int, comments: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO feedback (track_id, rating, comments, timestamp) VALUES (?, ?, ?, ?)",
                (track_id, rating, comments, datetime.utcnow().isoformat())
            )