import tempfile
import unittest
from pathlib import Path
from src.database import NewsDatabase

class DatabaseTests(unittest.TestCase):
    def test_url_upsert_and_run_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = NewsDatabase(str(Path(tmp) / "news.db"))
            run_id = db.start_run()
            item = {"title": "测试新闻", "url": "https://example.com/a", "source": "测试", "score": 8}
            first = db.save_news(item)
            second = db.save_news({**item, "title": "更新标题"})
            self.assertEqual(first, second)
            self.assertEqual(db.count_news(), 1)
            db.finish_run(run_id, status="success", collected_count=1, unique_count=1)
            with db.connection() as conn:
                row = conn.execute("SELECT status,collected_count FROM runs WHERE id=?", (run_id,)).fetchone()
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["collected_count"], 1)

if __name__ == "__main__":
    unittest.main()
