"""Tests for Phase 3: Backend & Storage."""

from datetime import datetime

from app.storage import StorageManager
from inference.qnn_session import load_config


def test_storage_manager(tmp_path):
    """Verify SQLite storage creation and retrieval."""
    cfg = load_config()
    db_path = tmp_path / "test_transcripts.db"
    cfg["storage"] = {"db_path": str(db_path), "max_history_entries": 10}
    
    storage = StorageManager(cfg)
    
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    # Save test transcript
    row_id = storage.save_transcript(timestamp, "Test sentence.", 150, "npu")
    assert row_id is not None
    
    # Retrieve history
    history = storage.get_history()
    assert len(history) == 1
    assert history[0]["text"] == "Test sentence."
    assert history[0]["device"] == "npu"
    assert history[0]["latency_ms"] == 150
