from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from eval.experimental.memory_lifecycle import EpisodeStore, MemoryEpisode


def test_episode_round_trip_uses_secure_files(tmp_path) -> None:
    store = EpisodeStore(tmp_path / "episodes")
    now = datetime.now(UTC)
    episode = MemoryEpisode(str(uuid4()), now, now + timedelta(seconds=1), "validated summary", ("fact",))
    store.persist(episode)
    loaded = EpisodeStore(tmp_path / "episodes").get(episode.id)
    assert loaded == episode
    assert (tmp_path / "episodes").stat().st_mode & 0o777 == 0o700
    assert next((tmp_path / "episodes").glob(f"{episode.id}.json")).stat().st_mode & 0o777 == 0o600
