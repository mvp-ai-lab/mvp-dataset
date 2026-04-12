import os
import shutil
import time
from pathlib import Path


class FileBarrierTimeout(TimeoutError):
    pass


class FileBarrier:
    """File-system based barrier for cross-process synchronization.

    All ranks call :meth:`wait`; the call blocks until every rank has arrived,
    then all ranks are released and the barrier directory is cleaned up.
    """

    def __init__(
        self,
        shared_path: str,
        world_size: int,
        rank: int,
        poll_interval: float = 0.2,
    ):
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        if not (0 <= rank < world_size):
            raise ValueError("rank must be in [0, world_size)")

        self.root = Path(shared_path)
        self.world_size = world_size
        self.rank = rank
        self.poll_interval = poll_interval

        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_touch(path: Path) -> bool:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(path), flags)
        except FileExistsError:
            return False
        os.close(fd)
        return True

    def wait(self, timeout: float | None = None):
        arrive_file = self.root / f"arrive_{self.rank}"
        release_file = self.root / "release"
        done_file = self.root / f"done_{self.rank}"

        deadline = None if timeout is None else time.time() + timeout

        # Phase 1: arrive and wait for release
        self._atomic_touch(arrive_file)

        while not release_file.exists():
            arrived = len(list(self.root.glob("arrive_*")))
            if arrived >= self.world_size:
                self._atomic_touch(release_file)
                break

            if deadline is not None and time.time() > deadline:
                raise FileBarrierTimeout(
                    f"timeout waiting for release: arrived={arrived}/{self.world_size}, rank={self.rank}"
                )
            time.sleep(self.poll_interval)

        # Phase 2: mark done and let the last rank clean up
        self._atomic_touch(done_file)

        done_count = len(list(self.root.glob("done_*")))
        if done_count >= self.world_size:
            shutil.rmtree(self.root, ignore_errors=True)
