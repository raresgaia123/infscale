"""Worker class."""

from multiprocessing.connection import Connection

from infscale import get_logger

logger = get_logger(__name__)


class Worker:
    """Worker class."""

    def __init__(self, local_rank: int, conn: Connection):
        """Initialize an instance."""
        self.local_rank = local_rank
        self.conn = conn

    def run(self) -> None:
        """Run the worker."""
        logger.info(f"worker {self.local_rank}")
