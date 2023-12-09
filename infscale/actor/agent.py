"""Agent class."""

import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing import connection

import torch
from infscale import get_logger
from infscale.actor.worker import Worker

logger = get_logger(__name__)


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process


class Agent:
    """Agent class manages workers in a node."""

    def __init__(self):
        """Initialize the agent instance."""
        # TODO: there can be more than one worker per GPU
        #       if resource (gpu memory, gpu cycle) are available
        #       explore this possibility later
        # one worker per GPU
        self.n_workers = torch.cuda.device_count()
        self._workers: dict[int, WorkerMetaData] = {}

    def run(self):
        """Start the agent."""
        logger.info("run agent")

        self.launch()

    def launch(self):
        """Launch workers."""
        ctx = mp.get_context("spawn")

        for local_rank in range(self.n_workers):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

            pipe, child_pipe = ctx.Pipe()
            w = Worker(local_rank, child_pipe)
            process = mp.Process(target=w.run, args=(), daemon=True)
            self._workers[local_rank] = WorkerMetaData(pipe, process)

        os.environ.pop("CUDA_VISIBLE_DEVICES")

    def configure(self):
        """Configure workers."""
        pass

    def terminate(self):
        """Terminate workers."""
        logger.info("terminate workers")

        for rank, wmd in self._workers.items():
            logger.info(f"terminate worker {rank}")
            wmd.process.terminate()

    def monitor(self):
        """Monitor workers."""
        pass
