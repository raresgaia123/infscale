# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from multiprocessing import connection

import torch.multiprocessing as mp
from infscale import get_logger

logger = get_logger()


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process


class JobMonitor:
    """JobMonitor class."""

    def __init__(self, metadata: WorkerMetaData):
        self.metadata = metadata

    def message_listener(self):
        """Asynchronous parent listener to handle communication with workers."""
        loop = asyncio.get_event_loop()
        for worker_data in self.metadata.values():
            loop.add_reader(
                worker_data.pipe.fileno(), self.on_read_ready, worker_data.pipe, loop
            )

    def on_read_ready(
        self, pipe: connection.Connection, loop: asyncio.AbstractEventLoop
    ):
        if pipe.poll():  # Check if there's data to read
            try:
                message = pipe.recv()  # Receive the message
                if message:
                    print(f"Parent received: {message}")
            except EOFError:
                loop.remove_reader(pipe.fileno())  # Clean up the reader
