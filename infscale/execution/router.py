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

"""Router class."""
import asyncio
import random

import torch
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender
from infscale.execution.world import WorldInfo
from torch.distributed import WorldManager

DEFAULT_QUEUE_SIZE = 3

logger = get_logger()


class Router:
    """Router class."""

    def __init__(
        self,
        world_manager: WorldManager,
        world_info_list: list[WorldInfo],
        spec: ServeConfig,
        device=torch.device("cpu"),
    ):
        """Initialize Router instance."""
        self.world_manager = world_manager
        self.device = device

        self._rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline
        self._tx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline

        # a collection of receivers that receive data from me
        self.receivers: list[WorldInfo] = []
        self.__tx_qs: dict[WorldInfo, asyncio.Queue] = {}

        # a collection of senders that send data to me
        self.senders: list[WorldInfo] = []
        self.__rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)

        for world_info in world_info_list:
            if world_info.me == 0:  # I am a sender to other
                self.receivers.append(world_info)
                self.__tx_qs[world_info] = asyncio.Queue(DEFAULT_QUEUE_SIZE)
            else:  # I am a receiver from other
                self.senders.append(world_info)

    @property
    def rx_q(self):
        """Return receiver queue."""
        return self._rx_q

    @property
    def tx_q(self):
        """Return transmit queue."""
        return self._tx_q

    def prepare(self):
        """Create asyncio tasks for sending and receiving."""
        _ = asyncio.create_task(self._send_arbiter())
        _ = asyncio.create_task(self._recv_arbiter())

        for world_info in self.receivers:
            _ = asyncio.create_task(self._send(world_info))

        for world_info in self.senders:
            _ = asyncio.create_task(self._recv(world_info))

    async def _recv(self, world_info: WorldInfo):
        logger.debug(
            f"start to receive tensors from {world_info.other} in world {world_info.name}"
        )
        receiver = TensorReceiver(
            self.world_manager.communicator,
            world_info.name,
            world_info.other,
            self.device,
        )
        logger.debug("created tensor receiver")

        while True:
            logger.debug("calling receiver.recv")
            tensors, index = await receiver.recv()
            logger.debug(f"received tensor {index}")
            await self.__rx_q.put((tensors, index))
            logger.debug(f"put tensors {index} into __rx_q")

    async def _send(self, world_info: WorldInfo):
        logger.debug(
            f"start to send tensors to {world_info.other} in world {world_info.name}"
        )
        sender = TensorSender(
            self.world_manager.communicator,
            world_info.name,
            world_info.other,
            self.device,
        )
        logger.debug("created tensor sender")
        tx_q = self.__tx_qs[world_info]
        logger.debug("acquired tx q")

        while True:
            tensor, seqno = await tx_q.get()
            logger.debug(f"got tensor {seqno} from __tx_q")
            await sender.send(tensor, seqno)
            logger.debug(f"sent tensor {seqno}")

    async def _recv_arbiter(self):
        logger.debug("start recv_arbiter")
        while True:
            tensor, seqno = await self.__rx_q.get()
            logger.debug(f"fetched tensor {seqno} from __rx_q")
            # TODO: introduce a prioritization policy
            await self._rx_q.put((tensor, seqno))
            logger.debug("put tensor to _rx_q")

    async def _send_arbiter(self):
        logger.debug("start send_arbiter")
        while True:
            tensor, seqno = await self._tx_q.get()
            logger.debug(f"fetched tensor {seqno} from _tx_q")
            # TODO: introduce a prioritization policy
            #       current default policy is to choose receiving rank randomly

            # TODO: choosing a rank randomly by converting dictionary keys into
            #       a list can be a performance bottleneck; look into it later.
            world_info = random.choice(list(self.__tx_qs.keys()))
            logger.debug(f"world name: {world_info.name}")
            logger.debug(f"receiver rank: {world_info.other}")

            await self.__tx_qs[world_info].put((tensor, seqno))
            logger.debug(f"put tensor {seqno} to __tx_q for {world_info.other}")
