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

"""apiserver class."""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import FastAPI
from infscale.constants import APISERVER_PORT
from pydantic import BaseModel
from uvicorn import Config, Server

if TYPE_CHECKING:
    from infscale.controller.controller import Controller

_ctrl = None
app = FastAPI()


class ApiServer:
    """ApiServer class."""

    def __init__(self, ctrl: Controller, port: int = APISERVER_PORT):
        """Initialize an instance."""
        global _ctrl
        _ctrl = ctrl

        self.port = port

    async def run(self):
        """Run apiserver."""
        config = Config(
            app=app,
            host="0.0.0.0",
            port=self.port,
            loop=asyncio.get_event_loop(),
        )

        server = Server(config)
        await server.serve()


class ReqType(str, Enum):
    """Enum class for request type."""

    UNKNOWN = "unknown"
    SERVE = "serve"


class ServeSpec(BaseModel):
    """ServiceSpec model."""

    name: str
    model: str
    nfaults: int  # # of faults a serve should tolerate


class Response(BaseModel):
    """Response model."""

    message: str


@app.post("/models", response_model=Response)
async def serve(spec: ServeSpec):
    """Serve a model."""
    await _ctrl.handle_fastapi_request(ReqType.SERVE, spec)

    res = {"message": "started serving"}
    return res
