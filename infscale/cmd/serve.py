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

"""serve subcommand."""
import click
import infscale.client as client
from infscale.constants import APISERVER_PORT, LOCALHOST


@click.command()
@click.option("--host", default=LOCALHOST, help="Controller's IP or hostname")
@click.option(
    "--port", default=APISERVER_PORT, help="Controller's apiserver port number"
)
@click.argument("specfile")
def serve(host: str, port: int, specfile: str) -> None:
    """Serve model based on config yaml file."""
    client.serve(host, port, specfile)
