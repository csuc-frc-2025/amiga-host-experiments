"""Modified from camera service client example for multi-camera streaming."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.stamp import get_stamp_by_semantics_and_clock_type
from farm_ng.core.stamp import StampSemantics

from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_service_pb2 import SubscribeRequest
from farm_ng.core.stamp import get_stamp_by_semantics_and_clock_type
from farm_ng.core.stamp import StampSemantics


class CameraMultiStreamer:
    """A service multi-streamer for all camera streams."""

    def __init__(self, service_config: EventServiceConfigList) -> None:
        """Initialize the camera multi-streamer.

        Args:
            service_config: The service config.
        """
        self.service_config = service_config
        self.clients: dict[str, EventClient] = {}

        # populate the clients
        config: EventServiceConfig
        for config in self.service_config.configs:
            if not config.port:
                self.subscriptions = config.subscriptions
                continue
            self.clients[config.name] = EventClient(config)


    async def _subscribe(self, subscription: SubscribeRequest) -> None:
        # the client name is the last part of the query
        client_name: str = subscription.uri.query.split("=")[-1]
        client: EventClient = self.clients[client_name]
        # subscribe to the event
        # NOTE: set decode to True to decode the message
        async for event, message in client.subscribe(subscription, decode=True):
            # decode the message type
            message_type = event.uri.query.split("&")[0].split("=")[-1]
            print(f"Received event from {client_name}{event.uri.path}: {message_type}")

            if "OakFrame" in event.uri.query:
                # Find the monotonic driver receive timestamp, or the first timestamp if not available.
                # stamp = (
                #     get_stamp_by_semantics_and_clock_type(event, StampSemantics.DRIVER_RECEIVE, "monotonic")
                #     or event.timestamps[0].stamp
                # )

                # # print the timestamp and metadata
                # print(f"Timestamp: {stamp}\n")
                # print(f"Meta: {message.meta}")
                # print("###################\n")

                # cast image data bytes to numpy and decode
                image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)
                if event.uri.path == "/disparity":
                    image = cv2.applyColorMap(image * 3, cv2.COLORMAP_JET)

                # visualize the image
                cv2.namedWindow(f"{client_name}{event.uri.path}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"{client_name}{event.uri.path}", image)
                cv2.waitKey(1)
            else:
                pass


    async def run(self) -> None:
        # start the subscribe routines
        tasks: list[asyncio.Task] = []
        for subscription in self.subscriptions:
            tasks.append(asyncio.create_task(self._subscribe(subscription)))
        # wait for the subscribe routines to finish
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python main.py", description="Amiga camera-stream.")
    parser.add_argument("--service-config", type=Path, required=True, help="The camera config.")
    args = parser.parse_args()

    # create a client to the camera service
    service_config: EventServiceConfigList = proto_from_json_file(args.service_config, EventServiceConfigList())

    # create the multi-client subscriber
    subscriber = CameraMultiStreamer(service_config)

    asyncio.run(subscriber.run())
