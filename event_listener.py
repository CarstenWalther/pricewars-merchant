import websockets
from typing import Callable


async def sales_event_listener(token: str, callback: Callable) -> None:
    async with websockets.connect('ws://localhost:8765') as websocket:
        await websocket.send(token)

        async for message in websocket:
            await callback(message)
