import asyncio
import websockets
import json
import httpx

async def test():
    async with websockets.connect("ws://localhost:8000/captions") as ws:
        # Trigger simulate
        async with httpx.AsyncClient() as client:
            await client.post("http://localhost:8000/api/simulate")
            
        while True:
            msg = await ws.recv()
            print("Received:", msg)
            if "TRANSCRIPT" in msg:
                break

asyncio.run(test())
