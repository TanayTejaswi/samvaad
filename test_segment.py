import asyncio
import numpy as np
from app.server import handle_speech_segment

segment = np.zeros(483776, dtype=np.float32)
handle_speech_segment(segment)
print("Done!")
