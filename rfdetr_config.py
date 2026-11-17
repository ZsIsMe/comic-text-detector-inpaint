"""Koharu Layout HighRes settings shared by download, loading and inference.

Keep this module standard-library-only so bootstrap can import it before the
application's virtual environment and model dependencies are installed.
"""

from pathlib import Path


MODEL_REPO = 'mayocream/koharu-layout-highres'
MODEL_URL = f'https://huggingface.co/{MODEL_REPO}/resolve/main/model.safetensors'
MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'koharu-layout-highres'
MODEL_FILE = MODEL_DIR / 'model.safetensors'
MODEL_SHA256 = '03de2f77281974f5084cdd8652ace2b55dc15ac0378aed66654e62537cdda255'

CLASS_NAMES = ('text', 'onomatopoeia', 'bubble', 'panel')
CLASS_THRESHOLDS = {'text': 0.25, 'onomatopoeia': 0.30, 'bubble': 0.50, 'panel': 0.40}
RESOLUTION = 1920
NUM_SELECT = 160
