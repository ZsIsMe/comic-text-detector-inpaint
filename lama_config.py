"""Standard-library-only settings for the PSD export inpainting model."""

from pathlib import Path


MODEL_NAME = 'lama_large_512px'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / f'{MODEL_NAME}.ckpt'
MODEL_URL = f'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/{MODEL_NAME}.ckpt'
MODEL_SHA256 = '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935'
MAX_INPAINT_SIZE = 768
PAD_MODULO = 64

MERGE_DISTANCE = 24
CONTEXT_RATIO = 1.7
MASK_EXPAND = 8
REPAIR_POLICY = 'manga-local-crop768-grow8-v3'
