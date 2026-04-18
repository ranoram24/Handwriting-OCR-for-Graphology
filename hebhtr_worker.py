"""
HebHTR batch worker — runs inside the Docker container.
Usage: python /app/worker.py /crops
Loads the model ONCE, processes all crops, prints JSON {filename: text}.
"""
import sys, json, os, io, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Patch open() to always use utf-8 (charList.txt contains Hebrew)
import builtins
_real_open = builtins.open
def _utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs and len(args) < 4:
        kwargs['encoding'] = 'utf-8'
    return _real_open(*args, **kwargs)
builtins.open = _utf8_open

sys.path.insert(0, '/app')

import cv2
import tensorflow as tf
tf.reset_default_graph()

from predictWord import getModel, infer
from processFunctions import preprocessImageForPrediction
from Model import Model

input_dir = sys.argv[1]
fnames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.png'))

results = {}

try:
    model = getModel(decoder_type='bestpath')
except Exception as e:
    print(json.dumps({f: f'MODEL_LOAD_ERROR:{e}' for f in fnames}, ensure_ascii=False))
    sys.exit(1)

for fname in fnames:
    path = os.path.join(input_dir, fname)
    try:
        img = cv2.imread(path)
        if img is None:
            results[fname] = ''
            continue
        # HebHTR expects grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = infer(model, img)
        # infer() returns a list of best-path results
        if isinstance(text, list):
            text = text[0] if text else ''
        results[fname] = text.strip() if text else ''
    except Exception as e:
        results[fname] = f'ERROR:{e}'
        traceback.print_exc(file=sys.stderr)

print(json.dumps(results, ensure_ascii=False))
