"""Fetch the pinned official source and checkpoint into the experiment folder."""
from pathlib import Path
import subprocess
import urllib.request

root = Path(__file__).resolve().parent
vendor = root/'vendor/RacketVision'
revision = 'c44af2a08524d3cb54d818f19686f4cdea4d2793'
if not vendor.exists():
    subprocess.run(['git','clone','https://github.com/OrcustD/RacketVision.git',str(vendor)],check=True)
subprocess.run(['git','-C',str(vendor),'checkout',revision],check=True)
weights = root/'balltrack_best.pth'
if not weights.exists():
    urllib.request.urlretrieve('https://huggingface.co/linfeng302/RacketVision-Models/resolve/main/checkpoints/balltrack_best.pth',weights)
print('Source and checkpoint ready. Run detect.py with your local video.')
