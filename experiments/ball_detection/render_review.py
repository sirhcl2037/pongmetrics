"""Render the saved audit on the original frames, without running inference again."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output')
args = parser.parse_args()
audit = json.loads((args.output/'audit.json').read_text())
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
images = []
for item in audit:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.resize(frame,(640,360))
    if item['position']:
        p = item['position']
        cv2.circle(frame,(round(p['x']*640),round(p['y']*360)),5,(0,255,255),1)
    cv2.putText(frame,f"{item['timestamp']:.2f}s {item['reason']}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
    images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
cap.release()
images[0].save(args.output/'review.gif',save_all=True,append_images=images[1:],duration=round(1000/fps),loop=0)
print(args.output/'review.gif')
