"""Run sequential, same-footage baseline versus original-image crop experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from evaluate_manual_labels import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-video', type=Path, required=True)
    parser.add_argument('--new-video', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('temporal_track.py')
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    report = {}
    for name, video, start, end in [('720p', args.old_video, 0, 4), ('4k', args.new_video, 6, 10)]:
        for mode in ['baseline', 'native-zoom']:
            destination = args.output_dir/f'{name}-{mode}.json'
            command = [sys.executable, str(script), '--base-model', '--video', str(video), '--start', str(start), '--end', str(end), '--output', str(destination)]
            if mode == 'native-zoom':
                command.append('--native-zoom')
            print(f'Running {name} {mode}, {start}–{end} seconds', flush=True)
            subprocess.run(command, check=True)
            data = json.loads(destination.read_text(encoding='utf-8'))
            result = dict(video=data['video'], fps=data['fps'], summary=data['summary'])
            if name == '720p':
                result['humanEvaluation'] = evaluate(data, labels, -1)
            report[f'{name}-{mode}'] = result
            (args.output_dir/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
