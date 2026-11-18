import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inspectPlayerView, movementReview } from './movementReview.ts';
import type { BodyLandmark, PlayerBox, PlayerTracks } from './playerTracks.ts';

const video = { width: 1280, height: 720, duration: .5 };
const fps = 20;

function clip(withOpponent = false): PlayerTracks {
  const frames = new Map<number, PlayerBox[]>();
  for (let frame = 0; frame < 10; frame++) {
    const x = .2 + frame * .01;
    const point = (name: BodyLandmark['name'], px: number, py: number, confidence = .9): BodyLandmark => ({
      name, position: { source: 'automatic', confidence, value: { x: px, y: py } },
    });
    frames.set(frame, [
      { trackId: 1, playerId: null, box: { x, y: .2, width: .24, height: .6 }, landmarks: [
        point('right-shoulder', x+.1, .26),
        point('right-wrist', x+.18, .34 + .01 * frame),
        point('right-ankle', x+.13, .77, frame === 4 ? .2 : .9),
      ] },
      { trackId: 2, playerId: null, box: { x: withOpponent ? .6 : .96, y: .1, width: withOpponent ? .2 : .04, height: .3 } },
    ]);
  }
  return { fps, firstTimestamp: 0, frames };
}

test('switches away from shot review with only one primary track and a person at the frame edge', () => {
  assert.deepEqual(inspectPlayerView(clip()), { trackId: 1, onePlayer: true });
  assert.deepEqual(inspectPlayerView(clip(true)), { trackId: 1, onePlayer: false });
});

test('measures swing relative to the shoulder and leaves weak ankle estimates blank', () => {
  const tracks = clip();
  const report = movementReview(tracks, 1, video, []);
  assert.equal(report.tested, 1);
  const [example] = report.windows;
  assert.ok(Math.abs(example.swing! - .08 / .6) < 1e-9);
  assert.ok(Math.abs(example.lateral - (.08 * (1280 / 720)) / .6) < 1e-9);
  assert.equal(example.feet, null);
  const cut = { timestamp: .2, sceneCut: true, prediction: null, uncertaintyRadiusPixels: null, reason: 'scene cut' };
  assert.deepEqual(movementReview(tracks, 1, video, [cut]).windows, []);
});
