import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inspectPlayerView, movementExamples, movementReview } from './movementReview.ts';
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

test('a consistently tracked background player does not hold up movement review', () => {
  const tracks = clip(true);
  for (const people of tracks.frames.values()) people[1].box.width = .08;
  assert.deepEqual(inspectPlayerView(tracks), { trackId: 1, onePlayer: true });
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

test('groups adjacent motion peaks and adds context from supported windows', () => {
  const windows = [.02, .16, .30, .14, .04, null, .01, .22, .02].map((swing, index) => ({
    start: index / 2.5, end: (index + 1) / 2.5, swing, feet: null, lateral: .01,
  }));
  const intervals = movementExamples(windows, 'swing').map(({ start, end }) => [start.toFixed(2), end.toFixed(2)]);
  assert.deepEqual(intervals, [['0.20', '1.85'], ['2.60', '3.45']]);
  assert.deepEqual(movementExamples(windows, 'feet'), []);
});

test('movement replay stops at a missing tracking window or a cut', () => {
  const windows = [
    { start: 0, end: .4, swing: null, feet: null, lateral: .01 },
    { start: .4, end: .8, swing: .31, feet: null, lateral: .01 },
    { start: 1.2, end: 1.6, swing: .24, feet: null, lateral: .01 },
  ];
  assert.deepEqual(movementExamples(windows, 'swing'), [{ start: .4, end: .8 }, { start: 1.2, end: 1.6 }]);
});
