import assert from 'node:assert/strict';
import { test } from 'node:test';
import { automaticRecovery } from './automaticRecovery.ts';
import type { PlayerTracks } from './playerTracks.ts';
import type { BallObservation } from './session.ts';

const fps = 25;
const xPositions = [.5, .44, .38, .32, .26, .2, .26, .32, .38, .44, .5,
  .56, .62, .68, .74, .8, .74, .68, .62, .56, .5];
const observations: BallObservation[] = xPositions.map((x, frame) => ({
  timestamp: frame / fps, position: { source: 'automatic', confidence: .9, value: { x, y: .5 } },
}));

function playerTracks(): PlayerTracks {
  return { fps, firstTimestamp: 0, frames: new Map(xPositions.map((_, frame) => [frame, [
    { trackId: 1, playerId: null, box: { x: .14 + .004 * frame, y: .3, width: .16, height: .5 } },
    { trackId: 2, playerId: null, box: { x: .72, y: .3, width: .16, height: .5 } },
  ]])) };
}

test('tracks recovery without named players or manual contacts, and abstains at a scene cut', () => {
  const tracks = playerTracks();
  const { players, turns } = automaticRecovery(observations, fps, tracks, []);
  assert.deepEqual(players.map(player => player.contacts), [1, 1]);
  assert.deepEqual(turns.map(turn => [turn.timestamp, turn.trackId]), [[.2, 1], [.6, 2]]);
  const [example] = players[0].examples;
  assert.equal(example.contact, .2);
  assert.equal(example.nextContact, .6);
  assert.ok(Math.abs(example.motion!.horizontalWidths - .075) < 1e-9);

  const cut = { timestamp: .4, sceneCut: true, reason: 'scene-cut', prediction: null, uncertaintyRadiusPixels: null };
  const report = automaticRecovery(observations, fps, tracks, [cut]);
  assert.deepEqual(report.players[0].examples, []);
});

test('keeps missing movement and missing hitter evidence out of the measured count', () => {
  const tracks = playerTracks();
  tracks.frames.delete(13);
  const report = automaticRecovery(observations, fps, tracks, []);
  assert.equal(report.players[0].examples[0].motion, null);

  tracks.frames.delete(15);
  const unknown = automaticRecovery(observations, fps, tracks, []);
  assert.deepEqual(unknown.players[0].examples, []);
  assert.equal(unknown.unassignedContacts, 1);
  assert.deepEqual(unknown.turns.map(turn => turn.trackId), [1, null]);
});

test('keeps observed contact steps available with only ball tracks', () => {
  const report = automaticRecovery(observations, fps, null, []);
  assert.deepEqual(report.turns.map(turn => [turn.timestamp, turn.trackId]), [[.2, null], [.6, null]]);
  assert.deepEqual(report.players, []);
});

test('movement uses only frames before the estimated contact and rejects a clipped box', () => {
  const tracks = playerTracks();
  const offset = observations.map(sample => ({ ...sample, timestamp: sample.timestamp + .019 }));
  const report = automaticRecovery(offset, fps, tracks, []);
  const [example] = report.players[0].examples;
  assert.equal(example.motion?.seconds, .08);
  assert.ok(Math.abs(example.motion!.horizontalWidths - .05) < 1e-9);
  tracks.frames.get(13)![0].box.x = 0;
  assert.equal(automaticRecovery(offset, fps, tracks, []).players[0].examples[0].motion, null);
});

test('finds the players by their ball contacts when a spectator is also tracked', () => {
  const tracks = playerTracks();
  for (const players of tracks.frames.values()) {
    players.push({ trackId: 3, playerId: null, box: { x: .4, y: 0, width: .18, height: .24 } });
  }
  const report = automaticRecovery(observations, fps, tracks, []);
  assert.deepEqual(report.players.map(player => player.trackId), [1, 2]);
  assert.equal(report.players[0].examples.length, 1);
});
