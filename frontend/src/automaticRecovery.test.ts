import assert from 'node:assert/strict';
import { test } from 'node:test';
import { automaticRecovery } from './automaticRecovery.ts';
import type { PlayerTracks } from './playerTracks.ts';
import type { BallObservation } from './session.ts';

const fps = 25;
const video = { width: 1280, height: 720 };
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
  const { players } = automaticRecovery(observations, fps, tracks, video, []);
  assert.deepEqual(players.map(player => player.contacts), [1, 1]);
  const [example] = players[0].examples;
  assert.equal(example.contact, .2);
  assert.equal(example.nextContact, .6);
  assert.ok(Math.abs(example.boxWidths! - .075) < 1e-9);

  const cut = { timestamp: .4, sceneCut: true, reason: 'scene-cut', prediction: null, uncertaintyRadiusPixels: null };
  const report = automaticRecovery(observations, fps, tracks, video, [cut]);
  assert.deepEqual(report.players[0].examples, []);
});

test('keeps missing movement and missing hitter evidence out of the measured count', () => {
  const tracks = playerTracks();
  tracks.frames.delete(13);
  const report = automaticRecovery(observations, fps, tracks, video, []);
  assert.equal(report.players[0].examples[0].boxWidths, null);

  tracks.frames.delete(15);
  const unknown = automaticRecovery(observations, fps, tracks, video, []);
  assert.deepEqual(unknown.players[0].examples, []);
  assert.equal(unknown.unassignedContacts, 1);
});

test('finds the players by their ball contacts when a spectator is also tracked', () => {
  const tracks = playerTracks();
  for (const players of tracks.frames.values()) {
    players.push({ trackId: 3, playerId: null, box: { x: .4, y: 0, width: .18, height: .24 } });
  }
  const report = automaticRecovery(observations, fps, tracks, video, []);
  assert.deepEqual(report.players.map(player => player.trackId), [1, 2]);
  assert.equal(report.players[0].examples.length, 1);
});
