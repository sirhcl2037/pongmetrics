import { test } from 'node:test';
import assert from 'node:assert/strict';
import { defaultPlayers, parseSession, type Session, type Shot } from './session.ts';
import type { ReviewedRally } from './rallies.ts';
import { makeRecoveryJudgment, recoverySummary, recoveryWindows } from './recovery.ts';

const shot = (id: number, time: number, playerId: string | null = 'player-1'): Shot =>
  ({ id, time, playerId, source: 'manual', wing: 'FH', stroke: 'Drive' });
const rally: ReviewedRally = { id: 'r1', start: 0, end: 5, winnerId: null, focus: null, note: '' };
const shots = [shot(1, 1), shot(2, 2, 'player-2'), shot(3, 3), shot(4, 4, 'player-2')];
const session: Session = { version: 2, video: { name: 'test.mp4', size: 100, width: 1280, height: 720, duration: 10 },
  fps: 25, coordinateSystem: 'normalized-video-image', calibration: null, shots, ballObservations: [],
  players: defaultPlayers(), selectedPlayerId: 'player-1', rallies: [rally] };
const windows = (rows = shots, rallies = [rally]) => recoveryWindows(rallies, rows, 'player-1');

test('pairs sorted contacts within each rally without skipping unknown or repeated players', () => {
  assert.deepEqual(windows([...shots].reverse()).map(w => w.opponent?.id), [2, 4]);
  for (const playerId of [null, 'player-1']) {
    assert.equal(windows([shots[0], shot(9, 1.5, playerId), ...shots.slice(1)])[0].opponent, null);
  }
  assert.equal(windows(shots, [{ ...rally, end: 2 }])[0].opponent, null);
  assert.equal(windows([shots[0], shot(8, 1, 'player-2'), ...shots.slice(1)])[0].opponent, null);
  assert.equal(windows([shots[0], shots[1], shot(8, 2), ...shots.slice(2)])[0].opponent, null);
});

test('missing contact permits only unclear and records its evidence reason', () => {
  const window = windows([shots[0]])[0];
  assert.throws(() => makeRecoveryJudgment(window, 'late'), /Tag the next/);
  assert.throws(() => makeRecoveryJudgment(window, 'ready'), /Tag the next/);
  assert.equal(makeRecoveryJudgment(window, 'unclear', 'feet-hidden').reason, 'missing-contact');
});

test('manual judgments round-trip without changing automatic contact provenance; old sessions still load', () => {
  const automatic = shots.map(s => ({ ...s, source: 'automatic' as const,
    playerIdentity: { value: s.playerId!, source: 'automatic' as const, confidence: .7 } }));
  const recovery = [makeRecoveryJudgment(windows(automatic)[0], 'late')];
  const saved = { ...session, shots: automatic, rallies: [{ ...rally, recovery }] };
  assert.deepEqual(parseSession(JSON.stringify(saved)), saved);
  assert.equal(recovery[0].source, 'manual');
  assert.equal(parseSession(JSON.stringify(session)).rallies?.[0].recovery, undefined);
});

test('tag deletion, retiming, player correction and range edits exclude stale judgments but preserve snapshots', () => {
  const recovery = [makeRecoveryJudgment(windows()[0], 'late')];
  const reviewed = { ...rally, recovery };
  for (const changed of [shots.slice(1), shots.filter(s => s.id !== 2),
    shots.map(s => s.id === 2 ? { ...s, time: 2.1 } : s),
    shots.map(s => s.id === 1 ? { ...s, playerId: 'player-2' } : s)]) {
    const restored = parseSession(JSON.stringify({ ...session, shots: changed, rallies: [reviewed] }));
    assert.deepEqual(restored.rallies?.[0].recovery, recovery);
    assert.equal(recoverySummary(windows(changed, restored.rallies)).late, 0);
  }
  assert.equal(recoverySummary(windows(shots, [{ ...reviewed, start: 1.5 }])).late, 0);
  assert.equal(recoverySummary(windows(shots, [{ ...reviewed, end: 2 }])).late, 0);
});

test('coverage and late rate report unclear, missing and unreviewed without inflating usable evidence', () => {
  const rows = [...shots, shot(5, 6), shot(6, 7, 'player-2'), shot(7, 8)];
  const range = { ...rally, end: 9 };
  const ws = windows(rows, [range]);
  const recovery = [makeRecoveryJudgment(ws[0], 'late'), makeRecoveryJudgment(ws[1], 'ready'),
    makeRecoveryJudgment(ws[2], 'unclear', 'feet-hidden')];
  assert.deepEqual(recoverySummary(windows(rows, [{ ...range, recovery }])), {
    total: 4, ready: 1, late: 1, unclear: 1, unreviewed: 1, usable: 2, missingContact: 1, lateRate: .5,
  });
  assert.equal(recoverySummary([]).lateRate, null);
});

test('import rejects malformed or nonmanual judgments and duplicate shot reviews', () => {
  const good = makeRecoveryJudgment(windows()[0], 'ready');
  const restore = (recovery: unknown) => parseSession(JSON.stringify({ ...session, rallies: [{ ...rally, recovery }] }));
  for (const bad of [null, { ...good, source: 'automatic' }, { ...good, outcome: 'good' },
    { ...good, reason: 'feet-hidden' }, { ...good, outcome: 'unclear' }, { ...good, opponent: null },
    { ...good, shot: { ...good.shot, playerId: 'missing' } },
    { ...good, opponent: { ...good.opponent, time: .5 } }]) {
    assert.throws(() => restore([bad]), /Invalid recovery/);
  }
  assert.throws(() => restore([good, good]), /Invalid recovery/);
});
