// Score actual racket contacts from frame review and inspect unmatched shots near OpenTT events.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { matchContacts } from '../../frontend/src/shotAnalysis.ts';
import { parsePlayerTracks } from '../../frontend/src/playerTracks.ts';
import * as current from '../../frontend/src/shotDetection.ts';

const [ballPath, trackPath, evaluationPath, reviewPath, baselinePath] = process.argv.slice(2);
if (!reviewPath) throw Error('Usage: node compare_opentt_contacts.mjs detections.json players.json evaluation.json contact-review.json [baseline-detector.ts]');
const read = path => JSON.parse(readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const { video, fps, ballObservations } = read(ballPath);
const tracks = parsePlayerTracks(readFileSync(trackPath, 'utf8'), video);
const { sourceRange, events } = read(evaluationPath);
const { interval, contacts, trackIds } = read(reviewPath);
if (interval[0] < sourceRange[0] || interval[1] > sourceRange[1] || interval[0] >= interval[1] ||
    tracks.fps !== fps || contacts.some(c => c.frame < interval[0] || c.frame >= interval[1] || !(c.side in trackIds))) {
  throw Error('Contact review does not match the scored segment');
}
const toTime = frame => (frame - sourceRange[0]) / fps;
const toFrame = time => Math.round(time * fps) + sourceRange[0];
const [start, end] = interval.map(toTime);
const refs = contacts.map((contact, id) => ({ id, time: toTime(contact.frame), playerId: contact.side }));
const ids = new Set(Object.values(trackIds));
if (ids.size !== 2 || [...ids].some(id => !Number.isSafeInteger(id))) {
  throw Error('Map both reviewed players to distinct track IDs');
}
const nonRacketEvents = events.filter(e => e.timestamp >= start && e.timestamp < end &&
  (e.label === 'bounce' || e.label === 'net'));
const detectors = [current];
if (baselinePath) detectors.unshift(await import(pathToFileURL(resolve(baselinePath)).href));

function run(detector) {
  const detect = () => detector.detectShotCandidates(ballObservations, fps, .15, tracks, ids);
  const candidates = detect();
  const matches = matchContacts(refs, candidates.filter(c => c.timestamp >= start - .12 && c.timestamp < end + .12));
  const matched = new Set(matches.map(m => m.candidate));
  const unmatched = candidates.filter(c => c.timestamp >= start && c.timestamp < end && !matched.has(c));
  const timings = [];
  for (let i = 0; i < 28; i++) {
    const began = performance.now();
    detect();
    if (i >= 8) timings.push(performance.now() - began);
  }
  timings.sort((a, b) => a - b);
  const offsets = matches.map(m => Math.abs(m.shot.time - m.candidate.timestamp));
  return {
    method: detector.shotDetectionMethod,
    reviewedContacts: refs.length,
    matchedContacts: matches.length,
    auditedEvents: Object.fromEntries(['bounce', 'net'].map(label =>
      [label, nonRacketEvents.filter(event => event.label === label).length])),
    falseSuggestions: unmatched.map(candidate => ({
      frame: toFrame(candidate.timestamp),
      trackId: candidate.trackId,
      nearbyEvents: nonRacketEvents.filter(e => Math.abs(e.timestamp - candidate.timestamp) <= .05)
        .map(e => ({ frame: e.sourceFrame, label: e.label })),
    })),
    missedFrames: refs.filter(ref => !matches.some(m => m.shot === ref)).map(ref => toFrame(ref.time)),
    meanAbsoluteTimingErrorMs: offsets.length ? 1000 * offsets.reduce((a, b) => a + b, 0) / offsets.length : null,
    matches: matches.map(({ shot, candidate }) => ({
      referenceFrame: toFrame(shot.time),
      suggestedFrame: toFrame(candidate.timestamp),
      hitter: candidate.trackId === null ? 'unknown' : candidate.trackId === trackIds[shot.playerId] ? 'correct' : 'incorrect',
      gap: candidate.gap ? [toFrame(candidate.gap.start), toFrame(candidate.gap.end)] : null,
    })),
    medianSuggestionMs: timings[Math.floor(timings.length / 2)],
  };
}

console.log(JSON.stringify(detectors.map(run), null, 2));
