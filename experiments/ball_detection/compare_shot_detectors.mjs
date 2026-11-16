// Re-run the browser's contact benchmark from a reviewed interval export.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { analyzeReviewedShots } from '../../frontend/src/shotAnalysis.ts';
import { clickedPlayerTrack, parsePlayerTracks, playerTrackNames } from '../../frontend/src/playerTracks.ts';
import * as current from '../../frontend/src/shotDetection.ts';

const [reviewPath, baselinePath] = process.argv.slice(2);
if (!reviewPath) throw Error('Usage: node compare_shot_detectors.mjs review.json [baseline-detector.ts]');
const review = JSON.parse(readFileSync(reviewPath, 'utf8').replace(/^\uFEFF/, ''));
if (review.format !== 'pongmetrics-shot-analysis-review' || review.version !== 1 || review.completeContactReview !== true) {
  throw Error('Export a complete contact interval from Shot-level analysis review.');
}
const { video, fps, players, shots, inputs, minimumBallScore, fixedCameraConfirmed } = review;
const tracks = inputs.playerTracks ? parsePlayerTracks(JSON.stringify(inputs.playerTracks), video) : null;
const names = tracks ? playerTrackNames(tracks, players) : new Map();
const playerIds = new Map();
if (tracks) {
  for (const player of players) {
    const association = player.association;
    if (!association) continue;
    const match = clickedPlayerTrack(tracks, association.timestamp, association.position.value);
    if (match && names.has(match.trackId)) playerIds.set(player.id, match.trackId);
  }
}
const allowed = names.size ? new Set(names.keys()) : undefined;
const detectors = [current];
if (baselinePath) detectors.unshift(await import(pathToFileURL(resolve(baselinePath)).href));
function run(detector) {
  const candidates = detector.detectShotCandidates(inputs.ballObservations, fps, minimumBallScore, tracks, allowed);
  const report = analyzeReviewedShots(shots, candidates, inputs.ballObservations, fps, minimumBallScore,
    tracks, playerIds, review.report.interval, fixedCameraConfirmed);
  return { detector: detector.shotDetectionMethod ?? baselinePath, ...report };
}
console.log(JSON.stringify(detectors.map(run), null, 2));
