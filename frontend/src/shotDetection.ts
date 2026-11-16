import type { BallObservation } from './session.ts';
import type { PlayerBox, PlayerTracks } from './playerTracks.ts';
import type { Point } from './geometry.ts';

export const shotDetectionMethod = 'player-supported-ball-turn-v3';

export type PlayerAxisEvidence = { from:{x:number;y:number}; to:{x:number;y:number};
  incomingProjection:number; outgoingProjection:number; alignment:number; reverses:boolean };

function bodyCenter(person:PlayerBox) {
  const hips = person.landmarks?.filter(j => j.name.endsWith('hip') && j.position && (j.position.confidence ?? 0) >= .35);
  if (hips?.length === 2) return {x:(hips[0].position!.value.x+hips[1].position!.value.x)/2,y:(hips[0].position!.value.y+hips[1].position!.value.y)/2};
  return {x:person.box.x+person.box.width/2,y:person.box.y+person.box.height/2};
}

// A current-frame, unambiguous pair only. No referee selection or stale boxes.
export function playerAxisEvidence(people:PlayerBox[], incoming:readonly number[], outgoing:readonly number[]):PlayerAxisEvidence|null {
  if (people.length !== 2) return null;
  const from=bodyCenter(people[0]), to=bodyCenter(people[1]);
  const dx=(to.x-from.x)*1280, dy=(to.y-from.y)*720, length=Math.hypot(dx,dy);
  if (length < 50) return null;
  const project=(v:readonly number[]) => (v[0]*dx+v[1]*dy)/length;
  const before=project(incoming), after=project(outgoing);
  const alignment=Math.min(1,Math.abs(before)/Math.max(1e-9,Math.hypot(...incoming)),Math.abs(after)/Math.max(1e-9,Math.hypot(...outgoing)));
  return {from,to,incomingProjection:before,outgoingProjection:after,alignment,
    reverses:before*after < 0 && Math.abs(before) >= 2 && Math.abs(after) >= 2};
}

export type ShotCandidate = { timestamp:number; position:{x:number;y:number}; turnDegrees:number;
  trackId:number|null; evidence:'wrist-proximity'|'player-proximity'|'ball-reversal'|'short-ball-gap';
  gap?:{start:number;end:number;fitErrorPixels:number};
  playerAxis:PlayerAxisEvidence|null; contactScore:number|null; needsReview:true };

function pixelDelta(from: Point, to: Point) {
  return [(to.x - from.x) * 1280, (to.y - from.y) * 720] as const;
}

function playerSupport(people: PlayerBox[], point: Point) {
  return people.flatMap(person => {
    const { x, y, width, height } = person.box;
    const boxHeight = height * 720;
    const boxDistance = Math.hypot(
      Math.max(x - point.x, 0, point.x - x - width) * 1280,
      Math.max(y - point.y, 0, point.y - y - height) * 720,
    );
    if (boxDistance > .24 * boxHeight) return [];
    const nearWrist = person.landmarks?.some(joint => joint.name.endsWith('wrist') &&
      joint.position && (joint.position.confidence ?? 0) >= .35 &&
      Math.hypot(...pixelDelta(point, joint.position.value)) <= .23 * boxHeight);
    return [{ id: person.trackId, boxHeight, nearWrist: !!nearWrist }];
  });
}

// Fit the two observed flight segments at the same time inside a short gap.
// The closest approach is a contact hypothesis, never a ball observation.
function estimateGap(first: BallObservation, before: BallObservation, after: BallObservation, last: BallObservation) {
  const duration = after.timestamp - before.timestamp;
  const start = before.position.value, end = after.position.value;
  const incoming = pixelDelta(first.position.value, start);
  const outgoing = pixelDelta(end, last.position.value);
  if (Math.hypot(...incoming) < 4 || Math.hypot(...outgoing) < 4) return null;
  const dtIn = before.timestamp - first.timestamp;
  const dtOut = last.timestamp - after.timestamp;
  const inX = incoming[0] / dtIn, inY = incoming[1] / dtIn;
  const outX = outgoing[0] / dtOut, outY = outgoing[1] / dtOut;
  const span = pixelDelta(start, end);
  const offsetX = span[0] - outX * duration;
  const offsetY = span[1] - outY * duration;
  const dx = inX - outX, dy = inY - outY;
  const denominator = dx * dx + dy * dy;
  if (!denominator) return null;
  const elapsed = Math.max(0, Math.min(duration, (offsetX * dx + offsetY * dy) / denominator));
  const fitErrorPixels = Math.hypot(offsetX - dx * elapsed, offsetY - dy * elapsed);
  const x = start.x + ((inX + outX) * elapsed + offsetX) / (2 * 1280);
  const y = start.y + ((inY + outY) * elapsed + offsetY) / (2 * 720);
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { timestamp: before.timestamp + elapsed, position: { x, y }, incoming, outgoing, fitErrorPixels };
}

// Image-space reversals suggest contacts, but can also be bounces or detector errors.
export function detectShotCandidates(observations:BallObservation[], fps:number, threshold:number, tracks:PlayerTracks|null = null, allowedTrackIds?:ReadonlySet<number>):ShotCandidate[] {
  const turns:ShotCandidate[] = [];
  const maxSampleStep = 2.1 / fps;
  function add(timestamp: number, position: Point, incoming: readonly [number, number], outgoing: readonly [number, number], gap?: ShotCandidate['gap']) {
    const speedA = Math.hypot(...incoming), speedB = Math.hypot(...outgoing);
    if (speedA < 4 || speedB < 4) return;
    const cosine = Math.max(-1, Math.min(1, (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (speedA * speedB)));
    if (cosine > .25) return;
    const people = (tracks?.frames.get(Math.round(timestamp * tracks.fps)) ?? [])
      .filter(person => !allowedTrackIds || allowedTrackIds.has(person.trackId));
    const nearby = playerSupport(people, position);
    const person = nearby.length === 1 ? nearby[0] : null;
    const axis = playerAxisEvidence(people, incoming, outgoing);
    // Two observed players provide a reach check. A missing player box
    // leaves an observed turn eligible, but its hitter may remain unknown.
    if (allowedTrackIds?.size === 2 && people.length === 2 && !nearby.length) return;
    if (gap && (!person || gap.fitErrorPixels > Math.min(45, .22 * person.boxHeight) || (axis && !axis.reverses))) return;
    const evidence = gap ? 'short-ball-gap' : person ? (person.nearWrist ? 'wrist-proximity' : 'player-proximity') : 'ball-reversal';
    const turnDegrees = Math.acos(cosine) * 180 / Math.PI;
    const contactScore = axis ? (axis.reverses ? .7 * axis.alignment : 0) + (person ? .2 : 0) +
      (person?.nearWrist ? .1 : 0) : null;
    turns.push({ timestamp, position, turnDegrees, trackId: person?.id ?? null,
      evidence, ...(gap ? { gap } : {}), playerAxis: axis, contactScore, needsReview: true });
  }
  for (let i = 2; i < observations.length - 2; i++) {
    const window = observations.slice(i - 2, i + 3);
    if (window.some((sample, j) => (sample.position.confidence ?? 1) < threshold ||
        (j > 0 && (sample.timestamp - window[j - 1].timestamp > maxSampleStep || sample.timestamp <= window[j - 1].timestamp)))) continue;
    const [start, , middle, , end] = window;
    const point = middle.position.value;
    add(middle.timestamp, point, pixelDelta(start.position.value, point), pixelDelta(point, end.position.value));
  }
  if (tracks && allowedTrackIds?.size === 2) {
    const eligible = observations.filter(sample => (sample.position.confidence ?? 1) >= threshold);
    for (let i = 1; i < eligible.length - 2; i++) {
      const before = eligible[i], after = eligible[i + 1];
      const duration = after.timestamp - before.timestamp;
      if (duration < 1.5 / fps || duration > .2 + 1e-9) continue;
      const first = eligible[i - 1], last = eligible[i + 2];
      const dtIn = before.timestamp - first.timestamp, dtOut = last.timestamp - after.timestamp;
      const maxLeg = Math.min(.12 + 1e-9, 3.1 / fps);
      if (dtIn <= 0 || dtOut <= 0 || dtIn > maxLeg || dtOut > maxLeg) continue;
      const fit = estimateGap(first, before, after, last);
      if (fit) add(fit.timestamp, fit.position, fit.incoming, fit.outgoing, {
        start: before.timestamp, end: after.timestamp, fitErrorPixels: fit.fitErrorPixels,
      });
    }
  }
  const candidates:ShotCandidate[] = [];
  const score = (candidate: ShotCandidate) => (candidate.contactScore ?? .4 * candidate.turnDegrees / 180) - (candidate.gap ? .15 : 0);
  for (const candidate of turns.sort((a, b) => a.timestamp - b.timestamp)) {
    const previous = candidates.at(-1);
    if (previous && candidate.timestamp-previous.timestamp < .16) {
      if (score(candidate) > score(previous)) candidates[candidates.length-1] = candidate;
    } else candidates.push(candidate);
  }
  return candidates;
}
