import type { BallObservation } from './session.ts';
import type { PlayerBox, PlayerTracks } from './playerTracks.ts';

export type PlayerAxisEvidence = { from:{x:number;y:number}; to:{x:number;y:number};
  incomingProjection:number; outgoingProjection:number; alignment:number; reverses:boolean };

function bodyCenter(person:PlayerBox) {
  const hips = person.landmarks?.filter(j => j.name.endsWith('hip') && j.position && (j.position.confidence ?? 0) >= .35);
  if (hips?.length === 2) return {x:(hips[0].position!.value.x+hips[1].position!.value.x)/2,y:(hips[0].position!.value.y+hips[1].position!.value.y)/2};
  return {x:person.box.x+person.box.width/2,y:person.box.y+person.box.height/2};
}

// A current-frame, unambiguous pair only. No referee selection or stale boxes.
export function playerAxisEvidence(people:PlayerBox[], incoming:number[], outgoing:number[]):PlayerAxisEvidence|null {
  if (people.length !== 2) return null;
  const from=bodyCenter(people[0]), to=bodyCenter(people[1]);
  const dx=(to.x-from.x)*1280, dy=(to.y-from.y)*720, length=Math.hypot(dx,dy);
  if (length < 50) return null;
  const project=(v:number[]) => (v[0]*dx+v[1]*dy)/length;
  const before=project(incoming), after=project(outgoing);
  const alignment=Math.min(1,Math.abs(before)/Math.max(1e-9,Math.hypot(...incoming)),Math.abs(after)/Math.max(1e-9,Math.hypot(...outgoing)));
  return {from,to,incomingProjection:before,outgoingProjection:after,alignment,
    reverses:before*after < 0 && Math.abs(before) >= 2 && Math.abs(after) >= 2};
}

export type ShotCandidate = { timestamp:number; position:{x:number;y:number}; turnDegrees:number;
  trackId:number|null; evidence:'wrist-proximity'|'player-proximity'|'ball-reversal';
  playerAxis:PlayerAxisEvidence|null; contactScore:number|null; needsReview:true };

// Image-space reversals suggest contacts, but can also be bounces or detector errors.
// Require observed samples on both sides; never invent a contact across a long gap.
export function detectShotCandidates(observations:BallObservation[], fps:number, threshold:number, tracks:PlayerTracks|null = null, allowedTrackIds?:ReadonlySet<number>):ShotCandidate[] {
  const candidates:ShotCandidate[] = [];
  for (let i = 2; i < observations.length - 2; i++) {
    const window = observations.slice(i-2,i+3);
    if (window.some((o,j) => (o.position.confidence ?? 1) < threshold ||
        (j > 0 && (o.timestamp-window[j-1].timestamp > 2.1/fps || o.timestamp <= window[j-1].timestamp)))) continue;
    const [start,,middle,,end] = window;
    const a = start.position.value, p = middle.position.value, b = end.position.value;
    const incoming = [(p.x-a.x)*1280,(p.y-a.y)*720];
    const outgoing = [(b.x-p.x)*1280,(b.y-p.y)*720];
    const speedA = Math.hypot(...incoming), speedB = Math.hypot(...outgoing);
    // Broadcast views can show travel mostly horizontally or vertically.
    // Use the full 2D turn rather than hard-coding the screen's vertical axis.
    if (speedA < 4 || speedB < 4) continue;
    const cosine = Math.max(-1,Math.min(1,(incoming[0]*outgoing[0]+incoming[1]*outgoing[1])/(speedA*speedB)));
    if (cosine > .25) continue;
    const people = (tracks?.frames.get(Math.round(middle.timestamp*tracks.fps)) ?? []).filter(p => !allowedTrackIds || allowedTrackIds.has(p.trackId));
    const ranked = people.map(person => {
      const wristDistance = Math.min(...(person.landmarks ?? []).filter(j => j.name.endsWith('wrist') && j.position && (j.position.confidence ?? 0) >= .35)
        .map(j => Math.hypot((p.x-j.position!.value.x)*1280,(p.y-j.position!.value.y)*720)));
      const box = person.box;
      const boxDistance = Math.hypot(Math.max(box.x-p.x,0,p.x-box.x-box.width)*1280,Math.max(box.y-p.y,0,p.y-box.y-box.height)*720);
      return {id:person.trackId, distance:wristDistance <= 100 ? wristDistance : boxDistance,
        evidence:wristDistance <= 100 ? 'wrist-proximity' as const : 'player-proximity' as const, eligible:wristDistance <= 100 || boxDistance <= 35};
    }).filter(p => p.eligible).sort((a,b) => a.distance-b.distance);
    // Overlapping people must not silently force a player assignment.
    const person = ranked.length === 1 ? ranked[0] : null;
    const axis=playerAxisEvidence(people,incoming,outgoing);
    // Heuristic ranking, not probability. Parallel reversal near a person is
    // stronger contact evidence; cross-axis turns remain available for review.
    const contactScore=axis ? (axis.reverses ? .8*axis.alignment : 0)+(person ? .2 : 0) : null;
    const candidate:ShotCandidate = {timestamp:middle.timestamp,position:p,turnDegrees:Math.acos(cosine)*180/Math.PI,
      trackId:person?.id ?? null,evidence:person?.evidence ?? 'ball-reversal',playerAxis:axis,contactScore,needsReview:true};
    const previous = candidates.at(-1);
    if (previous && candidate.timestamp-previous.timestamp < .16) {
      const score=(c:ShotCandidate) => c.contactScore ?? .4*c.turnDegrees/180;
      if (score(candidate) > score(previous)) candidates[candidates.length-1] = candidate;
    } else candidates.push(candidate);
  }
  return candidates;
}
