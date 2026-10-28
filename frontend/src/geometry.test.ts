import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { homography, transform, TABLE, validateCorners, imageRect, normalizePoint } from './geometry.ts';
import { parseSession, defaultPlayers } from './session.ts';
import { ballTrail, parseDetections } from './ballDetections.ts';
const corners = [{x:.3,y:.2},{x:.7,y:.2},{x:.9,y:.8},{x:.1,y:.8}];
const close = (a: number,b: number) => assert.ok(Math.abs(a-b)<1e-8, `${a} != ${b}`);
test('ball trail drops missing frames and uncertain detections without interpolation', () => {
  const observations = [0,.04,.12,.16].map(timestamp => ({timestamp,position:{value:{x:.5,y:.3},source:'automatic' as const,confidence:.9}}));
  assert.equal(ballTrail(observations,.08,25,.5).length,0);
  assert.equal(ballTrail(observations,.12,25,.5).length,1);
  assert.equal(ballTrail(observations,.16,25,.5).length,2);
  assert.equal(ballTrail(observations,.16,25,.95).length,0);
  const video={name:'clip.mp4',size:100,width:1280,height:720,duration:9};
  const payload={format:'pongmetrics-ball-detections',version:1,coordinateSystem:'normalized-video-image',video,fps:25,ballObservations:observations};
  assert.equal(parseDetections(JSON.stringify(payload),video).observations.length,4);
  assert.throws(()=>parseDetections(JSON.stringify(payload),{...video,size:101}));
  assert.throws(()=>parseDetections(JSON.stringify({...payload,ballObservations:[observations[0],observations[0]]}),video));
});
test('perspective corners map to table meters and back', () => {
  validateCorners(corners);
  const h = homography(corners,TABLE), inverse = homography(TABLE,corners);
  corners.forEach((p,i) => { const mapped = transform(h,p); close(mapped.x,TABLE[i].x); close(mapped.y,TABLE[i].y); });
  for (const p of [...TABLE,{x:.7625,y:1.37},{x:.15,y:2.1}]) { const roundtrip = transform(h,transform(inverse,p)); close(roundtrip.x,p.x); close(roundtrip.y,p.y); }
});
test('known affine center and letterbox normalization are resolution independent', () => {
  const h = homography([{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}],TABLE);
  const center = transform(h,{x:.5,y:.5}); close(center.x,.7625); close(center.y,1.37);
  for (const scale of [1,2]) { const r = imageRect(800*scale,800*scale,1280,720); assert.equal(normalizePoint({x:10,y:10},r),null); const p=normalizePoint({x:400*scale,y:400*scale},r)!; close(p.x,.5); close(p.y,.5); }
});
test('invalid, crossed and collinear corners are rejected', () => {
  assert.throws(() => validateCorners([corners[0],corners[2],corners[1],corners[3]]));
  assert.throws(() => validateCorners(Array(4).fill({x:.5,y:.5})));
  assert.throws(() => validateCorners([{x:NaN,y:0},...corners.slice(1)]));
});
test('session roundtrip preserves provenance and rejects malformed data', () => {
  const s = {version:2,video:{name:'test.mp4',size:100,width:1280,height:720,duration:9},fps:25,coordinateSystem:'normalized-video-image',calibration:{timestamp:0,corners:corners.map(value=>({value,source:'manual'}))},shots:[{id:0,time:.04,wing:'FH',stroke:'Loop',source:'manual'}],ballObservations:[{timestamp:.08,position:{value:{x:.5,y:.3},source:'automatic',confidence:.8}}]};
  const migrated = parseSession(JSON.stringify(s));
  assert.equal(migrated.shots[0].playerId,null);
  assert.deepEqual(migrated.players,defaultPlayers());
  const assigned = { ...migrated, shots: [{ ...migrated.shots[0], playerId:'player-2', playerIdentity:{value:'player-2',source:'automatic',confidence:.9} }] };
  assert.deepEqual(parseSession(JSON.stringify(assigned)),assigned);
  assert.throws(()=>parseSession(JSON.stringify({...assigned,selectedPlayerId:'missing'})));
  assert.throws(()=>parseSession(JSON.stringify({...assigned,shots:[{...assigned.shots[0],playerId:'missing'}]})));
  assert.throws(()=>parseSession(JSON.stringify({...s,fps:0})));
  assert.throws(()=>parseSession(JSON.stringify({...s,shots:[...s.shots,...s.shots]})));
});
