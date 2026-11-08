import { test } from 'node:test';
import assert from 'node:assert/strict';
import { homography, transform, TABLE, validateCorners, imageRect, normalizePoint } from './geometry.ts';
import { parseSession, defaultPlayers } from './session.ts';
import { ballTrail, parseDetections, parseTrackingFrames } from './ballDetections.ts';
import { playerTrackNames, type PlayerTracks } from './playerTracks.ts';
import { parseReviewDraft } from './reviewDraft.ts';

test('review drafts preserve human decisions and proposals and reject a different clip', () => {
  const video = {name:'review.mp4',size:100,width:1280,height:720,duration:9};
  const label = {frameIndex:0,timestamp:0,visibility:'visible',position:{value:{x:.4,y:.5},source:'manual'},
    review:{action:'accepted',proposal:{x:.4,y:.5},confidence:.8}};
  const draft = {version:1,video,fps:25,ballFps:25,labels:[label],time:0,
    observations:[{timestamp:0,position:{value:{x:.4,y:.5},source:'automatic',confidence:.8}}]};
  assert.deepEqual(parseReviewDraft(JSON.stringify(draft),video).labels,[label]);
  assert.throws(() => parseReviewDraft(JSON.stringify(draft),{...video,size:101}));
  assert.throws(() => parseReviewDraft(JSON.stringify({...draft,labels:[label,label]}),video));
  assert.throws(() => parseReviewDraft(JSON.stringify({...draft,labels:[{...label,visibility:'occluded'}]}),video));
});

test('saved player clicks bind names to track IDs and reject ambiguous assignments', () => {
  const tracks: PlayerTracks = { fps:25, firstTimestamp:0, frames:new Map([[0, [
    {trackId:7, playerId:'near-player', box:{x:0,y:.5,width:.4,height:.5}},
    {trackId:9, playerId:'far-player', box:{x:.5,y:0,width:.4,height:.4}},
  ]]]) };
  const players = defaultPlayers();
  players[0].name = 'Alex';
  players[0].association = {timestamp:0,position:{value:{x:.6,y:.2},source:'manual'}};
  assert.equal(playerTrackNames(tracks,players).get(9),'Alex');
  assert.equal(playerTrackNames(tracks,players).has(7),false);
  players[0].name = 'Sam';
  assert.equal(playerTrackNames(tracks,players).get(9),'Sam');
  players[1].association = players[0].association;
  assert.equal(playerTrackNames(tracks,players).size,0);
  players[1].association = {timestamp:0,position:{value:{x:1,y:1},source:'manual'}};
  assert.equal(playerTrackNames(tracks,players).get(9),'Sam');
  assert.equal(playerTrackNames({...tracks,frames:new Map()},players).size,0);
});
const corners = [{x:.3,y:.2},{x:.7,y:.2},{x:.9,y:.8},{x:.1,y:.8}];
const close = (a: number,b: number) => assert.ok(Math.abs(a-b)<1e-8, `${a} != ${b}`);

test('tracking diagnostics never become observations and require ordered finite values', () => {
  const trackingFrames = [{ timestamp: .04, prediction: { x: 1.2, y: -.1 }, uncertaintyRadiusPixels: 40, reason: 'occluded' }];
  assert.deepEqual(parseTrackingFrames(JSON.stringify({trackingFrames})), trackingFrames);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[null]})), /Invalid tracking diagnostics/);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:['bad']})), /Invalid tracking diagnostics/);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[{timestamp:0,prediction:null,uncertaintyRadiusPixels:null}]})), /Invalid tracking diagnostics/);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[...trackingFrames,...trackingFrames]})));
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[{...trackingFrames[0],uncertaintyRadiusPixels:-1}]})));
  const observations = Array.from({length:10000}, (_,i) => ({timestamp:i/25, position:{value:{x:.5,y:.3},source:'automatic' as const,confidence:.3}}));
  assert.equal(ballTrail(observations,399.96,25,.15).length,6);
  assert.equal(ballTrail(observations,399.96,25,.5).length,0);
});
test('ball trail drops missing frames and uncertain detections without interpolation', () => {
  const observations = [0,.04,.12,.16].map(timestamp => ({timestamp,position:{value:{x:.5,y:.3},source:'automatic' as const,confidence:.9}}));
  assert.equal(ballTrail(observations,.08,25,.5).length,0);
  assert.equal(ballTrail(observations,.12,25,.5).length,1);
  assert.equal(ballTrail(observations,.16,25,.5).length,2);
  assert.equal(ballTrail(observations,.16,25,.95).length,0);
  const video={name:'clip.mp4',size:100,width:1280,height:720,duration:9};
  const payload={format:'pongmetrics-ball-detections',version:1,coordinateSystem:'normalized-video-image',video,fps:25,ballObservations:observations};
  assert.equal(parseDetections(JSON.stringify(payload),video).observations.length,4);
  assert.equal(parseDetections(JSON.stringify({...payload,video:{...video,duration:9.04}}),video).observations.length,4);
  assert.throws(()=>parseDetections(JSON.stringify({...payload,video:{...video,duration:10}}),video));
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
