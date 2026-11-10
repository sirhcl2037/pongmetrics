// Playback time lies inside a frame's interval. Rounding a midpoint seek can
// select the next frame; epsilon only absorbs floating-point error at a boundary.
export function displayedFrameIndex(time:number,fps:number) {
  return Math.max(0,Math.floor(time*fps+1e-6));
}
