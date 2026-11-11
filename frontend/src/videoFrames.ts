// Playback time lies inside a frame's interval. Rounding a midpoint seek can
// select the next frame; epsilon only absorbs floating-point error at a boundary.
export function displayedFrameIndex(time:number,fps:number) {
  return Math.max(0,Math.floor(time*fps+1e-6));
}

export function formatVideoTime(time: number) {
  const hundredths = Math.round(time * 100);
  const minutes = Math.floor(hundredths / 6000);
  const seconds = (hundredths % 6000) / 100;
  return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(2).padStart(5, '0')}`;
}
