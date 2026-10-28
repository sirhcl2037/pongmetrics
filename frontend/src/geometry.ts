export type Point = { x: number; y: number };
export type Matrix = number[];
export const TABLE = [{ x: 0, y: 0 }, { x: 1.525, y: 0 }, { x: 1.525, y: 2.74 }, { x: 0, y: 2.74 }];

export function transform(h: Matrix, p: Point): Point {
  const w = h[6] * p.x + h[7] * p.y + h[8];
  if (Math.abs(w) < 1e-10) throw new Error('Point is outside the finite projection.');
  return { x: (h[0] * p.x + h[1] * p.y + h[2]) / w, y: (h[3] * p.x + h[4] * p.y + h[5]) / w };
}

// Four correspondences give eight equations for eight unknowns (h33 = 1).
export function homography(from: Point[], to: Point[]): Matrix {
  if (from.length !== 4 || to.length !== 4) throw new Error('Four corners are required.');
  const rows = from.flatMap(({ x, y }, i) => {
    const { x: u, y: v } = to[i];
    return [[x, y, 1, 0, 0, 0, -u*x, -u*y, u], [0, 0, 0, x, y, 1, -v*x, -v*y, v]];
  });
  for (let c = 0; c < 8; c++) {
    let pivot = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(rows[r][c]) > Math.abs(rows[pivot][c])) pivot = r;
    [rows[c], rows[pivot]] = [rows[pivot], rows[c]];
    const divisor = rows[c][c];
    if (Math.abs(divisor) < 1e-10) throw new Error('Corners are degenerate; select four distinct table corners.');
    rows[c] = rows[c].map(n => n / divisor);
    for (let r = 0; r < 8; r++) if (r !== c) {
      const factor = rows[r][c];
      rows[r] = rows[r].map((n, j) => n - factor * rows[c][j]);
    }
  }
  return [...rows.map(row => row[8]), 1];
}

export function validateCorners(points: Point[]) {
  if (points.length !== 4 || points.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) throw new Error('Corners must be normalized image coordinates.');
  const crosses = points.map((a, i) => {
    const b = points[(i+1)%4], c = points[(i+2)%4];
    return (b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x);
  });
  if (!crosses.every(n => n > 1e-6) && !crosses.every(n => n < -1e-6)) throw new Error('Select corners around the table perimeter, without crossing edges.');
  homography(points, TABLE);
}

// object-fit: contain can introduce letterboxing inside the video element.
export function imageRect(width: number, height: number, videoWidth: number, videoHeight: number) {
  const scale = Math.min(width/videoWidth, height/videoHeight);
  const w = videoWidth*scale, h = videoHeight*scale;
  return { x: (width-w)/2, y: (height-h)/2, width: w, height: h };
}
export function normalizePoint(p: Point, rect: ReturnType<typeof imageRect>): Point | null {
  const x = (p.x-rect.x)/rect.width, y = (p.y-rect.y)/rect.height;
  return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? { x, y } : null;
}
