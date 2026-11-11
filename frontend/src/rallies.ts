import { parseRecovery, type RecoveryJudgment } from './recovery.ts';

export const rallyFocusOptions = [
  { value: 'serve-receive', label: 'Serve & receive' },
  { value: 'footwork-recovery', label: 'Footwork & recovery' },
  { value: 'stroke-mechanics', label: 'Stroke mechanics' },
  { value: 'placement-decisions', label: 'Placement & decisions' },
] as const;

export const rallyNoteLimit = 1000;

export type RallyRange = { start: number; end: number };
export type ReviewedRally = RallyRange & {
  id: string;
  winnerId: string | null;
  focus: typeof rallyFocusOptions[number]['value'] | null;
  note: string;
  recovery?: RecoveryJudgment[];
};

function validRange({ start, end }: RallyRange, duration: number) {
  return Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start && end <= duration;
}

export function rallyRangeError(range: RallyRange, duration: number, rallies: ReviewedRally[], editingId?: string) {
  if (!validRange(range, duration)) return 'Choose a start before the end, within the video.';
  if (rallies.some(rally => rally.id !== editingId && range.start < rally.end && rally.start < range.end)) {
    return 'This range overlaps a saved rally.';
  }
  return '';
}

export function parseRallies(value: unknown, duration: number, playerIds: ReadonlySet<string>): ReviewedRally[] {
  if (!Array.isArray(value)) throw new Error('Invalid rally review.');
  const ids = new Set<string>();
  for (const rally of value) {
    if (!rally || typeof rally.id !== 'string' || !rally.id || ids.has(rally.id) ||
        !validRange(rally, duration) || (rally.winnerId !== null && !playerIds.has(rally.winnerId)) ||
        (rally.focus !== null && !rallyFocusOptions.some(option => option.value === rally.focus)) ||
        typeof rally.note !== 'string' || rally.note.length > rallyNoteLimit) {
      throw new Error('Invalid rally review.');
    }
    ids.add(rally.id);
    if (rally.recovery !== undefined) rally.recovery = parseRecovery(rally.recovery, duration, playerIds);
  }
  const sorted = (value as ReviewedRally[]).slice().sort((a, b) => a.start - b.start);
  if (sorted.some((rally, index) => index > 0 && sorted[index - 1].end > rally.start)) {
    throw new Error('Saved rallies overlap.');
  }
  return sorted;
}
