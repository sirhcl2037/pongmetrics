import type { Shot } from './session.ts';
import type { ReviewedRally } from './rallies.ts';

export const unclearReasons = [
  { value: 'feet-hidden', label: 'Feet hidden or out of frame' },
  { value: 'contact-uncertain', label: 'Contact time or player uncertain' },
  { value: 'view-uncertain', label: 'Camera cut, blur, or unclear readiness' },
  { value: 'missing-contact', label: 'No reliable next opponent contact' },
] as const;
export type RecoveryOutcome = 'ready' | 'late' | 'unclear';
type Contact = Pick<Shot, 'id' | 'time' | 'playerId'>;
export type RecoveryJudgment = {
  source: 'manual';
  shot: Contact;
  opponent: Contact | null;
  outcome: RecoveryOutcome;
  reason: typeof unclearReasons[number]['value'] | null;
};
export type RecoveryWindow = { rally: ReviewedRally; shot: Shot; opponent: Shot | null; judgment: RecoveryJudgment | null };

const sameContact = (a: Contact | null, b: Contact | null) =>
  a === null || b === null ? a === b : a.id === b.id && a.time === b.time && a.playerId === b.playerId;

export function recoveryWindows(rallies: ReviewedRally[], shots: Shot[], playerId: string): RecoveryWindow[] {
  return rallies.flatMap(rally => {
    const ordered = shots.filter(s => s.time >= rally.start && s.time < rally.end).sort((a, b) => a.time - b.time);
    return ordered.flatMap((shot, index) => {
      if (shot.playerId !== playerId) return [];
      const next = ordered[index + 1];
      // Never skip an unassigned or repeated-player contact to fabricate a pair.
      // Duplicate timestamps at either endpoint also need a human correction.
      const opponent = next && next.playerId !== null && next.playerId !== playerId &&
        next.time > shot.time && ordered[index - 1]?.time !== shot.time && ordered[index + 2]?.time !== next.time ? next : null;
      const judgment = rally.recovery?.find(row => sameContact(row.shot, shot) && sameContact(row.opponent, opponent)) ?? null;
      return [{ rally, shot, opponent, judgment }];
    });
  });
}

export function recoverySummary(windows: RecoveryWindow[]) {
  const ready = windows.filter(w => w.judgment?.outcome === 'ready').length;
  const late = windows.filter(w => w.judgment?.outcome === 'late').length;
  const unclear = windows.filter(w => w.judgment?.outcome === 'unclear').length;
  return { total: windows.length, ready, late, unclear, usable: ready + late,
    unreviewed: windows.length - ready - late - unclear,
    missingContact: windows.filter(w => !w.opponent).length,
    lateRate: ready + late ? late / (ready + late) : null };
}

export function makeRecoveryJudgment(window: RecoveryWindow, outcome: RecoveryOutcome,
  reason: RecoveryJudgment['reason'] = null): RecoveryJudgment {
  if (!window.opponent && outcome !== 'unclear') throw new Error('Tag the next opponent contact before judging readiness.');
  const contact = ({ id, time, playerId }: Shot): Contact => ({ id, time, playerId });
  return { source: 'manual', shot: contact(window.shot), opponent: window.opponent ? contact(window.opponent) : null,
    outcome, reason: outcome === 'unclear' ? (window.opponent ? reason ?? 'view-uncertain' : 'missing-contact') : null };
}

export function parseRecovery(value: unknown, duration: number, playerIds: ReadonlySet<string>): RecoveryJudgment[] {
  if (!Array.isArray(value)) throw new Error('Invalid recovery review.');
  const ids = new Set<number>();
  const contact = (c: Contact) => c && Number.isSafeInteger(c.id) && c.id >= 0 &&
    Number.isFinite(c.time) && c.time >= 0 && c.time <= duration && typeof c.playerId === 'string' && playerIds.has(c.playerId);
  for (const row of value) {
    if (!row || row.source !== 'manual' || !contact(row.shot) || ids.has(row.shot.id) ||
      !['ready', 'late', 'unclear'].includes(row.outcome) ||
      (row.opponent !== null && (!contact(row.opponent) || row.opponent.time <= row.shot.time ||
        row.opponent.id === row.shot.id || row.opponent.playerId === row.shot.playerId)) ||
      (row.outcome === 'unclear' ? !unclearReasons.some(r => r.value === row.reason) : row.reason !== null) ||
      (row.opponent === null && (row.outcome !== 'unclear' || row.reason !== 'missing-contact'))) {
      throw new Error('Invalid recovery review.');
    }
    ids.add(row.shot.id);
  }
  // Keep snapshots even if tags changed: the UI excludes stale judgments until re-reviewed.
  return value;
}
