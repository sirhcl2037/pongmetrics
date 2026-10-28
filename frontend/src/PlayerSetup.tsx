import type { Player } from './session';
type Props = { players: Player[]; selected: string; onSelect: (id: string) => void; onChange: (players: Player[]) => void; onAssociate: (id: string) => void; enabled: boolean; autoSwitch: boolean; onAutoSwitch: (enabled: boolean) => void };
export function PlayerSetup({ players, selected, onSelect, onChange, onAssociate, enabled, autoSwitch, onAutoSwitch }: Props) {
  return <div className="player-setup"><h3>Players</h3><p>Select the player taking the next shot.</p><label className="auto-switch"><input type="checkbox" checked={autoSwitch} onChange={event => onAutoSwitch(event.target.checked)} /><span className="auto-switch-copy"><strong>Auto switch</strong><small>Alternate selected player after each shot</small></span></label>{players.map((player, i) => <div className="player-row" key={player.id}>
    <button aria-pressed={selected === player.id} onClick={() => onSelect(player.id)}>P{i+1}{selected === player.id ? ' ✓' : ''}</button>
    <input aria-label={`Player ${i+1} name`} maxLength={80} value={player.name} onChange={e => onChange(players.map(p => p.id === player.id ? { ...p, name: e.target.value } : p))} onBlur={() => { if (!player.name.trim()) onChange(players.map(p => p.id === player.id ? { ...p, name: `Player ${i+1}` } : p)); }} />
    <button disabled={!enabled} onClick={() => onAssociate(player.id)}>{player.association ? 'Reposition' : 'Locate'}</button>
  </div>)}<p>Locate pauses the video; click the player. Markers describe that timestamp, not tracking.</p></div>;
}
