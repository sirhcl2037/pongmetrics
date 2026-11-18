import { useEffect, useRef, type KeyboardEvent } from 'react';
import type { ShotCandidate } from './shotDetection';
import { formatVideoTime } from './videoFrames';

const seekTolerance = .002;

function stepPosition(steps: ShotCandidate[], time: number) {
  let first = 0, end = steps.length;
  while (first < end) {
    const middle = Math.floor((first + end) / 2);
    if (steps[middle].timestamp <= time + seekTolerance) first = middle + 1;
    else end = middle;
  }
  const active = first - 1;
  const atContact = active >= 0 && time - steps[active].timestamp <= seekTolerance;
  return { active, previous: active - Number(atContact), next: first };
}

type Props = {
  steps: ShotCandidate[];
  time: number;
  readTime: () => number;
  onJump: (time: number) => void;
};

export function ShotStepper({ steps, time, readTime, onJump }: Props) {
  const { active, previous, next } = stepPosition(steps, time);
  const list = useRef<HTMLOListElement>(null);

  useEffect(() => {
    const strip = list.current;
    const item = strip?.children[active];
    if (!strip || !item) return;
    const reveal = () => {
      const viewport = strip.getBoundingClientRect();
      const bounds = item.getBoundingClientRect();
      if (bounds.left < viewport.left) strip.scrollLeft += bounds.left - viewport.left;
      else if (bounds.right > viewport.right) strip.scrollLeft += bounds.right - viewport.right;
    };
    reveal();
    const observer = new ResizeObserver(reveal);
    observer.observe(strip);
    return () => observer.disconnect();
  }, [active, steps]);

  function move(direction: -1 | 1) {
    const position = stepPosition(steps, readTime());
    const target = direction === 1 ? position.next : position.previous;
    if (steps[target]) onJump(steps[target].timestamp);
  }

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (!steps.length || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    move(event.key === 'ArrowLeft' ? -1 : 1);
  }

  return <section className="shot-stepper" aria-labelledby="shot-stepper-heading" tabIndex={steps.length ? 0 : undefined} onKeyDown={onKeyDown}>
    <h4 id="shot-stepper-heading">Shot steps</h4>
    {!steps.length ? <p>No contact estimates to step through.</p> : <>
      <div className="workspace-tools">
        <button disabled={previous < 0} onClick={() => move(-1)}>← Previous shot</button>
        <button disabled={next >= steps.length} onClick={() => move(1)}>Next shot →</button>
        <span>{active < 0 ? `${steps.length} estimated contacts` : `Step ${active + 1} of ${steps.length}`}</span>
      </div>
      <ol ref={list} className="shot-step-list" aria-label="Estimated contacts in order">
        {steps.map((step, index) => <li key={step.timestamp}>
          <button type="button" aria-current={index === active ? 'step' : undefined} onClick={() => onJump(step.timestamp)}>
            <strong>{index + 1}. {step.trackId === null ? 'Hitter unclear' : `Track ${step.trackId}`}</strong>
            <span>{formatVideoTime(step.timestamp)}{step.gap ? ' · Ball gap' : ''}</span>
          </button>
        </li>)}
      </ol>
      <p>Use ← and → while focused on the shot controls. Estimates can miss hits or include bounces. Ball-gap steps infer contact from the surrounding flight.</p>
    </>}
  </section>;
}
