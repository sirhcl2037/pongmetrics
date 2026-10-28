
import { useEffect, useRef, useState, type ChangeEvent } from "react";

type Wing = "FH" | "BH";
type Stroke = "Loop" | "Drive" | "Push" | "Block" | "Other";

type Shot = {
  id: number;
  time: number;
  wing: Wing;
  stroke: Stroke;
};

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const nextId = useRef(0);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("");
  const [shots, setShots] = useState<Shot[]>([]);
  const [stroke, setStroke] = useState<Stroke>("Loop");

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  function uploadVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setVideoUrl(URL.createObjectURL(file));
    setVideoName(file.name);
    setShots([]);
    nextId.current = 0;
  }

  function addShot(wing: Wing) {
    if (!videoRef.current) return;

    const time = Number(videoRef.current.currentTime.toFixed(2));

    const shot: Shot = {
      id: nextId.current++,
      time,
      wing,
      stroke
    };

    setShots(previous =>
        [...previous, shot].sort((a, b) => a.time - b.time)
    );
  }

  function removeShot(id: number) {
    setShots(previous => previous.filter(shot => shot.id !== id));
  }

  function jumpTo(time: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      videoRef.current.pause();
    }
  }

  const forehands = shots.filter(s => s.wing === "FH").length;
  const backhands = shots.filter(s => s.wing === "BH").length;

  return (
      <main>
        <header>
          <h1>PongMetrics</h1>
          <p>Table Tennis Analytics Platform</p>
        </header>

        <section>
          <h2>Match Video Analyzer</h2>

          <input
              type="file"
              accept="video/*"
              onChange={uploadVideo}
          />

          {videoUrl && (
              <>
                <h3>{videoName}</h3>

                <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    width="100%"
                />

                <h2>Shot Annotation</h2>

                <label>
                  Stroke Type:{" "}
                  <select
                      value={stroke}
                      onChange={e => setStroke(e.target.value as Stroke)}
                  >
                    {(["Loop", "Drive", "Push", "Block", "Other"] as const)
                        .map(type => (
                            <option key={type}>{type}</option>
                        ))}
                  </select>
                </label>

                <div className="annotation-controls">
                  <button onClick={() => addShot("FH")}>
                    + Forehand
                  </button>
                  <button onClick={() => addShot("BH")}>
                    + Backhand
                  </button>
                </div>

                <h2>Player Statistics</h2>
                <p>Total shots: {shots.length}</p>
                <p>Forehands: {forehands}</p>
                <p>Backhands: {backhands}</p>
                <p>
                  Forehand usage:{" "}
                  {shots.length
                      ? (forehands / shots.length * 100).toFixed(1)
                      : "0.0"}%
                </p>

                <h2>Shot Timeline</h2>

                {shots.map(shot => (
                    <div className="shot-entry" key={shot.id}>
                      <button onClick={() => jumpTo(shot.time)}>
                        {shot.time.toFixed(2)}s
                      </button>
                      <span>{shot.wing} — {shot.stroke}</span>
                      <button onClick={() => removeShot(shot.id)}>
                        Remove
                      </button>
                    </div>
                ))}
              </>
          )}
        </section>
      </main>
  );
}

export default App;
