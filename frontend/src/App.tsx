
import { useEffect, useState } from "react";

function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("");

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  function handleVideoUpload(
      event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];
    if (!file) return;

    setVideoUrl(URL.createObjectURL(file));
    setVideoName(file.name);
  }

  return (
      <main>
        <header>
          <h1>PongMetrics</h1>
          <p>Table Tennis Analytics Platform</p>
        </header>

        <section>
          <h2>Match Video Analyzer</h2>
          <p>Upload a match recording to begin analysis.</p>

          <input
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
          />

          {videoUrl && (
              <div>
                <h3>{videoName}</h3>
                <video
                    src={videoUrl}
                    controls
                    width="100%"
                />
              </div>
          )}
        </section>
      </main>
  );
}

export default App;
