"""Offline white-ball mixture model; only heavy observations calibrate appearance."""
import cv2
import numpy as np

LUT = np.where(np.arange(256)/255 <= .04045, np.arange(256)/255/12.92,
               ((np.arange(256)/255+.055)/1.055)**2.4).astype(np.float32)


class BackgroundColorModel:
    def __init__(self, samples):
        self.background_sample_count = len(samples)
        samples = LUT[np.asarray(samples, dtype=np.uint8)]
        self.background = np.median(samples, axis=0)
        self.noise = np.median(np.abs(samples-self.background), axis=0)
        self.stable = self.noise.max(axis=2) < .04
        self.white = np.ones(3, dtype=np.float32)
        self.colors = []
        self.observations = []
        self._prepare_projection()

    def _prepare_projection(self):
        # Recompute only when heavy calibration changes white appearance.
        self.direction = self.white-self.background
        self.length = np.sum(self.direction*self.direction, axis=2)
        self.projection = self.direction/np.maximum(self.length[:,:,None],1e-6)

    def fit(self, pixels, background):
        direction = self.white-background
        length = np.sum(direction*direction, axis=-1)
        change = pixels-background
        alpha = np.sum(change*direction, axis=-1)/np.maximum(length, 1e-6)
        error = np.sqrt(np.mean((change-alpha[...,None]*direction)**2, axis=-1))
        return alpha, error, length

    def learn(self, frame, point, confidence, timestamp):
        # TrackNet's peaks are not probabilities; this checkpoint commonly peaks
        # near .7 even on clear balls. Require image evidence as well as the peak.
        if confidence < .65:
            return
        x, y = np.clip(np.round(point), [0,0], [frame.shape[1]-1,frame.shape[0]-1]).astype(int)
        ys, xs = slice(max(0,y-3),y+4), slice(max(0,x-3),x+4)
        patch = frame[ys,xs]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # Learn white appearance only from bright, neutral cores; blurred samples
        # must not gradually teach the model that background/player colors are white.
        core = (hsv[:,:,1] <= 80) & (hsv[:,:,2] >= 200)
        if core.sum() >= 2:
            color = np.median(LUT[patch][core], axis=0)
            self.colors.append(color)
            self.colors = self.colors[-8:]
            self.white = np.maximum(.8, np.median(self.colors, axis=0))
            self._prepare_projection()
        alpha, error, _ = self.fit(LUT[patch], self.background[ys,xs])
        evidence = (alpha > .05) & (alpha <= 1.2) & (error < .05) & self.stable[ys,xs]
        if not evidence.any():
            return
        self.observations.append({'timestamp':float(timestamp), 'confidence':float(confidence),
            'patchBgr':patch.tolist(), 'backgroundLinearBgr':self.background[ys,xs].tolist(),
            'mixturePixels':int(evidence.sum()),
            'mixtureBoundsPixels':np.ptp(np.argwhere(evidence),axis=0)[::-1].astype(int).tolist(),
            'medianMixtureFraction':float(np.median(alpha[evidence])),
            'medianColorChangeLinearBgr':np.median((LUT[patch]-self.background[ys,xs])[evidence],axis=0).tolist(),
            'source':'heavy-detection'})

    def candidates(self, frame, gray, previous, center, radius):
        if not self.observations:
            return []
        cx, cy = np.clip(center,[0,0],[frame.shape[1]-1,frame.shape[0]-1]).astype(int)
        x0, x1 = max(0,cx-radius), min(frame.shape[1],cx+radius+1)
        y0, y1 = max(0,cy-radius), min(frame.shape[0],cy+radius+1)
        region = np.s_[y0:y1,x0:x1]
        change = LUT[frame[region]]-self.background[region]
        alpha = np.sum(change*self.projection[region],axis=2)
        residual = change-alpha[:,:,None]*self.direction[region]
        error_squared = np.mean(residual*residual,axis=2)
        delta = gray[region].astype(np.int16)-previous[region].astype(np.int16)
        # Near-white background is uninformative, and moving-player regions have
        # an unreliable background. Fall back to heavy recovery there.
        # White shoes/skin highlights can fit the mixture too. Large surrounding
        # changes inconsistent with white indicate a foreground object occluding
        # the stored background, where that background cannot be trusted.
        foreground = (error_squared > .07**2) & (np.max(np.abs(change),axis=2) > .1)
        foreground = cv2.dilate(foreground.astype(np.uint8),np.ones((7,7),np.uint8)) != 0
        valid = (alpha > .05) & (alpha <= 1.2) & (error_squared < .05**2) & (self.length[region] > .08) & self.stable[region]
        count, labels, stats, centers = cv2.connectedComponentsWithStats(valid.astype(np.uint8))
        output = []
        for k in range(1,count):
            left, top, width, height, area = stats[k]
            if not 3 <= area <= 120 or max(width,height) > 32 or area/(width*height) < .2:
                continue
            sub = np.s_[top:top+height,left:left+width]
            pixels = labels[sub] == k
            if np.mean(foreground[sub][pixels]) > .25:
                continue
            changes = delta[sub][pixels]
            if changes.mean() <= 3 or np.mean(changes > 10) < .15:
                continue
            point = centers[k]+[x0,y0]
            proximity = np.exp(-np.linalg.norm(point-center)/max(radius*.5,1))
            fit = max(0, 1-float(np.sqrt(error_squared[sub][pixels]).mean())/.05)
            motion = min(1,float(np.maximum(changes,0).mean())/45)
            strength = min(1,float(alpha[sub][pixels].mean())/.25)
            score = float(.2*proximity+.4*fit+.2*motion+.2*strength)
            output.append((score,point))
        return sorted(output,key=lambda item:item[0],reverse=True)

    def summary(self):
        return {'sampleCount':len(self.observations), 'whiteLinearBgr':self.white.tolist(),
                'stableBackgroundFraction':float(self.stable.mean()),
                'heavyAppearanceSamples':self.observations, 'classification':'white',
                'sampledBackgroundFrames':self.background_sample_count}


def prepare_background(cap, start, end, detector, size):
    samples = []
    # Bounded sampling avoids retaining the whole decoded video in cheap mode.
    for timestamp in np.linspace(start, end, 12, endpoint=False):
        if hasattr(detector, 'frames'):
            i = min(len(detector.frames)-1, max(0,round(timestamp*detector.fps)))
            frame = detector.frames[i]
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp*1000))
            ok, frame = cap.read()
            if not ok:
                continue
        samples.append(cv2.resize(frame,size,interpolation=cv2.INTER_AREA))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if not samples:
        raise RuntimeError('No frames available for background preprocessing')
    return BackgroundColorModel(samples)
