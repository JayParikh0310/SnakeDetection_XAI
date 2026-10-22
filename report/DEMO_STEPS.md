# SnakeGuard: live demo steps (about 5 min)

Sample files: `results\demo_samples\`. Full comparison figures: `results\XAI_COMPARISON\`. Backup page: `SHOWCASE.html`.

## Before sir arrives (setup)
1. Close games, Chrome tabs with video, and anything else that uses the GPU.
2. Open a terminal in the project folder and run:
   ```
   .venv\Scripts\python scripts\app.py
   ```
3. Open **http://127.0.0.1:7860** in the browser.
4. **Warm-up:** upload any sample photo once and click Analyze. The first run takes about 4 s; after that it's fast. Then clear the image with the ✕ on it.
5. Keep a File Explorer window open at `results\demo_samples\` so you can drag images straight in.
6. Keep `SHOWCASE.html` open in another tab as a backup.

## Step 1: introduce the app (20 s)
Point at the title.
> "This is SnakeGuard. It takes a frame from the ankle camera, detects snakes, separates them from look-alikes like ropes and roots, checks whether the snake is likely venomous, gives a threat score, and **explains every decision with a heatmap**."

## Step 2: a venomous snake (1.5 min)
1. In the **Photo** tab, drag in **`snake_1.jpg`** (the green snake). Leave Audit mode **unticked** and click **Analyze**.
2. **Point at the annotated image:** "A red box means the detector found a snake. The label shows the detector confidence, 0.80, and the venom score."
3. **Point at the verdict and table on the right:** "It says *possibly venomous*, threat about 73 out of 100, and the beep interval. The threat combines three things: detector confidence, how close the snake is (estimated from box size), and venom status."
4. **Point at the detector heatmap ("why the detector fired"):** "This is LayerCAM, the method that won our comparison. Red is what the model relied on. It's **on the snake's body**, not the background, so the detection is for the right reason."
5. **Point at the venom heatmap:** "This is the venom classifier's explanation. It looks at the head and body pattern of the snake crop."

## Step 3: a look-alike, the key contrast (1 min)
1. Drag in **`lookalike_2.jpg`** (mangrove roots) and click **Analyze**.
2. Point at the **blue boxes:** "These roots look like snakes, but the detector labels them *snake-like object*: blue boxes, **no alert, threat 0**, so no beep. This is the main problem we set out to solve: false alarms from ropes and roots."
3. Optionally also show **`lookalike_3.jpg`** (a rope underwater): the same result, no alert.

## Step 4: audit mode, checking the explanation with a second method (1 min)
1. Go back to **`snake_1.jpg`**, **tick Audit mode**, and click **Analyze**. This takes about 5 s, so say this while it runs: "LayerCAM is fast, about 27 ms, so it runs live. Audit mode adds occlusion and RISE. They're much slower but don't depend on gradients: they hide parts of the image and measure how the score drops."
2. When it finishes: "Both kinds of method point at the same region. When two completely different method families agree, we can trust the explanation more. In our benchmark, occlusion was the most faithful method and LayerCAM came very close, at 45 times the speed."
3. If the details say something like "X % of the explanation lies inside the box", point at it: "It also reports how much of the explanation falls inside the box. If that number is low, the app warns that the detection may be reacting to the background."

## Step 5: a harmless snake, showing the safety trade-off (30 s)
Drag in **`snake_3.jpg`** and click **Analyze**.
> "Here it's a snake, but the venom model says *likely harmless*. The threat is lower (about 53) but **still not zero**, because it's still a snake and it tells you to keep your distance. The classifier is tuned to catch 95 % of venomous snakes, so it's deliberately cautious."

## Step 6: video and temporal confirmation (1 min)
1. Switch to the **Video** tab, upload **`demo_clip.mp4`**, and click the run button. It takes a few seconds.
2. While it processes: "On a real ankle camera we get a stream, not single photos. To avoid beeping on one bad frame, an alert needs a snake in **3 of the last 5 frames**."
3. When it finishes, play the output video and point at the **timeline table:** "The clip alternates roots and snakes. During the roots: no alert, threat 0. When the snake appears, the first frames aren't confirmed yet, and then the alert turns on with the threat and beep rate. The summary at the top says how many frames had a confirmed alert."

## Step 6b: LIVE camera, the real-time demo (1.5 min)
Open a snake photo on your **phone** (e.g. `snake_1.jpg`, or any snake image from Google) before you start.
1. Switch to the **Live** tab. Click the camera box → **allow camera access** → click **Record**. Click anywhere on the page once so the browser allows sound.
2. Point the laptop camera at a **rope, cable or charger wire** first: "The live feed runs the full pipeline on every frame. A cable gives no alert and no beep."
3. Hold the phone with the snake photo **far from the camera**: "The box appears, the dots at the bottom right fill up, and on the 3rd of 5 frames the alert confirms and it starts beeping slowly."
4. **Move the phone closer**: "The box grows, so proximity goes up, the threat score rises and the beeps get faster and faster until they merge into one **continuous tone**: that means *very close and dangerous*. A venomous call also uses a higher pitch."
5. **Move it out of frame**: "The snake is gone, so after a couple of frames it goes silent again."
6. Optionally tick **Show detector heatmap**: "This is LayerCAM live on every frame. It shows what the detector is looking at in real time, with the share of the heatmap inside the box below." (It runs a bit slower, about 3–4 fps.)
7. Click **Stop** on the camera when done.

> If sir asks why the alert takes a moment: "That's the 3-of-5 confirmation. It stops one bad frame from triggering a false alarm."

## Step 7: wrap up (20 s)
Point at the disclaimer at the bottom.
> "It's a research prototype, not medical advice. If venom status is uncertain, the app treats the snake as venomous. Now I'll show how we chose LayerCAM, the quantitative comparison."

Then switch to the `SHOWCASE.html` tab, open the **XAI comparison** section, and continue with the table and charts.

## If sir asks during the demo
| Question | Answer |
|---|---|
| "Why LayerCAM and not Grad-CAM++ from your proposal?" | We tested both. Grad-CAM++ scored 0.152 on deletion and LayerCAM 0.082 (lower is better), and LayerCAM is just as fast. Multi-scale was still essential: P5-only Grad-CAM++ was empty for 53 % of detections. |
| "How do you know the heatmap is correct?" | Deletion/insertion tests against a random baseline, a sanity check that randomizes the weights, and agreement with occlusion. All of it is in the comparison table. |
| "Try your own image?" | Any photo works, even a webcam shot of a rope or cable. Upload it and the heatmap shows what it looked at. Avoid tiny or far-away snakes; small objects are where the detector misses. |
| "Why did the harmless snake still get a threat score?" | Safety first: any snake is a potential threat. Venom status only raises or lowers the score. |

## If something goes wrong
- **App freezes or there's a CUDA error:** close it with Ctrl+C, run the command again (it takes about 10 s), and re-upload.
- **It won't start at all:** open `SHOWCASE.html` → the **App output** section, which has screenshots of the same output, and explain from there.
- **A sample gives a different result than expected:** say "it's a real model, not hand-picked output", then move on to the next sample.
