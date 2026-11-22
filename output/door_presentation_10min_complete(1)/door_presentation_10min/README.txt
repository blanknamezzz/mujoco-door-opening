PREPARING TO OPEN A DOOR — PRESENTATION PACKAGE

Main deliverable: presentation.pdf (13 slides, 16:9, all English).
Editable source: main.tex and assets/.
Talk: speaker_notes.txt, with a ten-minute schedule including videos.

PLAYBACK
Keep presentation.pdf beside the videos/ folder. Slides 3 and 8 have
relative links to media2.mp4 and media6.mp4. These are external
video links, not embedded PDF playback. PDF viewers may block Launch links.
Reliable fallback: open demo_player.html in a browser before the talk and
switch to it on each demo slide. Open the MP4 directly if necessary.
The selected clips total 18.58 seconds, already included in the talk schedule.
Five remaining original clips are retained as optional backup; do not play
all seven within the scheduled ten-minute version.

BUILD
Run from this directory with XeLaTeX and Arial available:
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
The result is main.pdf; presentation.pdf is the delivery copy.

EVIDENCE
Videos and figures were extracted unchanged from the user-supplied
18-slide classroom presentation dated 2026-11-11. This packaging task did
not rerun training or simulation. Results use the recorded experiment
reports cited in the slide footnotes. Standalone grasp and mobile studies
have different protocols. Correct decisions include correct refusals.
No claim of improved imagined-rollout performance is made.

REVISION: Training coupling is the central method. Slide 4 reproduces the six historical comparisons supplied by the user. Their controls and cohort boundaries are preserved. PRM video is backup.

TELEMETRY UPDATE: use demo_player.html for slides 3 and 8. Slide 8 now explicitly uses a NEW geometry-baseline inspection experiment. Original videos remain in videos/. All 15 annotated clips are in telemetry/index.html. Do not attribute the supplementary geometry decision to the learned policy.

FINAL APPLICATION SLIDE: Future directions for precise everyday manipulation. Revised 13-slide schedule totals 600 seconds including demos.
