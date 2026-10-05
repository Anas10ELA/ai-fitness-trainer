"""
scripts/label_tool.py
═════════════════════
أداة الـ Labeling التفاعلية

بتفتح كل ملف processed NPZ وبتسمحلك تعمل label لكل frame:
  • rep_state  : 0=neutral  1=down  2=up  3=plank  4=jump  5=extended  6=flexed
  • form_score : 0.0 → 1.0  (1.0 = perfect form)

الـ Labels بتتحفظ في data/labeled/{exercise}/{video_id}.npz

Usage:
    python scripts/label_tool.py --processed-dir data/processed --labeled-dir data/labeled
    python scripts/label_tool.py --processed-dir data/processed --exercise squat

Controls (OpenCV window):
    r = state NEUTRAL   (resting / start position)
    d = state DOWN      (squatting / bottom of pushup / hip hinge)
    u = state UP        (standing / top / lockout)
    p = state PLANK     (horizontal hold — plank, top of pushup)
    j = state JUMP      (airborne / explosive phase)
    e = state EXTENDED  (full extension — overhead press lockout, leg extension)
    f = state FLEXED    (peak contraction — bicep curl top, leg curl top)
    1-9 = form score    (1=0.1 bad … 9=0.9 good, 0=1.0 perfect)
    n = next frame      (carry same label forward)
    b = go back one frame
    s = save and next video
    q = quit            (saves current progress)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from exercises import Exercise, REGISTRY  # noqa: F401 — keep for project imports

# ── Skeleton connectivity (COCO 17-keypoint) ──────────────────────────────────

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head / ears / eyes
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso sides
    (11, 12),                                  # hip bar
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# ── 7-State definitions ───────────────────────────────────────────────────────

STATE_TO_IDX: Dict[str, int] = {
    "neutral":  0,
    "down":     1,
    "up":       2,
    "plank":    3,
    "jump":     4,
    "extended": 5,
    "flexed":   6,
}
IDX_TO_STATE: Dict[int, str] = {v: k for k, v in STATE_TO_IDX.items()}

# BGR colours — visually distinct across 7 states
STATE_COLOURS: Dict[int, tuple] = {
    0: (180, 180, 180),   # neutral   — light grey
    1: (100, 100, 255),   # down      — red-ish (BGR: low G/B, high R → appears red)
    2: (100, 255, 100),   # up        — green
    3: (255, 200,  50),   # plank     — cyan-gold
    4: (255,  80, 200),   # jump      — magenta
    5: ( 50, 200, 255),   # extended  — amber-yellow
    6: (200,  80,  50),   # flexed    — teal-blue
}

# Key bindings: single char → state index
KEY_TO_STATE: Dict[str, int] = {
    'r': 0,   # neutral
    'd': 1,   # down
    'u': 2,   # up
    'p': 3,   # plank
    'j': 4,   # jump
    'e': 5,   # extended
    'f': 6,   # flexed
}


# ═══════════════════════════════════════════════════════════════════════════════
#  LabelSession
# ═══════════════════════════════════════════════════════════════════════════════

class LabelSession:
    """Interactive labeling session for one processed video."""

    # Layout constants
    _W, _H        = 760, 650
    _SKEL_W       = 460   # left panel width reserved for skeleton
    _INFO_X       = 470   # x-origin for info text
    _PROGRESS_Y   = 215

    def __init__(
        self,
        keypoints:       np.ndarray,          # (N, 17, 2)
        scores:          np.ndarray,          # (N, 17)
        exercise:        str,
        video_id:        str,
        existing_labels: Optional[Dict] = None,
    ) -> None:
        self.kp        = keypoints
        self.sc        = scores
        self.exercise  = exercise
        self.video_id  = video_id
        self.n_frames  = len(keypoints)

        if existing_labels:
            self.rep_states  = existing_labels["rep_states"].tolist()
            self.form_scores = existing_labels["form_scores"].tolist()
        else:
            self.rep_states  = [0]   * self.n_frames   # default: neutral
            self.form_scores = [1.0] * self.n_frames   # default: perfect form

        self.frame_idx = 0
        self.modified  = False

        # ── FIX-1: track the last explicitly assigned state so 'n' carries
        #           the right value even after a state-key auto-advances the frame
        self._last_set_state:  int   = 0    # default neutral
        self._last_set_form:   float = 1.0

        # ── FIX-4: undo stack — list of (frame_idx, old_state, old_form)
        self._undo_stack: list = []
        self._MAX_UNDO        = 200

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(self) -> Optional[Dict]:
        """Run the interactive labeling loop.

        Returns the labels dict when the user presses **s** (save),
        or ``None`` if the user presses **q** (quit entire tool).
        """
        win_title = f"Label Tool — {self.exercise} / {self.video_id}"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, self._W, self._H)

        print(
            f"\n[Labeling] {self.exercise} / {self.video_id}  ({self.n_frames} frames)\n"
            f"  r=NEUTRAL  d=DOWN  u=UP  p=PLANK  j=JUMP  e=EXTENDED  f=FLEXED\n"
            f"  0-9=form_score   m=copy_fwd   n=next_safe   b=back   z=undo   s=save   q=quit"
        )

        while True:
            cv2.imshow(win_title, self._render_frame())
            key = cv2.waitKey(0) & 0xFF

            # ── Navigation / control ────────────────────────────────────────
            if key == ord('q'):
                cv2.destroyWindow(win_title)
                return None

            if key == ord('s'):
                cv2.destroyWindow(win_title)
                return self._pack_labels()

            # ── 'n' = safe next (move forward only, no copy) ─────────────────
            if key == ord('n'):
                self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                continue

            # ── 'm' = copy & move ────────────────────────────────────────────
            # Copies _last_set_state / _last_set_form into the CURRENT frame
            # independently (state and form are evaluated separately, not as a
            # combined default guard), then advances one frame.
            if key == ord('m'):
                self._push_undo(self.frame_idx)
                if self.rep_states[self.frame_idx] == 0:
                    self.rep_states[self.frame_idx]  = self._last_set_state
                if self.form_scores[self.frame_idx] == 1.0:
                    self.form_scores[self.frame_idx] = self._last_set_form
                self.modified  = True
                self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                continue

            if key == ord('b'):
                self.frame_idx = max(0, self.frame_idx - 1)
                continue

            # ── FIX-4: 'z' = undo last change ───────────────────────────────
            if key == ord('z'):
                self._pop_undo()
                continue

            # ── State keybindings ────────────────────────────────────────────
            char = chr(key) if key < 128 else ''
            if char in KEY_TO_STATE:
                self._push_undo(self.frame_idx)
                new_state = KEY_TO_STATE[char]
                self.rep_states[self.frame_idx] = new_state
                # FIX-1: remember this as the state to carry with 'n'
                self._last_set_state = new_state
                self._last_set_form  = self.form_scores[self.frame_idx]
                self.modified        = True
                self.frame_idx       = min(self.frame_idx + 1, self.n_frames - 1)
                continue

            # ── Form score (digit keys) ──────────────────────────────────────
            # Intentionally does NOT advance frame_idx — the user stays on the
            # same frame so the score bar visually updates before they move on.
            if char.isdigit():
                self._push_undo(self.frame_idx)
                digit    = int(char)
                new_form = 1.0 if digit == 0 else digit / 10.0
                self.form_scores[self.frame_idx] = new_form
                self._last_set_form = new_form
                self.modified       = True
                continue

        # unreachable, but satisfies type-checkers
        cv2.destroyWindow(win_title)
        return self._pack_labels()

    # ── Undo helpers ──────────────────────────────────────────────────────────

    def _push_undo(self, frame_idx: int) -> None:
        """Save a snapshot of one frame before modifying it."""
        entry = (frame_idx,
                 self.rep_states[frame_idx],
                 self.form_scores[frame_idx])
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)   # drop oldest

    def _pop_undo(self) -> None:
        """Restore the last saved snapshot."""
        if not self._undo_stack:
            return
        frame_idx, old_state, old_form = self._undo_stack.pop()
        self.rep_states[frame_idx]  = old_state
        self.form_scores[frame_idx] = old_form
        self.frame_idx = frame_idx   # jump back to that frame
        self.modified  = True

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_frame(self) -> np.ndarray:
        """Compose a 760×650 canvas: skeleton on left, info panel on right."""
        canvas = np.zeros((self._H, self._W, 3), dtype=np.uint8)
        # subtle panel divider
        cv2.rectangle(canvas, (self._SKEL_W, 0), (self._SKEL_W + 2, self._H), (50, 50, 50), -1)

        kp     = self.kp[self.frame_idx]          # (17, 2)
        sc_row = self.sc[self.frame_idx]           # (17,)
        state  = self.rep_states[self.frame_idx]
        fscore = self.form_scores[self.frame_idx]
        col    = STATE_COLOURS[state]

        # ── Skeleton ─────────────────────────────────────────────────────────
        for a, b in SKELETON:
            pa = (int(kp[a, 0]), int(kp[a, 1]))
            pb = (int(kp[b, 0]), int(kp[b, 1]))
            if pa == (0, 0) or pb == (0, 0):
                continue
            cv2.line(canvas, pa, pb, col, 2, cv2.LINE_AA)

        for i, p in enumerate(kp):
            x, y = int(p[0]), int(p[1])
            if x == 0 and y == 0:
                continue
            dot_col = (0, 255, 255) if sc_row[i] > 0.3 else (70, 70, 70)
            cv2.circle(canvas, (x, y), 4, dot_col, -1, cv2.LINE_AA)

        # ── Info panel ────────────────────────────────────────────────────────
        ix = self._INFO_X

        def put(text: str, y: int, color: tuple = (210, 210, 210), scale: float = 0.52):
            cv2.putText(canvas, text, (ix, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        put(f"Exercise", 30, (0, 190, 255), 0.46)
        put(self.exercise.upper(), 52, (0, 230, 255), 0.65)

        put(f"Video", 85, (160, 160, 160), 0.44)
        put(self.video_id, 105, (210, 210, 210), 0.50)

        put(f"Frame  {self.frame_idx + 1} / {self.n_frames}", 140)
        put(f"Prog   {(self.frame_idx + 1) / self.n_frames * 100:.1f}%", 162)

        # State badge
        cv2.rectangle(canvas, (ix - 2, 178), (self._W - 8, 202), (30, 30, 30), -1)
        put(f"State  {IDX_TO_STATE[state].upper()}", 197, col, 0.62)

        # Form score bar
        put(f"Form   {fscore:.1f}", 228, (255, 190, 80), 0.60)
        bar_max = self._W - ix - 8
        bar_fill = int(fscore * bar_max)
        cv2.rectangle(canvas, (ix, 235), (ix + bar_max, 245), (40, 40, 40), -1)
        bar_col = (80, 220, 80) if fscore >= 0.7 else (80, 150, 255) if fscore >= 0.4 else (80, 80, 220)
        cv2.rectangle(canvas, (ix, 235), (ix + bar_fill, 245), bar_col, -1)

        # State legend
        put("─── States ───", 275, (100, 100, 100), 0.40)
        legend_items = [
            ("r", "NEUTRAL"),
            ("d", "DOWN"),
            ("u", "UP"),
            ("p", "PLANK"),
            ("j", "JUMP"),
            ("e", "EXTENDED"),
            ("f", "FLEXED"),
        ]
        for row_i, (k, label) in enumerate(legend_items):
            y_pos    = 295 + row_i * 22
            s_idx    = KEY_TO_STATE[k]
            is_cur   = (s_idx == state)
            txt_col  = STATE_COLOURS[s_idx] if is_cur else (100, 100, 100)
            prefix   = "►" if is_cur else " "
            put(f"{prefix} [{k}] {label}", y_pos, txt_col, 0.44)

        # Bottom hint bar
        hints = ["0-9=form", "m=copy_fwd", "n=next_safe", "b=back", "z=undo", "s=save", "q=quit"]
        hint_x = ix
        for h in hints:
            cv2.putText(canvas, h, (hint_x, self._H - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
            hint_x += len(h) * 7 + 8

        # Undo depth indicator (top-right of info panel)
        if self._undo_stack:
            undo_txt = f"undo:{len(self._undo_stack)}"
            cv2.putText(canvas, undo_txt, (self._W - 75, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 80, 60), 1, cv2.LINE_AA)

        # 'n will carry' indicator — shows what the next 'n' press will assign
        carry_name  = IDX_TO_STATE[self._last_set_state].upper()
        carry_col   = STATE_COLOURS[self._last_set_state]
        put(f"'m' carries: {carry_name}", 560, carry_col, 0.42)

        # ── Frame progress bar (full width, bottom) ───────────────────────────
        bar_w = int((self.frame_idx + 1) / self.n_frames * (self._W - 20))
        cv2.rectangle(canvas, (10, self._H - 8), (self._W - 10, self._H - 2), (35, 35, 35), -1)
        cv2.rectangle(canvas, (10, self._H - 8), (10 + bar_w,  self._H - 2), (0, 190, 255), -1)

        return canvas

    # ── Serialization ─────────────────────────────────────────────────────────

    def _pack_labels(self) -> Dict:
        return {
            "rep_states":  np.array(self.rep_states,  dtype=np.int32),
            "form_scores": np.array(self.form_scores, dtype=np.float32),
            "exercise":    self.exercise,
            "video_id":    self.video_id,
            "n_frames":    self.n_frames,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive keypoint labeling tool (7 states)")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--labeled-dir",   default="data/labeled")
    parser.add_argument("--exercise",      default=None, help="Only label this exercise")
    parser.add_argument("--overwrite",     action="store_true",
                        help="Re-label already labeled videos")
    args = parser.parse_args()

    proc_dir    = Path(args.processed_dir)
    labeled_dir = Path(args.labeled_dir)

    # Collect processed NPZ files
    npz_files = []
    for ex_dir in sorted(proc_dir.iterdir()):
        if not ex_dir.is_dir():
            continue
        if args.exercise and ex_dir.name != args.exercise:
            continue
        for npz in sorted(ex_dir.glob("*.npz")):
            npz_files.append((ex_dir.name, npz))

    if not npz_files:
        print(f"[ERROR] No .npz files found in {proc_dir}")
        sys.exit(1)

    print(f"\n[Label Tool] {len(npz_files)} video(s) to process  |  7 states active")

    labeled = skipped = 0

    for ex_name, npz_path in npz_files:
        video_id = npz_path.stem
        out_dir  = labeled_dir / ex_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{video_id}.npz"

        if out_file.exists() and not args.overwrite:
            print(f"  [SKIP] {ex_name}/{video_id}  (already labeled)")
            skipped += 1
            continue

        data = np.load(npz_path, allow_pickle=True)
        kp   = data["keypoints"]
        sc   = data["scores"]

        existing = None
        if out_file.exists():
            existing = dict(np.load(out_file, allow_pickle=True))

        session = LabelSession(kp, sc, ex_name, video_id, existing)
        result  = session.run()

        if result is None:
            print("[INFO] Quit — progress saved for previous videos.")
            break

        np.savez_compressed(
            out_file,
            rep_states  = result["rep_states"],
            form_scores = result["form_scores"],
            exercise    = np.array(ex_name),
            video_id    = np.array(video_id),
        )
        print(f"  ✓ Saved → {out_file}")
        labeled += 1

    print(f"\n[Done]  Labeled: {labeled}  |  Skipped: {skipped}")


if __name__ == "__main__":
    main()
