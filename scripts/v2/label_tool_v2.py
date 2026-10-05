"""
scripts/label_tool_v2.py
═════════════════════════
Labeling Tool v2 — Per-Exercise States + Rep Boundaries + Per-Rep Quality

Workflow (two phases)
─────────────────────
  Phase 1 — Frame labeling + rep marking
    Label every frame with a state number, mark rep start/end with [s]/[e].
    When done, press [d] (done) to move to Phase 2.

  Phase 2 — Quality rating screen
    Each rep is shown one at a time.
    [LEFT]/[RIGHT] scrub between the rep's start and end frames.
    Press [0-9] to set a rating, then [ENTER] to confirm.
    Cannot advance to the next rep without confirming.
    After all reps are rated, [s] saves the file.

Controls — Phase 1
───────────────────
  [1..5]   assign state to current frame and advance
  [m]      copy last state to current frame and advance
  [n]      next frame (no state change)
  [b]      back one frame
  [z]      undo (200-level stack)
  [s]      mark rep START at current frame
  [e]      mark rep END at current frame (opens next rep slot)
  [x]      skip this video (bad / unusable)
  [d]      done labeling → go to Phase 2 (rating screen)
  [q]      quit without saving

Controls — Phase 2 (rating screen)
────────────────────────────────────
  [LEFT]/[RIGHT]   scrub through the rep's frames
  [0-9]            set quality rating for this rep
  [ENTER]          confirm rating → advance to next rep
                   (asks "Are you sure?" before confirming)
  [y]              confirm the "are you sure?" prompt
  [n]              cancel the "are you sure?" prompt
  [q]              quit without saving

Usage
─────
  python scripts/label_tool_v2.py
  python scripts/label_tool_v2.py --exercise squat
  python scripts/label_tool_v2.py --exercise squat --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Exercise configuration
# ══════════════════════════════════════════════════════════════════════════════

EXERCISE_CONFIG: Dict[str, Dict] = {
    "squat": {
        "timed": False,
        "states": [
            ("sq_standing",   "1", (180, 180, 180)),
            ("sq_descending", "2", ( 80, 160, 255)),
            ("sq_bottom",     "3", ( 60, 220,  60)),
            ("sq_ascending",  "4", (255, 160,  40)),
        ],
        "rep_note": "descending → bottom → ascending → standing",
    },
    "pushup": {
        "timed": False,
        "states": [
            ("pu_plank",      "1", (180, 180, 180)),
            ("pu_descending", "2", ( 80, 160, 255)),
            ("pu_bottom",     "3", ( 60, 220,  60)),
            ("pu_ascending",  "4", (255, 160,  40)),
        ],
        "rep_note": "plank → descending → bottom → ascending → plank",
    },
    "situp": {
        "timed": False,
        "states": [
            ("su_lying",      "1", (180, 180, 180)),
            ("su_ascending",  "2", ( 80, 160, 255)),
            ("su_top",        "3", ( 60, 220,  60)),
            ("su_descending", "4", (255, 160,  40)),
        ],
        "rep_note": "lying → ascending → top → descending → lying",
    },
    "pullup": {
        "timed": False,
        "states": [
            ("pu_hanging",  "1", (180, 180, 180)),
            ("pu_pulling",  "2", ( 80, 160, 255)),
            ("pu_top",      "3", ( 60, 220,  60)),
            ("pu_lowering", "4", (255, 160,  40)),
        ],
        "rep_note": "hanging → pulling → top → lowering → hanging",
    },
    "lunge": {
        "timed": False,
        "states": [
            ("lu_standing",  "1", (180, 180, 180)),
            ("lu_stepping",  "2", ( 80, 160, 255)),
            ("lu_bottom",    "3", ( 60, 220,  60)),
            ("lu_returning", "4", (255, 160,  40)),
        ],
        "rep_note": "standing → stepping → bottom → returning → standing",
    },
    "bicep_curl": {
        "timed": False,
        "states": [
            ("bc_extended", "1", (180, 180, 180)),
            ("bc_curling",  "2", ( 80, 160, 255)),
            ("bc_top",      "3", ( 60, 220,  60)),
            ("bc_lowering", "4", (255, 160,  40)),
        ],
        "rep_note": "extended → curling → top → lowering → extended",
    },
    "tricep_dip": {
        "timed": False,
        "states": [
            ("td_top",        "1", (180, 180, 180)),
            ("td_descending", "2", ( 80, 160, 255)),
            ("td_bottom",     "3", ( 60, 220,  60)),
            ("td_ascending",  "4", (255, 160,  40)),
        ],
        "rep_note": "top → descending → bottom → ascending → top",
    },
    "lateral_raise": {
        "timed": False,
        "states": [
            ("lr_down",     "1", (180, 180, 180)),
            ("lr_raising",  "2", ( 80, 160, 255)),
            ("lr_top",      "3", ( 60, 220,  60)),
            ("lr_lowering", "4", (255, 160,  40)),
        ],
        "rep_note": "down → raising → top → lowering → down",
    },
    "shoulder_press": {
        "timed": False,
        "states": [
            ("sp_rack",     "1", (180, 180, 180)),
            ("sp_pressing", "2", ( 80, 160, 255)),
            ("sp_top",      "3", ( 60, 220,  60)),
            ("sp_lowering", "4", (255, 160,  40)),
        ],
        "rep_note": "rack → pressing → top → lowering → rack",
    },
    "jumping_jack": {
        "timed": False,
        "states": [
            ("jj_standing", "1", (180, 180, 180)),
            ("jj_airborne", "2", (255,  80, 200)),
            ("jj_wide",     "3", ( 60, 220,  60)),
            ("jj_closing",  "4", (255, 160,  40)),
        ],
        "rep_note": "standing → airborne → wide → closing → standing",
    },
    "high_knees": {
        "timed": False,
        "states": [
            ("hk_standing",  "1", (180, 180, 180)),
            ("hk_knee_up",   "2", ( 60, 220,  60)),
            ("hk_knee_down", "3", ( 80, 160, 255)),
        ],
        "rep_note": "standing → knee_up → knee_down  (any leg)",
    },
    "mountain_climber": {
        "timed": False,
        "states": [
            ("mc_plank",       "1", (180, 180, 180)),
            ("mc_knee_in",     "2", ( 80, 160, 255)),
            ("mc_knee_flexed", "3", ( 60, 220,  60)),
        ],
        "rep_note": "plank → knee_in → knee_flexed  (any leg)",
    },
    "plank": {
        "timed": True,
        "states": [("pk_hold", "1", (255, 200, 50))],
        "rep_note": "timed hold only",
    },
    "wall_sit": {
        "timed": True,
        "states": [("ws_hold", "1", (255, 200, 50))],
        "rep_note": "timed hold only",
    },
    "burpee": {
        "timed": False,
        "states": [
            ("bu_standing",   "1", (180, 180, 180)),
            ("bu_squat_down", "2", ( 80, 160, 255)),
            ("bu_plank",      "3", (255, 200,  50)),
            ("bu_squat_up",   "4", (255, 160,  40)),
            ("bu_jump",       "5", (255,  80, 200)),
        ],
        "rep_note": "standing → squat_down → plank → squat_up → jump → standing",
    },
}

SKELETON: List[Tuple[int, int]] = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Rep data structure
# ══════════════════════════════════════════════════════════════════════════════

class Rep:
    def __init__(self, start: int) -> None:
        self.start:   int           = start
        self.end:     Optional[int] = None
        self.quality: Optional[int] = None

    @property
    def is_closed(self) -> bool:
        return self.end is not None

    @property
    def is_rated(self) -> bool:
        return self.quality is not None

    @property
    def quality_pct(self) -> int:
        if self.quality is None:
            return 0
        return max(10, self.quality * 10)

    def quality_color(self) -> Tuple[int,int,int]:
        if self.quality is None:
            return (80, 80, 80)
        if self.quality >= 7:
            return (60, 200, 60)
        if self.quality >= 4:
            return (40, 180, 220)
        return (60, 60, 200)


# ══════════════════════════════════════════════════════════════════════════════
#  LabelSession — Phase 1: frame labeling + rep marking
# ══════════════════════════════════════════════════════════════════════════════

class LabelSession:
    _W, _H  = 860, 700
    _SKEL_W = 460
    _INFO_X = 474
    _FONT   = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, keypoints, scores, exercise, video_id, existing=None):
        self.kp        = keypoints
        self.sc        = scores
        self.exercise  = exercise
        self.video_id  = video_id
        self.n_frames  = len(keypoints)
        self.cfg       = EXERCISE_CONFIG[exercise]
        self.is_timed  = self.cfg["timed"]

        # key char → state index  (1,2,3,4,5)
        self._key_map: Dict[str, int] = {
            s[1]: i for i, s in enumerate(self.cfg["states"])
        }

        if existing and "frame_states" in existing:
            self.frame_states: List[int] = existing["frame_states"].tolist()
        else:
            self.frame_states = [0] * self.n_frames

        self.reps: List[Rep] = []
        if existing and "rep_starts" in existing:
            starts    = existing["rep_starts"].tolist()
            ends      = existing["rep_ends"].tolist()
            qualities = existing["rep_qualities"].tolist()
            for st, en, q in zip(starts, ends, qualities):
                r = Rep(st); r.end = en; r.quality = q
                self.reps.append(r)

        self.frame_idx    = 0
        self._last_state  = 0
        self._undo_stack: List[Tuple] = []
        self._MAX_UNDO    = 300
        self._open_rep: Optional[Rep] = None

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self):
        """
        Returns:
          dict       → proceed to rating screen
          "SKIP"     → skip this video
          None       → quit entirely
        """
        win = f"Label v2 — {self.exercise} / {self.video_id}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self._W, self._H)

        keys_str = "  ".join(f"[{s[1]}]={s[0]}" for s in self.cfg["states"])
        print(f"\n[Phase 1] {self.exercise} / {self.video_id}  ({self.n_frames} frames)")
        print(f"  States: {keys_str}")
        if not self.is_timed:
            print(f"  Reps  : [s]=start rep  [e]=end rep")
        print(f"  Nav   : [m]=copy+next  [n]=next  [b]=back  [z]=undo")
        print(f"  Done  : [d]=go to rating  [x]=skip video  [q]=quit")

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(win)
                return None

            if key == ord('x'):
                cv2.destroyWindow(win)
                return "SKIP"

            if key == ord('d'):
                # Validate before leaving phase 1
                if self._open_rep is not None:
                    print("  [BLOCKED] Close the open rep first — press [e]")
                    continue
                if not self.is_timed and len(self.reps) == 0:
                    print("  [BLOCKED] Mark at least one rep with [s]...[e]")
                    continue
                cv2.destroyWindow(win)
                return self._pack_partial()

            if key == ord('n'):
                self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                continue

            if key == ord('b'):
                self.frame_idx = max(0, self.frame_idx - 1)
                continue

            if key == ord('z'):
                self._undo()
                continue

            if key == ord('m'):
                self._push_undo()
                self.frame_states[self.frame_idx] = self._last_state
                self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                continue

            # Rep start
            if key == ord('s') and not self.is_timed:
                if self._open_rep is not None:
                    print("  [WARN] Rep already open — press [e] to end it first")
                    continue
                r = Rep(self.frame_idx)
                self.reps.append(r)
                self._open_rep = r
                continue

            # Rep end
            if key == ord('e') and not self.is_timed:
                if self._open_rep is None:
                    print("  [WARN] No open rep — press [s] to start one first")
                    continue
                if self.frame_idx <= self._open_rep.start:
                    print("  [WARN] End frame must be after start frame")
                    continue
                self._open_rep.end = self.frame_idx
                self._open_rep     = None
                continue

            # State keys (1-5)
            char = chr(key) if key < 128 else ''
            if char in self._key_map:
                self._push_undo()
                idx = self._key_map[char]
                self.frame_states[self.frame_idx] = idx
                self._last_state = idx
                self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                continue

        cv2.destroyWindow(win)
        return None

    # ── undo ─────────────────────────────────────────────────────────────────

    def _push_undo(self):
        snap = (
            self.frame_idx,
            self.frame_states[self.frame_idx],
            [r.__dict__.copy() for r in self.reps],
        )
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            return
        idx, state, reps_snap = self._undo_stack.pop()
        self.frame_states[idx] = state
        self.frame_idx = idx
        self.reps = []
        for d in reps_snap:
            r = Rep(d["start"]); r.end = d["end"]; r.quality = d["quality"]
            self.reps.append(r)
        self._open_rep = next((r for r in self.reps if not r.is_closed), None)

    # ── pack partial (no ratings yet) ────────────────────────────────────────

    def _pack_partial(self) -> Dict:
        state_names = [s[0] for s in self.cfg["states"]]
        return {
            "frame_states":      np.array(self.frame_states, dtype=np.int32),
            "frame_state_names": np.array(
                [state_names[i] for i in self.frame_states], dtype=object),
            "reps":       self.reps,          # List[Rep] passed to RatingScreen
            "state_names": np.array(state_names, dtype=object),
            "exercise":   self.exercise,
            "video_id":   self.video_id,
            "n_frames":   self.n_frames,
        }

    # ── rendering ────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        canvas = np.zeros((self._H, self._W, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (self._SKEL_W, 0), (self._SKEL_W+2, self._H),
                      (50,50,50), -1)

        kp     = self.kp[self.frame_idx]
        sc_row = self.sc[self.frame_idx]
        s_idx  = self.frame_states[self.frame_idx]
        col    = self.cfg["states"][s_idx][2]

        for a, b in SKELETON:
            pa = (int(kp[a,0]), int(kp[a,1]))
            pb = (int(kp[b,0]), int(kp[b,1]))
            if pa == (0,0) or pb == (0,0): continue
            cv2.line(canvas, pa, pb, col, 2, cv2.LINE_AA)
        for i, p in enumerate(kp):
            x, y = int(p[0]), int(p[1])
            if x == 0 and y == 0: continue
            dot = (0,255,255) if sc_row[i] > 0.3 else (60,60,60)
            cv2.circle(canvas, (x,y), 4, dot, -1, cv2.LINE_AA)

        if self._open_rep is not None:
            cv2.rectangle(canvas, (0,0), (self._SKEL_W-2, 4), (0,180,255), -1)

        self._draw_info(canvas, s_idx, col)
        self._draw_progress(canvas)
        return canvas

    def _put(self, c, text, x, y, color=(210,210,210), scale=0.48, thick=1):
        cv2.putText(c, text, (x,y), self._FONT, scale, color, thick, cv2.LINE_AA)

    def _draw_info(self, canvas, s_idx, col):
        ix = self._INFO_X
        self._put(canvas, self.exercise.upper(), ix, 28, (0,220,255), 0.60, 2)
        self._put(canvas, self.video_id, ix, 50, (160,160,160), 0.42)
        self._put(canvas, f"Frame {self.frame_idx+1}/{self.n_frames}", ix, 76)
        pct = (self.frame_idx+1)/self.n_frames*100
        self._put(canvas, f"Progress {pct:.0f}%", ix, 96, (120,120,120), 0.40)

        # State badge
        cv2.rectangle(canvas, (ix-4,108), (self._W-6,132), (30,30,30), -1)
        sname = self.cfg["states"][s_idx][0]
        self._put(canvas, f"STATE: {sname}", ix, 127, col, 0.52)

        # Rep status
        rep_y = 150
        if self.is_timed:
            self._put(canvas, "TIMED HOLD — no rep marking", ix, rep_y,
                      (255,200,50), 0.42)
        elif self._open_rep is not None:
            self._put(canvas, f"REP OPEN  start={self._open_rep.start+1}",
                      ix, rep_y, (0,180,255), 0.44)
            self._put(canvas, "press [e] to mark end", ix, rep_y+16,
                      (100,140,200), 0.38)
        else:
            self._put(canvas, f"Reps marked: {len(self.reps)}", ix, rep_y,
                      (160,160,160), 0.42)
            self._put(canvas, "[s]=start rep", ix, rep_y+16, (80,80,80), 0.38)

        # State legend
        leg_y = 188
        self._put(canvas, "── States ──", ix, leg_y, (80,80,80), 0.38)
        for i, (sn, sk, sc2) in enumerate(self.cfg["states"]):
            y = leg_y + 18 + i*20
            is_cur = (i == s_idx)
            self._put(canvas, f"{'>'if is_cur else ' '} [{sk}] {sn}",
                      ix, y, sc2 if is_cur else (80,80,80), 0.40)

        # Rep list
        rly = leg_y + 18 + len(self.cfg["states"])*20 + 10
        self._put(canvas, "── Reps ──", ix, rly, (80,80,80), 0.38)
        for i, r in enumerate(self.reps[-6:]):
            y = rly + 16 + i*18
            if not r.is_closed:
                self._put(canvas, f"Rep{i+1}  {r.start+1}..?",
                          ix, y, (0,180,255), 0.38)
            else:
                self._put(canvas, f"Rep{i+1}  {r.start+1}..{r.end+1}  (rate later)",
                          ix, y, (120,120,120), 0.38)

        # Bottom hint
        self._put(canvas, "[d]=done→rating  [s/e]=rep  [m/n/b/z]=nav  [x]=skip  [q]=quit",
                  ix, self._H-22, (60,60,60), 0.34)

    def _draw_progress(self, canvas):
        bx, by = 10, self._H-10
        bw, bh = self._W-20, 6
        cv2.rectangle(canvas, (bx, by-bh), (bx+bw, by), (35,35,35), -1)
        for r in self.reps:
            x1 = bx + int(r.start/self.n_frames*bw)
            x2 = bx + int((r.end or self.frame_idx)/self.n_frames*bw)
            cv2.rectangle(canvas, (x1, by-bh), (x2, by), (0,140,220), -1)
        if self._open_rep:
            x1 = bx + int(self._open_rep.start/self.n_frames*bw)
            x2 = bx + int(self.frame_idx/self.n_frames*bw)
            cv2.rectangle(canvas, (x1, by-bh), (x2, by), (0,180,255), -1)
        cx = bx + int(self.frame_idx/self.n_frames*bw)
        cv2.rectangle(canvas, (cx-1, by-bh-2), (cx+1, by+2), (255,255,255), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  RatingScreen — Phase 2: rate each rep
# ══════════════════════════════════════════════════════════════════════════════

class RatingScreen:
    """
    Shows each rep one at a time.
    User scrubs between start/end with LEFT/RIGHT arrows.
    Presses 0-9 to set rating, ENTER to confirm (with "are you sure?" prompt).
    Cannot proceed to next rep until current one is confirmed.
    """

    _W, _H  = 860, 700
    _SKEL_W = 460
    _INFO_X = 474
    _FONT   = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, keypoints, scores, partial: Dict):
        self.kp          = keypoints
        self.sc          = scores
        self.partial     = partial
        self.reps: List[Rep] = partial["reps"]
        self.exercise    = partial["exercise"]
        self.cfg         = EXERCISE_CONFIG[self.exercise]

        self.rep_idx     = 0          # which rep we're rating now
        self.frame_idx   = self.reps[0].start if self.reps else 0

        self._pending_rating: Optional[int] = None   # digit pressed, not yet confirmed
        self._confirm_mode   = False                 # waiting for y/n

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self) -> Optional[Dict]:
        """
        Returns final packed dict on success, None on quit.
        """
        if not self.reps:
            return self._pack_final()

        win = f"Rating — {self.exercise} / {self.partial['video_id']}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self._W, self._H)

        print(f"\n[Phase 2] Rating {len(self.reps)} reps for {self.exercise}")
        print(f"  LEFT/RIGHT = scrub frames   [0-9] = set rating   ENTER = confirm")
        print(f"  [q] = quit without saving")

        while self.rep_idx < len(self.reps):
            rep = self.reps[self.rep_idx]
            cv2.imshow(win, self._render(rep))
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(win)
                return None

            # ── Confirm mode (are you sure?) ──────────────────────────────────
            if self._confirm_mode:
                if key == ord('y') or key in (13, 10):
                    rep.quality          = self._pending_rating
                    self._pending_rating = None
                    self._confirm_mode   = False
                    self.rep_idx        += 1
                    if self.rep_idx < len(self.reps):
                        self.frame_idx = self.reps[self.rep_idx].start
                elif key == ord('n'):
                    self._confirm_mode   = False
                    self._pending_rating = None
                continue

            # ── Scrub frames within this rep ──────────────────────────────────
            if key in (81, 2424832, 65361):   # LEFT arrow variants
                self.frame_idx = max(rep.start, self.frame_idx - 1)
                continue
            if key in (83, 2555904, 65363):   # RIGHT arrow variants
                self.frame_idx = min(rep.end, self.frame_idx + 1)
                continue

            # ── Set rating ────────────────────────────────────────────────────
            char = chr(key) if key < 128 else ''
            if char in "0123456789":
                self._pending_rating = int(char)
                continue

            # ── Confirm with ENTER ────────────────────────────────────────────
            if key in (13, 10):
                if self._pending_rating is None:
                    print(f"  [WARN] Press [0-9] to set a rating first")
                    continue
                self._confirm_mode = True
                continue

        cv2.destroyWindow(win)
        return self._pack_final()

    # ── pack ─────────────────────────────────────────────────────────────────

    def _pack_final(self) -> Dict:
        p = self.partial
        return {
            "frame_states":      p["frame_states"],
            "frame_state_names": p["frame_state_names"],
            "rep_starts":   np.array([r.start   for r in self.reps], dtype=np.int32),
            "rep_ends":     np.array([r.end     for r in self.reps], dtype=np.int32),
            "rep_qualities":np.array([r.quality for r in self.reps], dtype=np.int32),
            "exercise":     p["exercise"],
            "video_id":     p["video_id"],
            "n_frames":     p["n_frames"],
            "state_names":  p["state_names"],
        }

    # ── rendering ────────────────────────────────────────────────────────────

    def _render(self, rep: Rep) -> np.ndarray:
        canvas = np.zeros((self._H, self._W, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (self._SKEL_W, 0), (self._SKEL_W+2, self._H),
                      (50,50,50), -1)

        # Header band — rating screen indicator
        cv2.rectangle(canvas, (0, 0), (self._SKEL_W-2, 5), (180, 100, 0), -1)

        kp     = self.kp[self.frame_idx]
        sc_row = self.sc[self.frame_idx]
        s_idx  = self.partial["frame_states"][self.frame_idx]
        col    = self.cfg["states"][s_idx][2]

        for a, b in SKELETON:
            pa = (int(kp[a,0]), int(kp[a,1]))
            pb = (int(kp[b,0]), int(kp[b,1]))
            if pa == (0,0) or pb == (0,0): continue
            cv2.line(canvas, pa, pb, col, 2, cv2.LINE_AA)
        for i, p in enumerate(kp):
            x, y = int(p[0]), int(p[1])
            if x == 0 and y == 0: continue
            dot = (0,255,255) if sc_row[i] > 0.3 else (60,60,60)
            cv2.circle(canvas, (x,y), 4, dot, -1, cv2.LINE_AA)

        self._draw_info(canvas, rep)
        self._draw_progress(canvas, rep)
        return canvas

    def _put(self, c, text, x, y, color=(210,210,210), scale=0.48, thick=1):
        cv2.putText(c, text, (x,y), self._FONT, scale, color, thick, cv2.LINE_AA)

    def _draw_info(self, canvas, rep: Rep):
        ix = self._INFO_X
        W  = self._W

        self._put(canvas, "RATING SCREEN", ix, 24, (180,100,0), 0.52, 2)
        self._put(canvas, self.exercise.upper(), ix, 46, (0,220,255), 0.55, 1)

        # Rep counter
        self._put(canvas, f"Rep  {self.rep_idx+1}  of  {len(self.reps)}",
                  ix, 75, (210,210,210), 0.55)
        self._put(canvas, f"Frames  {rep.start+1} → {rep.end+1}",
                  ix, 96, (140,140,140), 0.42)
        self._put(canvas, f"Viewing  frame {self.frame_idx+1}",
                  ix, 114, (140,140,140), 0.42)

        # Rating display
        rating_y = 145
        cv2.rectangle(canvas, (ix-4, rating_y-14), (W-6, rating_y+30),
                      (30,30,30), -1)
        if self._confirm_mode:
            pct = max(10, self._pending_rating * 10)
            qcol = (60,200,60) if self._pending_rating>=7 else \
                   (40,180,220) if self._pending_rating>=4 else (60,60,200)
            self._put(canvas, f"Rating: {self._pending_rating}  ({pct}%)",
                      ix, rating_y, qcol, 0.55, 2)
            self._put(canvas, "Are you sure?  [y]=yes  [n]=no",
                      ix, rating_y+22, (255,180,0), 0.44)
        elif self._pending_rating is not None:
            pct = max(10, self._pending_rating * 10)
            self._put(canvas, f"Rating: {self._pending_rating}  ({pct}%)",
                      ix, rating_y, (200,200,200), 0.55)
            self._put(canvas, "press ENTER to confirm",
                      ix, rating_y+22, (120,120,120), 0.40)
        else:
            self._put(canvas, "No rating yet", ix, rating_y, (80,80,80), 0.50)
            self._put(canvas, "press [0-9] to set rating",
                      ix, rating_y+22, (80,80,80), 0.40)

        # Rating scale reference
        scale_y = 210
        self._put(canvas, "Scale:  0-3 = bad   4-6 = ok   7-9 = good",
                  ix, scale_y, (80,80,80), 0.38)
        self._put(canvas, "0/1 → 10%   5 → 50%   9 → 90%",
                  ix, scale_y+16, (60,60,60), 0.36)

        # Previous reps summary
        prev_y = 248
        self._put(canvas, "── Rated so far ──", ix, prev_y, (80,80,80), 0.38)
        for i, r in enumerate(self.reps):
            y = prev_y + 16 + i*18
            if i == self.rep_idx:
                marker = ">"
                c = (255,255,255)
            elif r.is_rated:
                marker = "✓"
                c = r.quality_color()
            else:
                marker = " "
                c = (60,60,60)
            txt = f"{marker} Rep{i+1}"
            if r.is_rated:
                txt += f"  {r.quality_pct}%"
            elif i < self.rep_idx:
                txt += "  (skipped?)"
            self._put(canvas, txt, ix, y, c, 0.38)

        # Controls at bottom
        self._put(canvas,
                  "LEFT/RIGHT=scrub   [0-9]=rate   ENTER=confirm   [q]=quit",
                  ix, self._H-22, (60,60,60), 0.34)

    def _draw_progress(self, canvas, rep: Rep):
        """Progress bar showing position within this rep only."""
        bx, by = 10, self._H-10
        bw, bh = self._W-20, 6
        cv2.rectangle(canvas, (bx, by-bh), (bx+bw, by), (35,35,35), -1)

        # Rep span highlighted
        rep_len = max(rep.end - rep.start, 1)
        x1 = bx
        x2 = bx + bw
        cv2.rectangle(canvas, (x1, by-bh), (x2, by), (50,50,80), -1)

        # Current position within rep
        pos_frac = (self.frame_idx - rep.start) / rep_len
        cx = bx + int(pos_frac * bw)
        cv2.rectangle(canvas, (cx-2, by-bh-2), (cx+2, by+2), (255,255,255), -1)

        # Start / end markers
        cv2.rectangle(canvas, (bx, by-bh-4), (bx+3, by), (0,200,200), -1)
        cv2.rectangle(canvas, (bx+bw-3, by-bh-4), (bx+bw, by), (200,100,0), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Tool v2 — states + rep boundaries + quality rating"
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--labeled-dir",   default="data/labeled_v2")
    parser.add_argument("--exercise",      default=None)
    parser.add_argument("--overwrite",     action="store_true")
    args = parser.parse_args()

    proc_dir    = Path(args.processed_dir)
    labeled_dir = Path(args.labeled_dir)

    if not proc_dir.exists():
        print(f"[ERROR] processed-dir not found: {proc_dir}")
        sys.exit(1)

    npz_files = []
    for ex_dir in sorted(proc_dir.iterdir()):
        if not ex_dir.is_dir(): continue
        ex_name = ex_dir.name
        if args.exercise and ex_name != args.exercise: continue
        if ex_name not in EXERCISE_CONFIG:
            print(f"  [SKIP] {ex_name} — not in EXERCISE_CONFIG")
            continue
        for npz in sorted(ex_dir.glob("*.npz")):
            npz_files.append((ex_name, npz))

    if not npz_files:
        print(f"[ERROR] No .npz files found under {proc_dir}")
        sys.exit(1)

    print(f"\n[Label Tool v2]  {len(npz_files)} video(s)  →  {labeled_dir}\n")

    labeled = skipped = 0

    for ex_name, npz_path in npz_files:
        video_id = npz_path.stem
        out_dir  = labeled_dir / ex_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{video_id}.npz"

        if out_file.exists() and not args.overwrite:
            print(f"  [SKIP]  {ex_name}/{video_id}")
            skipped += 1
            continue

        print(f"  [LOAD]  {ex_name}/{video_id}")
        try:
            data = np.load(npz_path, allow_pickle=True)
            kp   = data["keypoints"]
            sc   = data["scores"]
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        existing = None
        if out_file.exists():
            try:
                existing = dict(np.load(out_file, allow_pickle=True))
                print(f"  [RESUME] existing labels found")
            except Exception:
                pass

        # ── Phase 1: label frames + mark reps ────────────────────────────────
        phase1 = LabelSession(kp, sc, ex_name, video_id, existing)
        result = phase1.run()

        if result is None:
            print("  [QUIT]")
            break

        if result == "SKIP":
            print(f"  [SKIP]  {ex_name}/{video_id} — marked unusable")
            skipped += 1
            continue

        # result is the partial dict with reps list
        partial = result

        # Timed exercises: no rating screen needed
        if EXERCISE_CONFIG[ex_name]["timed"]:
            final = {
                "frame_states":      partial["frame_states"],
                "frame_state_names": partial["frame_state_names"],
                "rep_starts":    np.array([], dtype=np.int32),
                "rep_ends":      np.array([], dtype=np.int32),
                "rep_qualities": np.array([], dtype=np.int32),
                "exercise":      partial["exercise"],
                "video_id":      partial["video_id"],
                "n_frames":      partial["n_frames"],
                "state_names":   partial["state_names"],
            }
        else:
            # ── Phase 2: rate each rep ────────────────────────────────────────
            rating = RatingScreen(kp, sc, partial)
            final  = rating.run()

            if final is None:
                print("  [QUIT]")
                break

        np.savez_compressed(
            out_file,
            frame_states       = final["frame_states"],
            frame_state_names  = final["frame_state_names"],
            rep_starts         = final["rep_starts"],
            rep_ends           = final["rep_ends"],
            rep_qualities      = final["rep_qualities"],
            exercise           = np.array(final["exercise"]),
            video_id           = np.array(final["video_id"]),
            n_frames           = np.array(final["n_frames"]),
            state_names        = final["state_names"],
        )
        n_reps = len(final["rep_starts"])
        print(f"  [SAVE]  {out_file}  ({n_reps} reps)")
        labeled += 1

    print(f"\n[Done]  Labeled: {labeled}  |  Skipped: {skipped}")


if __name__ == "__main__":
    main()
