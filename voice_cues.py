"""
voice_cues.py
═════════════
Natural English pre-exercise coach scripts.

These scripts are intentionally short: they should play before a set starts,
not compete with the real-time form feedback loop.
"""

from __future__ import annotations

from typing import Dict, List

from exercises import Exercise, REGISTRY


VOICE_CUES: Dict[Exercise, List[str]] = {
    Exercise.SQUAT: [
        "Stand with your feet about shoulder-width apart and your toes slightly turned out.",
        "Brace your core, keep your chest proud, and send your hips back as you bend your knees.",
        "Drive your knees in line with your toes, then stand tall through your heels.",
        "Move with control and keep breathing.",
    ],
    Exercise.PUSHUP: [
        "Set your hands just outside shoulder width and build one straight line from shoulders to heels.",
        "Brace your abs and squeeze your glutes before you lower.",
        "Keep your elbows at a comfortable angle, lower your chest with control, then press the floor away.",
        "Reset your body line before every rep.",
    ],
    Exercise.JUMPING_JACK: [
        "Stand tall with your feet together and arms relaxed by your sides.",
        "Jump your feet out as your arms sweep overhead, then return to the start position.",
        "Land softly, stay light on your feet, and keep a steady rhythm.",
        "Start smooth, then build speed once the movement feels clean.",
    ],
    Exercise.HIGH_KNEES: [
        "Stand tall with your ribs stacked over your hips.",
        "Run in place and drive one knee up toward hip height at a time.",
        "Pump your arms naturally, stay on the balls of your feet, and avoid leaning back.",
        "Keep the steps quick, light, and controlled.",
    ],
    Exercise.PLANK: [
        "Set your elbows under your shoulders and step your feet back.",
        "Brace your core, squeeze your glutes, and make a straight line from head to heels.",
        "Keep your hips from dropping or lifting too high.",
        "Hold steady and breathe slowly.",
    ],
    Exercise.PULLUP: [
        "Grip the bar firmly and start from a controlled dead hang.",
        "Pull your shoulder blades down before you bend your elbows.",
        "Drive your chest toward the bar, then lower with control until your arms are extended again.",
        "Avoid swinging and own each rep.",
    ],
    Exercise.SITUP: [
        "Lie on your back with your knees bent and feet planted.",
        "Brace your core, keep your neck relaxed, and curl your torso up smoothly.",
        "Come up with control, then lower back down without dropping.",
        "Let your abs do the work, not your neck.",
    ],
    Exercise.LUNGE: [
        "Stand tall, brace your core, and step forward with control.",
        "Lower until both knees bend naturally and your front knee tracks over your toes.",
        "Keep your torso upright, then push through the front foot to stand back up.",
        "Alternate sides only after you feel balanced.",
    ],
    Exercise.MOUNTAIN_CLIMBER: [
        "Start in a strong high plank with your hands under your shoulders.",
        "Brace your core and keep your hips level.",
        "Drive one knee toward your chest, switch legs, and keep the plank shape steady.",
        "Move quickly, but do not let your hips bounce.",
    ],
    Exercise.BURPEE: [
        "Start standing tall with your feet about shoulder-width apart.",
        "Squat down, place your hands on the floor, and step or jump back to a solid plank.",
        "Return your feet under you, stand fully tall, and add the jump only if it feels controlled.",
        "Keep the sequence clean before you chase speed.",
    ],
    Exercise.BICEP_CURL: [
        "Stand tall with your elbows close to your sides and your core lightly braced.",
        "Curl the weight up without leaning back or swinging.",
        "Squeeze at the top, then lower until your arms are nearly straight.",
        "Keep the elbows quiet and make the weight move smoothly.",
    ],
    Exercise.TRICEP_DIP: [
        "Set your hands firmly behind you and keep your chest open.",
        "Bend your elbows to lower your body while keeping your shoulders away from your ears.",
        "Press through your hands until your arms are straight again.",
        "Stay close to the bench or chair and keep the motion controlled.",
    ],
    Exercise.LATERAL_RAISE: [
        "Stand tall with a light brace through your core.",
        "Raise both arms out to the sides until they reach about shoulder height.",
        "Keep a soft bend in your elbows and avoid shrugging your shoulders.",
        "Lower slowly and keep both sides moving evenly.",
    ],
    Exercise.SHOULDER_PRESS: [
        "Start with the weights at shoulder height and your core braced.",
        "Keep your ribs down and press straight overhead.",
        "Finish with your arms strong above you, then lower back to shoulder level with control.",
        "Avoid arching your lower back as you press.",
    ],
    Exercise.WALL_SIT: [
        "Place your back flat against the wall and walk your feet forward.",
        "Slide down until your knees are close to a right angle.",
        "Keep your back pressed into the wall and your weight through your heels.",
        "Hold still, breathe, and keep your knees tracking forward.",
    ],
}


def get_voice_cues(exercise: Exercise) -> List[str]:
    """Return the cue list for an exercise, falling back to a generic setup."""
    return list(
        VOICE_CUES.get(
            exercise,
            [
                "Set your position carefully and brace before you start.",
                "Move through the full range you can control.",
                "Keep your breathing steady and stop if anything feels sharp or unsafe.",
            ],
        )
    )


def build_voice_script(exercise: Exercise, language: str = "en-US") -> str:
    """Build a single natural spoken script for the requested exercise."""
    reg = REGISTRY[exercise]
    cues = " ".join(get_voice_cues(exercise))
    return f"Let's get ready for {reg.display_name}. {cues} Ready when you are."
