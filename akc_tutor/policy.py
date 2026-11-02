"""Portable transcription of the supplied Phase IV LESSONS/ACK/NEG/respond code.

This module performs no model inference, network request, or training. Inputs are
provided knowledge probabilities and emotion labels. The `text` argument is
retained for parity but is not used by the original response function.
"""
LESSONS = {
    "linear equations": {
        "core": "To solve 2x + 3 = 11, subtract 3 from both sides, then divide both sides by 2.",
        "hint": "First isolate the term containing x by subtracting 3 on both sides.",
        "challenge": "Try solving 3(x + 2) = 18 and check by substitution.",
        "check": "What operation undoes adding 3?",
    },
    "fractions": {
        "core": "To add 1/2 and 1/3, use denominator 6: 3/6 + 2/6 = 5/6.",
        "hint": "Find the least common denominator of 2 and 3 before adding.",
        "challenge": "Try 3/4 + 2/5 and simplify your result.",
        "check": "What common denominator works for 1/2 and 1/3?",
    },
    "probability": {
        "core": "For independent events A and B, P(A and B) = P(A) × P(B).",
        "hint": "Check whether the events are independent before multiplying their probabilities.",
        "challenge": "A fair coin is tossed twice. What is the chance of two heads?",
        "check": "If P(A)=1/2 and P(B)=1/3 and they are independent, what is P(A and B)?",
    },
}
ACK = {
    "confusion": "It is understandable to need another explanation.",
    "frustration": "Repeated mistakes can feel frustrating; we can work through one step.",
    "annoyance": "We can slow down and focus on one step.",
    "anger": "We can approach this calmly, one step at a time.",
    "nervousness": "We can begin with a small, manageable step.",
    "fear": "We can start with a simpler example.",
    "sadness": "We can work through this together at a comfortable pace.",
    "curiosity": "That is a useful question to explore.",
    "excitement": "Great enthusiasm; let us extend the idea.",
    "gratitude": "You are welcome; let us check the idea.",
    "neutral": "Let us work through the concept.",
}
NEG = {"confusion", "frustration", "annoyance", "anger", "nervousness",
       "fear", "sadness", "grief", "disappointment", "embarrassment", "remorse"}
VARIANTS = {
    "generic": (False, False),
    "knowledge_only": (True, False),
    "emotion_only": (False, True),
    "knowledge_plus_predicted_emotion": (True, True),
}


def respond(skill, text, knowledge_prob=None, emotion=None,
            use_knowledge=True, use_emotion=True):
    lesson = LESSONS[skill]
    if not use_knowledge:
        level = "medium"
    elif knowledge_prob < .45:
        level = "low"
    elif knowledge_prob > .75:
        level = "high"
    else:
        level = "medium"
    if level == "low":
        content, action = lesson["core"], "worked_example"
    elif level == "high":
        content, action = lesson["challenge"], "challenge"
    else:
        content, action = lesson["hint"], "hint"
    if use_emotion and emotion in NEG and level == "high":
        content, action = lesson["hint"], "hint"
    ack = ACK.get(emotion, "Let us work through the concept.") if use_emotion else "Let us work through the concept."
    return {"response": f'{ack} {content} {lesson["check"]}', "action": action,
            "knowledge_level": level, "used_emotion": emotion if use_emotion else None}
