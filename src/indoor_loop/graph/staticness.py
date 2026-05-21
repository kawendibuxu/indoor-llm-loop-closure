from indoor_loop.types import StaticnessLevel


def classify_staticness(label: str, observation_quality: float) -> StaticnessLevel:
    if label == "person":
        return "dynamic"
    if label in {"wall", "floor", "ceiling", "bed", "cabinet", "window", "door"}:
        return "hard_static"
    if label in {"chair", "monitor", "table", "desk", "sofa"} and observation_quality >= 0.5:
        return "soft_static"
    return "dynamic"
