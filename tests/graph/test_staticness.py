from indoor_loop.graph.staticness import classify_staticness


def test_classify_staticness_uses_category_prior() -> None:
    assert classify_staticness("person", observation_quality=0.9) == "dynamic"
    assert classify_staticness("bed", observation_quality=0.9) == "hard_static"
    assert classify_staticness("chair", observation_quality=0.9) == "soft_static"
