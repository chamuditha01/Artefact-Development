"""
test_event_registry.py

Sanity tests for event_registry.py, built around REAL data already
collected this session - not synthetic examples. The fragment-union tests
use the actual CAL_002 / CAL_008 coordinates pulled from GEE (chat,
2026-09-27), the same two cases the 10m buffer decision (decisions.md)
was measured from. If these don't pass, the buffer decision itself needs
re-checking, not just the code.
"""

from datetime import date

from event_registry import EventRegistry, union_same_round_candidates, ClassificationState


# --- real coordinates, from this session's GEE console output -------------

CAL_002_PART_0 = {
    "type": "Polygon",
    "coordinates": [[
        [80.48133834280961, 7.776529525490816],
        [80.48079883976573, 7.7765295260892895],
        [80.48079883976573, 7.775989959413965],
        [80.48133834280961, 7.775989958088586],
        [80.48133834280961, 7.776529525490816],
    ]],
}

CAL_002_PART_1 = {
    "type": "Polygon",
    "coordinates": [[
        [80.48133834280961, 7.776529525490816],
        [80.48161034128398, 7.776529575432299],
        [80.48161034128398, 7.775722438013692],
        [80.48187792718483, 7.7757224699974605],
        [80.48187792718483, 7.776797098928556],
        [80.48133834280961, 7.776797098054861],
        [80.48133834280961, 7.776529525490816],
    ]],
}

CAL_008_PART_0 = {
    "type": "Polygon",
    "coordinates": [[
        [80.80769550928325, 7.238079687297326],
        [80.80796752144182, 7.238079737134366],
        [80.80796752144182, 7.238619236948086],
        [80.80769550928325, 7.238619268126671],
        [80.80769550928325, 7.238079687297326],
    ]],
}

CAL_008_PART_1 = {
    "type": "Polygon",
    "coordinates": [[
        [80.80796752144182, 7.236733040977801],
        [80.80796752144182, 7.236461101166532],
        [80.80877459528254, 7.236461065226472],
        [80.80877459528254, 7.237000634361084],
        [80.8090421252484, 7.2370005921801415],
        [80.8090421252484, 7.238079744261733],
        [80.80877459528254, 7.2380797043692136],
        [80.80877459528254, 7.237807677155004],
        [80.80823505409317, 7.2378077507173835],
        [80.80823505409317, 7.236733080541859],
        [80.80796752144182, 7.236733040977801],
    ]],
}

CAL_008_PART_2 = {
    "type": "Polygon",
    "coordinates": [[
        [80.80769550928325, 7.238079687297326],
        [80.8071559608529, 7.238079677652594],
        [80.8071559608529, 7.238347229427195],
        [80.80688842461122, 7.238347269451645],
        [80.80688842461122, 7.238079717850919],
        [80.80662088747944, 7.238079676495508],
        [80.80662088747944, 7.237272634632107],
        [80.8071559608529, 7.23727263473926],
        [80.8071559608529, 7.2370006118910375],
        [80.80688842461122, 7.237000652790603],
        [80.80688842461122, 7.236733106598137],
        [80.8071559608529, 7.236733065524704],
        [80.8071559608529, 7.236461044879842],
        [80.80769550928325, 7.236461052401091],
        [80.80769550928325, 7.236733073402813],
        [80.80796752144182, 7.236733040977801],
        [80.80796752144182, 7.237807710453729],
        [80.80769550928325, 7.237807742168585],
        [80.80769550928325, 7.238079687297326],
    ]],
}


def test_cal_002_two_touching_pieces_union_to_one():
    result = union_same_round_candidates([CAL_002_PART_0, CAL_002_PART_1])
    assert len(result) == 1, f"expected CAL_002's 2 touching pieces to union into 1, got {len(result)}"
    print("PASS: CAL_002 (2 pieces, 0m gap) unions to 1 shape")


def test_cal_008_three_chained_pieces_union_to_one():
    # part_0 and part_1 do NOT touch directly (measured ~89m apart) -
    # this is exactly the case a naive pairwise check would fail on.
    # They only join because part_2 touches both.
    result = union_same_round_candidates([CAL_008_PART_0, CAL_008_PART_1, CAL_008_PART_2])
    assert len(result) == 1, f"expected CAL_008's 3 chained pieces to union into 1, got {len(result)}"
    print("PASS: CAL_008 (3 pieces, chained through the middle piece) unions to 1 shape")


def test_far_apart_candidates_stay_separate():
    # Sanity check the union does NOT over-merge: two genuinely distant
    # patches (CAL_002's location vs CAL_008's location, ~30km apart)
    # must stay as two separate shapes.
    result = union_same_round_candidates([CAL_002_PART_0, CAL_008_PART_0])
    assert len(result) == 2, f"expected 2 separate shapes for genuinely distant patches, got {len(result)}"
    print("PASS: genuinely distant patches (CAL_002 vs CAL_008 locations) stay separate")


def test_registry_new_event_then_recurrence_years_later():
    reg = EventRegistry()
    cal_002_unioned = union_same_round_candidates([CAL_002_PART_0, CAL_002_PART_1])

    # round 1: first detection
    s1 = reg.process_round("R1", date(2019, 6, 1), cal_002_unioned)
    assert len(s1["new_events"]) == 1
    assert len(s1["updated_events"]) == 0
    eid = s1["new_events"][0]

    # round 2, THREE YEARS LATER, same location - must be an UPDATE, not a
    # new event, per D7's "no time cutoff" rule (the cultivation-cycling
    # signal depends on this).
    s2 = reg.process_round("R2", date(2022, 6, 1), cal_002_unioned)
    assert len(s2["new_events"]) == 0, "a recurrence 3 years later at the same spot must not create a new event"
    assert s2["updated_events"] == [eid]
    print("PASS: recurrence 3 years later at the same location updates the SAME event (no time cutoff)")


def test_registry_bridge_creates_merge_not_silent_combine():
    reg = EventRegistry()

    # two separate events, far apart within the same round
    reg.process_round("R1", date(2020, 1, 1), [CAL_002_PART_0])
    reg.process_round("R1", date(2020, 1, 1), [CAL_008_PART_0])
    ids_before = set(reg.events().keys())
    assert len(ids_before) == 2

    # A patch that happens to overlap BOTH is artificial here (CAL_002 and
    # CAL_008 are real, ~30km apart) - built by hand only to exercise the
    # merge-logging path, not claimed as a real measured case.
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    bridge = mapping(unary_union([shape(CAL_002_PART_0), shape(CAL_008_PART_0)]).convex_hull)

    s3 = reg.process_round("R2", date(2020, 2, 1), [bridge])
    assert len(s3["merges_logged"]) >= 1, "a patch bridging two events must log a merge, not silently combine them"
    assert len(reg.events()) == 2, "a bridged merge must NOT reduce the event count - both IDs must survive"

    for eid, ev in reg.events().items():
        other = [e for e in ids_before if e != eid][0]
        assert other in ev.linked_events, "each bridged event must record the other as linked"
    print("PASS: a bridging patch logs a merge and keeps both event IDs (no silent combine)")


def test_classification_history_is_append_only():
    reg = EventRegistry()
    s1 = reg.process_round("R1", date(2021, 1, 1), [CAL_002_PART_0])
    eid = s1["new_events"][0]

    reg.add_classification(eid, ClassificationState("R1", date(2021, 1, 1), "stage1", "uncertain", "medium"))
    reg.add_classification(eid, ClassificationState("R2", date(2021, 4, 1), "stage2", "cultivation", "high", "recovery trajectory confirmed"))

    hist = reg.events()[eid].classification_history
    assert len(hist) == 2, "both classification states must be kept, not overwritten"
    assert hist[0].label == "uncertain", "the FIRST state must still be readable, not replaced"
    assert reg.events()[eid].current_classification().label == "cultivation"
    print("PASS: classification history is append-only - earlier states survive later ones")


if __name__ == "__main__":
    test_cal_002_two_touching_pieces_union_to_one()
    test_cal_008_three_chained_pieces_union_to_one()
    test_far_apart_candidates_stay_separate()
    test_registry_new_event_then_recurrence_years_later()
    test_registry_bridge_creates_merge_not_silent_combine()
    test_classification_history_is_append_only()
    print("\nAll tests passed.")
