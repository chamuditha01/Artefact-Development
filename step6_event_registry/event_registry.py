"""
event_registry.py

Persistent event registry for the forest-loss monitoring loop.

WHAT THIS IS: matching and identity logic only - decides whether a new
candidate patch is an update to an existing tracked event or a new one,
and keeps classification history without overwriting it. This is
supporting machinery (project.md), not a claimed contribution - no
novelty here, just correct bookkeeping.

WHAT THIS IS NOT: it does not detect candidates (the detection loop is
still unbuilt) and does not classify them (the classifier is still
unbuilt). It takes candidate patches in, and gives event identity +
footprint history + classification history out.

Design rule (decisions.md, 2026-09-27, closes issues.md D7):
  - Match by LOCATION only (buffered overlap). No time-based cutoff - a
    disturbance recurring years later at the same spot is the SAME event,
    not a new one. This preserves the repeated clear-regrow-clear signal
    the driver classifier depends on to spot cultivation.
  - Buffer = 10m (one Sentinel-2 pixel), measured from the two known
    multi-part vectorisation artefacts (issues.md B17: CAL_002, CAL_008).
    n=2 - thin evidence, stated as such in decisions.md. Re-measure if
    more multi-part cases turn up during the real build.
  - Same-round candidate patches are CHAIN-unioned (transitively, through
    a touching middle piece) before matching against history - not just
    pairwise-checked between the two outermost pieces. (CAL_008's outer
    two pieces are ~89m apart and would never have matched on a plain
    pairwise check; they touch a middle piece at 0m each.)
  - A new patch that bridges two already-separate tracked events is a
    MERGE. It is NOT silently combined into one record - both event IDs
    are kept, and a linkage note is added to each.

Coordinates in and out of this module are WGS84 (lon, lat) GeoJSON
geometries - the format GEE exports. All buffering/overlap math happens
internally in UTM zone 44N (EPSG:32644, the zone Sri Lanka sits in),
because a fixed-degree buffer is not a fixed-metre buffer once latitude
changes - exactly the kind of silent error this project's own SKILL.md
warns against ("don't let me plan around a nominal figure when a
measured one is obtainable").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Optional

import pyproj
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform

# --- coordinate handling ----------------------------------------------

_WGS84 = pyproj.CRS("EPSG:4326")
_UTM44N = pyproj.CRS("EPSG:32644")
_TO_UTM = pyproj.Transformer.from_crs(_WGS84, _UTM44N, always_xy=True).transform
_TO_WGS84 = pyproj.Transformer.from_crs(_UTM44N, _WGS84, always_xy=True).transform

BUFFER_M = 10.0  # decisions.md 2026-09-27 - measured, not assumed


def _to_utm(geom_wgs84):
    return transform(_TO_UTM, geom_wgs84)


def _to_wgs84(geom_utm):
    return transform(_TO_WGS84, geom_utm)


# --- data model ----------------------------------------------------------

@dataclass
class ClassificationState:
    """One classification call, at one point in time. The registry only
    ever appends these - it never edits or deletes a past one. This is
    what makes 'progressive, confidence-aware' (project.md) actually true
    in the data, not just in the write-up."""
    round_id: str
    date: Date
    stage: str          # "stage1" | "stage2" | "final"
    label: str            # e.g. "cultivation", "uncertain", ...
    confidence: str        # "high" / "medium" / "low" - labelling_protocol.md's rule
    evidence_note: str = ""


@dataclass
class Event:
    event_id: str
    footprint_wgs84: dict                       # GeoJSON geometry - union of every patch ever attached
    first_round: str
    first_date: Date
    last_round: str
    last_date: Date
    observation_rounds: list = field(default_factory=list)
    classification_history: list = field(default_factory=list)   # list[ClassificationState]
    linked_events: list = field(default_factory=list)             # merge case - see decisions.md
    status: str = "active"    # not enforced by this module; left for the caller to use

    def current_classification(self) -> Optional["ClassificationState"]:
        """Most recent classification state, or None if never classified.
        There is exactly one place classification state lives - this read,
        not a separately-maintained 'current label' field that could drift
        out of sync with the history."""
        return self.classification_history[-1] if self.classification_history else None

    def add_classification(self, state: "ClassificationState") -> None:
        self.classification_history.append(state)


# --- fragment union (within one round) ------------------------------------

def union_same_round_candidates(candidate_geoms_wgs84: list, buffer_m: float = BUFFER_M) -> list:
    """
    Chain-unions candidate polygons produced in the SAME round.

    Two candidates join if their buffered footprints touch/overlap
    directly, OR if each touches a shared third candidate (transitive).
    This is why: a single real clearing can come back from GEE's
    vectoriser as 2-3 separate pieces (issues.md B17) that don't all
    touch each other directly. CAL_008's two end pieces sit ~89m apart
    with no direct touch, but both touch a middle piece at 0m - checking
    only the two farthest pieces would have demanded an ~89m buffer;
    chaining through the actual touch-points needs only ~10m.

    Returns a list of GeoJSON geometries. A candidate with no neighbours
    is returned unchanged, as its own single-element group.
    """
    if not candidate_geoms_wgs84:
        return []

    shapes_utm = [_to_utm(shape(g)) for g in candidate_geoms_wgs84]
    buffered = [g.buffer(buffer_m) for g in shapes_utm]

    n = len(shapes_utm)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_idx(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if buffered[i].intersects(buffered[j]):
                union_idx(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        geoms = [shapes_utm[i] for i in members]
        merged_geom_utm = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        merged.append(mapping(_to_wgs84(merged_geom_utm)))

    return merged


# --- registry --------------------------------------------------------------

class EventRegistry:
    """Holds every tracked event and applies the D7 matching rule to each
    new round's candidates."""

    def __init__(self):
        self._events: dict = {}

    def events(self) -> dict:
        return dict(self._events)  # shallow copy - callers should not mutate directly

    def _existing_matches(self, candidate_wgs84: dict, buffer_m: float) -> list:
        cand_utm = _to_utm(shape(candidate_wgs84)).buffer(buffer_m)
        hits = []
        for eid, ev in self._events.items():
            ev_utm = _to_utm(shape(ev.footprint_wgs84)).buffer(buffer_m)
            if cand_utm.intersects(ev_utm):
                hits.append(eid)
        return hits

    def process_round(
        self,
        round_id: str,
        round_date: Date,
        raw_candidate_geoms_wgs84: list,
        buffer_m: float = BUFFER_M,
    ) -> dict:
        """
        Runs one round's candidates through matching.

        Returns {"new_events": [...], "updated_events": [...],
        "merges_logged": [(event_id, event_id), ...]} - the shape this
        needs to come back in for duplicate/fragmentation/merge to be
        reported as MEASURED RATES over an event set later (SKILL.md
        guardrail), not as one successful demonstration case.
        """
        summary = {"new_events": [], "updated_events": [], "merges_logged": []}

        unioned = union_same_round_candidates(raw_candidate_geoms_wgs84, buffer_m)

        for cand in unioned:
            matches = self._existing_matches(cand, buffer_m)

            if len(matches) == 0:
                eid = str(uuid.uuid4())
                self._events[eid] = Event(
                    event_id=eid,
                    footprint_wgs84=cand,
                    first_round=round_id,
                    first_date=round_date,
                    last_round=round_id,
                    last_date=round_date,
                    observation_rounds=[round_id],
                )
                summary["new_events"].append(eid)

            elif len(matches) == 1:
                eid = matches[0]
                ev = self._events[eid]
                merged_utm = unary_union([_to_utm(shape(ev.footprint_wgs84)), _to_utm(shape(cand))])
                ev.footprint_wgs84 = mapping(_to_wgs84(merged_utm))
                ev.last_round = round_id
                ev.last_date = round_date
                ev.observation_rounds.append(round_id)
                summary["updated_events"].append(eid)

            else:
                # Bridges >=2 previously separate events - MERGE case.
                # decisions.md 2026-09-27: do not silently combine. Keep
                # every event ID, log the linkage on each, and still
                # record this round's observation against all of them -
                # the physical footprint now genuinely touches all of
                # them, so pretending it doesn't would be the dishonest
                # choice.
                for eid in matches:
                    ev = self._events[eid]
                    merged_utm = unary_union([_to_utm(shape(ev.footprint_wgs84)), _to_utm(shape(cand))])
                    ev.footprint_wgs84 = mapping(_to_wgs84(merged_utm))
                    ev.last_round = round_id
                    ev.last_date = round_date
                    ev.observation_rounds.append(round_id)
                    for other in matches:
                        if other != eid and other not in ev.linked_events:
                            ev.linked_events.append(other)
                    summary["updated_events"].append(eid)
                for a in matches:
                    for b in matches:
                        if a < b:
                            summary["merges_logged"].append((a, b))

        return summary

    # --- classification interface ------------------------------------------

    def add_classification(self, event_id: str, state: "ClassificationState") -> None:
        """The classifier calls this. History is append-only. Raises
        KeyError if the event doesn't exist - deliberately: classifying
        an event that was never detected is a bug to surface, not
        silently swallow."""
        self._events[event_id].add_classification(state)
