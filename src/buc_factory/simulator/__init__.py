"""Candidate simulation package — demo tool, peer of agent/, easily removable.

Simulates a candidate of a given proficiency level completing an assessment
starter project (PBIP or Jupyter Notebook).

Decommissioning: remove this package and the /simulations endpoints in app/api.py.
The core assessment generator (agent/) and scorer (scorer/) are unaffected.
"""

from .candidate_graph import CandidateState, build_candidate_graph

__all__ = ["CandidateState", "build_candidate_graph"]
