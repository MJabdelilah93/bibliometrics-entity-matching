"""queue.py — Human adjudication queue management.

Evidence boundary: the adjudication queue contains only pairs routed from the
guard layer. Human adjudicators make decisions from Scopus CSV field values;
no LLM suggestions are shown to the annotator.

TODOs:
- Write uncertain-band pairs to a structured adjudication queue file
  (parquet or CSV) under runs/<run_id>/outputs/.
- Each queue record must include: left_id, right_id, entity_type, similarity
  features, LLM label, confidence, and trace_ref (so the annotator can audit
  the LLM call if desired — without the label being pre-anchored).
- Provide a function to read back completed adjudication decisions and
  merge them with the auto-routed decisions.
- Decisions: {match, non-match} (no 'uncertain' allowed in adjudication output).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_adjudication_queue(
    uncertain_pairs: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write the adjudication queue for human review.

    Args:
        uncertain_pairs: Pairs routed to the adjudication queue by
            ``apply_guards.apply_guards``.
        output_dir: Directory under the current run to write the queue file.

    Returns:
        Path to the written queue file.

    TODO: implement.
    """
    raise NotImplementedError("write_adjudication_queue is not yet implemented.")


def read_adjudication_decisions(queue_path: Path) -> pd.DataFrame:
    """Read completed human adjudication decisions from a queue file.

    Args:
        queue_path: Path to the completed adjudication queue file.

    Returns:
        DataFrame with columns: ``left_id``, ``right_id``, ``decision``
        ∈ {match, non-match}.

    TODO: implement.
    """
    raise NotImplementedError("read_adjudication_decisions is not yet implemented.")
