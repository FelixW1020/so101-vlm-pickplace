"""Success-rate bookkeeping over held-out placements.

This exists because LeRobot does not provide it. `lerobot-eval` computes
pc_success from a *simulator's* reward signal and has no real-robot path;
`lerobot-rollout --strategy.type=episodic` records episodes but writes no
success/failure column (it logs timing only). So the headline number for this
project -- success rate over held-out placements -- has to be tallied here.

Splitting the outcome into stages is what makes the evaluation say something.
"37% success" is not a finding; "grounding was right 95% of the time and the
grasp failed on 40% of those" tells you where to spend the next week.

Usage:
    python -m so101_pickplace.evaluation.success_log log --run held_out_v1
    python -m so101_pickplace.evaluation.success_log report --run held_out_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class Outcome(StrEnum):
    SUCCESS = "success"
    GROUNDING_FAILED = "grounding_failed"
    """Wrong object identified, or abstained when the object was present."""
    APPROACH_FAILED = "approach_failed"
    """Grounded correctly but the arm went to the wrong place -- calibration."""
    GRASP_FAILED = "grasp_failed"
    """Right place, gripper closed on nothing or dropped it."""
    PLACE_FAILED = "place_failed"
    """Grasped it, missed the container."""


@dataclass
class Trial:
    episode_index: int
    phrase: str
    outcome: Outcome
    placement: str = ""
    """Held-out placement id, e.g. "grid_c4". Lets you find systematic dead zones."""
    grounding_conf: float | None = None
    grounding_source: str = ""
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SuccessLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.trials: list[Trial] = []
        if self.path.exists():
            self.load()

    def load(self) -> None:
        raw = json.loads(self.path.read_text())
        self.trials = [Trial(**{**t, "outcome": Outcome(t["outcome"])}) for t in raw]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(t) for t in self.trials], indent=2))

    def add(self, trial: Trial) -> None:
        self.trials.append(trial)
        self.save()

    def report(self) -> dict:
        n = len(self.trials)
        if n == 0:
            return {"n": 0}

        by_outcome: dict[str, int] = {}
        for t in self.trials:
            by_outcome[t.outcome] = by_outcome.get(t.outcome, 0) + 1

        n_success = by_outcome.get(Outcome.SUCCESS, 0)
        n_grounded = n - by_outcome.get(Outcome.GROUNDING_FAILED, 0)

        by_placement: dict[str, dict[str, int]] = {}
        for t in self.trials:
            if not t.placement:
                continue
            slot = by_placement.setdefault(t.placement, {"n": 0, "success": 0})
            slot["n"] += 1
            slot["success"] += int(t.outcome == Outcome.SUCCESS)

        return {
            "n": n,
            "success_rate": n_success / n,
            "grounding_accuracy": n_grounded / n,
            # Isolates stage 2: of the episodes stage 1 got right, how many
            # did the policy finish?
            "policy_success_given_grounding": (n_success / n_grounded) if n_grounded else 0.0,
            "by_outcome": dict(sorted(by_outcome.items())),
            "by_placement": by_placement,
        }


def _log_interactive(log: SuccessLog) -> None:
    options = list(Outcome)
    print("outcomes:")
    for i, o in enumerate(options):
        print(f"  {i}) {o}")
    print("blank episode index to finish\n")

    next_index = max((t.episode_index for t in log.trials), default=-1) + 1
    while True:
        raw = input(f"episode [{next_index}]: ").strip()
        if raw == "" and not input("finish? [y/N] ").lower().startswith("y"):
            continue
        if raw == "":
            break
        try:
            idx = int(raw)
            phrase = input("  phrase: ").strip()
            placement = input("  placement id: ").strip()
            choice = int(input(f"  outcome [0-{len(options) - 1}]: ").strip())
            log.add(Trial(
                episode_index=idx, phrase=phrase,
                outcome=options[choice], placement=placement,
            ))
            next_index = idx + 1
            print(f"  logged ({len(log.trials)} total)\n")
        except (ValueError, IndexError):
            print("  bad input, try again\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Success-rate tally for real-robot eval")
    ap.add_argument("command", choices=["log", "report"])
    ap.add_argument("--run", default="default", help="run name -- one file per eval sweep")
    ap.add_argument("--dir", type=Path, default=Path("outputs/eval"))
    args = ap.parse_args()

    log = SuccessLog(args.dir / f"{args.run}.json")

    if args.command == "log":
        _log_interactive(log)

    report = log.report()
    if report["n"] == 0:
        print("no trials logged yet")
        return

    print(f"\n=== {args.run} ===")
    print(f"n                              {report['n']}")
    print(f"success rate                   {report['success_rate']:.1%}")
    print(f"grounding accuracy             {report['grounding_accuracy']:.1%}")
    print(f"policy success | grounded ok   {report['policy_success_given_grounding']:.1%}")
    print("\nfailure breakdown:")
    for outcome, count in report["by_outcome"].items():
        print(f"  {outcome:<20} {count:>3}  ({count / report['n']:.0%})")
    if report["by_placement"]:
        print("\nby placement:")
        for name, s in sorted(report["by_placement"].items()):
            print(f"  {name:<12} {s['success']}/{s['n']}")


if __name__ == "__main__":
    main()
