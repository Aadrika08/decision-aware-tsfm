"""Export a pinned, synthetic producer-shaped handoff for PR #44/#35 consumers."""

import argparse
import json
from pathlib import Path

from decision_aware_tsfm.cache.handoff_fixture import export_handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-source", required=True, type=Path,
                        help="clean checkout at pinned Veer PR #44 commit")
    parser.add_argument("--output", required=True, type=Path,
                        help="new output directory")
    args = parser.parse_args()
    manifest = export_handoff(args.decision_source, args.output)
    print(json.dumps({"output": str(args.output), "partitions": len(manifest["partition_hashes"]),
                      "source_revision": manifest["source_revision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
