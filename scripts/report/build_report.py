"""Write the published site (docs/*.html) from whichever section fragments exist.

Thin entry point around report.site, which holds the page list, the assembly
logic, and the hand-written content -- see that module for how fragments are
combined into pages.

Run with: python scripts/report/build_report.py (from the repo root, after
benchmark.py, scripts/report/build_correctness_report.py, and/or
scripts/report/build_bev_report.py have produced their fragments).
"""

import os

from report.site import REPORT_DIR, build_site


if __name__ == "__main__":
    pages = build_site()
    if not pages:
        raise SystemExit("No section fragments found under docs/ -- nothing to assemble. "
                         "Run benchmark.py and/or scripts/report/build_correctness_report.py first.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    for path, html in pages.items():
        with open(path, "w") as outfile:
            outfile.write(html)

    print(f"Wrote {len(pages)} page(s): {', '.join(pages)}")
