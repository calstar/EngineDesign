"""Organize examples folder by moving old test/debug scripts to archive."""

import os
import shutil
from pathlib import Path

# Define source and archive directories
examples_dir = Path(__file__).parent
archive_dir = examples_dir / "archive"

# Create archive subdirectories
for subdir in ["debug", "test", "old_scripts"]:
    (archive_dir / subdir).mkdir(parents=True, exist_ok=True)

# Files to archive
debug_files = [
    "debug_cf_issue.py",
    "debug_feed_loss.py",
    "debug_regen_iteration.py",
    "check_cf_issue.py",
    "check_regen_impact.py",
    "diagnose_cf_and_graphite.py",
    "diagnose_cf_mach_turbulence.py",
]

test_files = [
    "test_3d_cea_cache.py",
    "test_ablative_Lstar_impact.py",
    "test_cooling_impact.py",
    "test_fixed_mr.py",
    "test_inverse_MR.py",
    "test_physics_validation.py",
    "test_pipeline.py",
    "test_root_finding.py",
    "test_time_varying_ablation.py",
]

old_scripts = [
    "fix_my_csv.py",
    "scale_pressure_csv.py",
    "analyze_regen_impact.py",
    "compare_ideal_vs_actual.py",
]

# Move files
moved = []
for file_list, target_dir in [
    (debug_files, "debug"),
    (test_files, "test"),
    (old_scripts, "old_scripts"),
]:
    for filename in file_list:
        src = examples_dir / filename
        dst = archive_dir / target_dir / filename
        if src.exists():
            shutil.move(str(src), str(dst))
            moved.append(f"{filename} -> archive/{target_dir}/")

print(f"Moved {len(moved)} files to archive:")
for item in moved:
    print(f"  {item}")

