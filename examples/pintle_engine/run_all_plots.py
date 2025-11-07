"""Run all plotting scripts to regenerate all plots"""

import subprocess
import sys
from pathlib import Path

scripts = [
    "comprehensive_performance_plots.py",
    "chamber_3d_plots.py",
    "compare_ideal_vs_actual.py",
    "validate_chamber_intrinsics.py",
    "pressure_sweep_example.py",
]

base_dir = Path(__file__).parent

print("=" * 80)
print("RUNNING ALL PLOTTING SCRIPTS")
print("=" * 80)

for script in scripts:
    script_path = base_dir / script
    if not script_path.exists():
        print(f"[SKIP] {script} - not found")
        continue
    
    print(f"\n{'='*80}")
    print(f"Running: {script}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[OK] {script} completed successfully")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
        else:
            print(f"[ERROR] {script} failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {script} took too long (>10 minutes)")
    except Exception as e:
        print(f"[ERROR] {script} failed: {e}")

print(f"\n{'='*80}")
print("[OK] ALL PLOTS COMPLETE")
print(f"{'='*80}")

