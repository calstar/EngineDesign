"""Helper script to scale tank pressures in a CSV file."""

import pandas as pd
import sys
from pathlib import Path

def scale_csv_pressures(input_path: str, output_path: str, scale_factor: float = 2.5):
    """
    Scale tank pressures in a CSV file.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV (must have P_tank_O and P_tank_F columns)
    output_path : str
        Path to output CSV
    scale_factor : float
        Multiplicative factor for pressures (default 2.5x)
    """
    df = pd.read_csv(input_path)
    
    if "P_tank_O" not in df.columns or "P_tank_F" not in df.columns:
        raise ValueError("CSV must contain P_tank_O and P_tank_F columns")
    
    print(f"Original pressure ranges:")
    print(f"  LOX: {df['P_tank_O'].min():.1f} - {df['P_tank_O'].max():.1f} psi")
    print(f"  Fuel: {df['P_tank_F'].min():.1f} - {df['P_tank_F'].max():.1f} psi")
    
    df["P_tank_O"] = df["P_tank_O"] * scale_factor
    df["P_tank_F"] = df["P_tank_F"] * scale_factor
    
    print(f"\nScaled pressure ranges (factor = {scale_factor}):")
    print(f"  LOX: {df['P_tank_O'].min():.1f} - {df['P_tank_O'].max():.1f} psi")
    print(f"  Fuel: {df['P_tank_F'].min():.1f} - {df['P_tank_F'].max():.1f} psi")
    
    df.to_csv(output_path, index=False)
    print(f"\nScaled CSV saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scale_pressure_csv.py <input.csv> [output.csv] [scale_factor]")
        print("\nExample:")
        print('  python scale_pressure_csv.py "Untitled spreadsheet - Sheet1 (2).csv" scaled_pressures.csv 2.5')
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else input_csv.replace(".csv", "_scaled.csv")
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    
    scale_csv_pressures(input_csv, output_csv, scale)

