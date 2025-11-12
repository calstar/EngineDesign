"""Check edge_to_canopy mapping"""
import yaml
from pathlib import Path

with open('examples/realistic_rocket_deployment.yaml') as f:
    config = yaml.safe_load(f)

print("Edges:")
for edge in config['edges']:
    edge_id = edge['id']
    n_minus = edge['n_minus']
    n_plus = edge['n_plus']
    print(f"  {edge_id}:")
    print(f"    n_minus: {n_minus}")
    print(f"    n_plus: {n_plus}")
    
    # Check which end is a canopy
    canopies = list(config['canopies'].keys())
    if n_minus in canopies:
        print(f"    -> Canopy on n_minus: {n_minus}")
    elif n_plus in canopies:
        print(f"    -> Canopy on n_plus: {n_plus}")
    else:
        print(f"    -> No canopy on either end")
    
    # For leg_drogue specifically
    if edge_id == 'leg_drogue':
        print(f"\n  leg_drogue mapping:")
        print(f"    n_minus = {n_minus} (should be 'drogue')")
        print(f"    n_plus = {n_plus} (should be 'avionics:pilot_attach')")
        print(f"    'drogue' in canopies: {'drogue' in canopies}")

