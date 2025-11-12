"""Debug why canopies aren't inflating"""
import pandas as pd

df = pd.read_csv('out/realistic_test/telemetry.csv')
events = pd.read_csv('out/realistic_test/events.csv')

print("="*60)
print("INFLATION DEBUG")
print("="*60)

# Find drogue pickup time
drogue_pickup = events[events['id'] == 'leg_drogue']['t_event'].iloc[0] if len(events[events['id'] == 'leg_drogue']) > 0 else None
print(f"\nDrogue pickup time: {drogue_pickup:.6f}s")

# Check canopy area after pickup
if drogue_pickup:
    after_pickup = df[df['t'] >= drogue_pickup].iloc[0:10]
    print(f"\nAfter pickup (first 10 timesteps):")
    print(after_pickup[['t', 'canopy:drogue:A', 'canopy:drogue:CD', 'canopy:drogue:p_z']])
    
    # Check conditions
    for idx, row in after_pickup.iterrows():
        t = row['t']
        altitude = row['canopy:drogue:p_z']
        deploy_alt = 3048.0
        t_pickup = drogue_pickup
        
        print(f"\nt={t:.3f}s:")
        print(f"  Altitude: {altitude:.1f}m (deploy: {deploy_alt:.1f}m)")
        print(f"  Altitude > deploy: {altitude > deploy_alt} (should be False to allow inflation)")
        print(f"  t >= t_pickup: {t >= t_pickup} (should be True to allow inflation)")
        print(f"  Area: {row['canopy:drogue:A']:.6f} m²")
        if altitude > deploy_alt:
            print(f"  ❌ BLOCKED: Altitude check failed")
        elif t < t_pickup:
            print(f"  ❌ BLOCKED: Pickup check failed")
        else:
            print(f"  ✓ Should inflate, but area is still 0 - check other conditions")

