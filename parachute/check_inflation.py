import pandas as pd

df = pd.read_csv('out/realistic_test/telemetry.csv')
print("Canopy area after pickup:")
after = df[df['t'] > 0.18]
print(after[['t', 'canopy:drogue:A', 'canopy:drogue:CD']].head(15))
print(f'\nMax area: {after["canopy:drogue:A"].max():.3f} m²')
print(f'Max CD: {after["canopy:drogue:CD"].max():.3f}')

