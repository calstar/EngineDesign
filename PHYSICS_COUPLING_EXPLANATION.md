# Physics Coupling: Why Oxidizer Tank Pressure Affects Fuel Flow

## The Coupling Mechanism

The oxidizer and fuel mass flow rates are **coupled through the chamber pressure balance**. This is correct physics and essential for accurate modeling.

## Flow Path

1. **Tank Pressures** (INPUT):
   - P_tank_O (oxidizer)
   - P_tank_F (fuel)

2. **Feed System Losses**:
   - Δp_feed_O = f(mdot_O, P_tank_O)
   - Δp_feed_F = f(mdot_F, P_tank_F) + Δp_regen (if enabled)
   - P_injector_O = P_tank_O - Δp_feed_O
   - P_injector_F = P_tank_F - Δp_feed_F

3. **Injector Flow** (depends on chamber pressure):
   - mdot_O = Cd_O × A_LOX × √(2ρ_O × (P_injector_O - Pc))
   - mdot_F = Cd_F × A_fuel × √(2ρ_F × (P_injector_F - Pc))

4. **Chamber Pressure Balance** (SOLVED):
   - Supply: mdot_supply = mdot_O + mdot_F
   - Demand: mdot_demand = Pc × At / c*_actual
   - Equilibrium: mdot_supply = mdot_demand

## Why Coupling Occurs

When you change **P_tank_O**:

1. **Direct effect**: mdot_O changes (higher P_tank_O → higher mdot_O)

2. **Indirect effect on fuel**:
   - Higher mdot_O → higher mdot_supply
   - The solver adjusts Pc to balance supply = demand
   - New Pc affects **both** mdot_O and mdot_F
   - Since mdot_F = Cd_F × A_fuel × √(2ρ_F × (P_injector_F - Pc))
   - Changing Pc changes mdot_F

3. **Result**: 
   - Higher P_tank_O → higher Pc → affects mdot_F
   - This is **correct physics** - the flows are coupled through the chamber

## Example

**Scenario**: Increase P_tank_O from 1000 to 1200 psi

**What happens**:
1. mdot_O increases (direct effect)
2. mdot_supply increases
3. Solver increases Pc to balance supply = demand
4. Higher Pc reduces (P_injector_F - Pc) for fuel
5. mdot_F decreases (indirect effect)
6. O/F ratio increases

**This is physically correct!** The chamber pressure couples the two flows.

## Mathematical Formulation

The system solves:
```
mdot_O(P_tank_O, Pc) + mdot_F(P_tank_F, Pc) = Pc × At / c*_actual(MR, Pc)
```

where:
- MR = mdot_O / mdot_F (mixture ratio)
- c*_actual depends on MR and Pc
- This is a **coupled nonlinear system**

The coupling is through:
1. **Pc** (chamber pressure) - affects both flows
2. **MR** (mixture ratio) - affects c*_actual, which affects demand

## Conclusion

The coupling is **correct physics**. It's not a bug - it's how real engines work. The chamber pressure balances the supply and demand, creating coupling between the two propellant flows.

