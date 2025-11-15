# UI Integration: Ablative Geometry Evolution

## ✅ Complete - All Features Integrated into Streamlit UI

### 1. Automatic Detection & Method Selection

The UI now **automatically detects** when ablative geometry tracking is enabled and uses the appropriate evaluation method:

```python
# In compute_timeseries_dataframe()
if ablative_tracking_enabled and len(times) >= 2:
    results = runner.evaluate_arrays_with_time(times, P_O, P_F)  # Time-varying
else:
    results = runner.evaluate_arrays(P_O, P_F)  # Standard
```

**When ablative tracking activates:**
- `ablative_cooling.enabled = true`
- `ablative_cooling.track_geometry_evolution = true`
- Time-series analysis is selected (not single point)

### 2. New Ablative Configuration Controls

Added comprehensive UI controls in the **Ablative Cooling** section:

**Geometry Evolution:**
- ✅ **Enable geometry evolution tracking** (checkbox)
  - Tracks cumulative recession
  - Updates chamber/throat geometry over time
  
- ✅ **Throat recession multiplier** (physics-based or fixed)
  - Physics-based (default): Calculated from Bartz correlation
  - Fixed: User specifies 1.0-3.0x multiplier
  
- ✅ **Char layer properties**
  - Conductivity [W/(m·K)]
  - Thickness [m]

### 3. New Time-Series Plots

When ablative tracking is enabled, 4 new plots appear:

**Plot 1: L* Evolution**
- Shows characteristic length changing over time
- Demonstrates efficiency degradation

**Plot 2: Cumulative Recession**
- Chamber recession (purple line)
- Throat recession (orange line)
- Shows throat recedes ~1.2-2.0x faster

**Plot 3: Geometry Growth**
- Chamber volume change (%)
- Throat area change (%)
- Shows how geometry expands

**Plot 4: Throat Recession Multiplier**
- Physics-based multiplier vs time
- Mean value displayed
- Only shown if physics-based mode is enabled

### 4. Performance Impact Summary

After plots, an info box displays:

```
Ablative Geometry Impact:
- L* increased by +X.XX% (initial → final mm)
- Throat area grew by +X.XXX%
- Thrust degraded by +X.XX% (initial → final kN)
- Total chamber recession: X.X µm
- Total throat recession: X.X µm
```

### 5. New Dataframe Columns

Time-series CSV exports now include:

| Column | Unit | Description |
|--------|------|-------------|
| `L* (mm)` | mm | Time-varying characteristic length |
| `Chamber Volume (cm³)` | cm³ | Evolving chamber volume |
| `Throat Area (mm²)` | mm² | Growing throat area |
| `Cumulative Chamber Recession (µm)` | µm | Total chamber recession |
| `Cumulative Throat Recession (µm)` | µm | Total throat recession |
| `Throat Recession Multiplier` | - | Physics-based multiplier |

---

## 🔧 FIXED: L* Override Bug

### The Problem

Previously, the UI would **always** save `Lstar` to the config, which would override `volume` and `A_throat` values. This made it impossible to specify chamber geometry by volume and throat area.

### The Solution

Added a **mode selector** for chamber geometry specification:

**Mode 1: Volume + Throat Area**
- User specifies: `volume`, `A_throat`
- L* is **calculated** and displayed (info box)
- Config stores: `volume`, `A_throat`, `Lstar = None`
- Solver calculates L* from V/A (no override)

**Mode 2: L* (Characteristic Length)**
- User specifies: `Lstar`, `A_throat`
- Volume is **calculated** from `L* × A_throat`
- Config stores: all three values
- Solver uses specified L*

### UI Changes

```python
# Before (buggy):
chamber["volume"] = st.number_input(...)
chamber["A_throat"] = st.number_input(...)
chamber["Lstar"] = st.number_input(...)  # ❌ Always overrides!

# After (fixed):
geom_mode = st.radio(["Volume + Throat Area", "L* (Characteristic Length)"])

if geom_mode == "Volume + Throat Area":
    chamber["volume"] = st.number_input(...)
    chamber["A_throat"] = st.number_input(...)
    chamber["Lstar"] = None  # ✅ No override!
    st.info(f"Calculated L* = {chamber['volume']/chamber['A_throat']:.4f} m")
else:
    chamber["A_throat"] = st.number_input(...)
    chamber["Lstar"] = st.number_input(...)
    chamber["volume"] = chamber["Lstar"] * chamber["A_throat"]  # Sync volume
    st.info(f"Calculated Volume = {chamber['volume']:.6f} m³")
```

---

## Usage Example

### Enable Ablative Tracking in UI

1. Go to **Configuration Editor**
2. Expand **Ablative Cooling**
3. Check **Enable ablative cooling**
4. Check **Enable geometry evolution tracking** ✅
5. Choose throat multiplier mode:
   - ✅ **Physics-based** (recommended) - Bartz correlation
   - Fixed - Specify 1.2-2.0x manually

### Run Time-Series Analysis

1. Go to **Time-Series Evaluation**
2. Generate or upload pressure profile
3. Click **Run Time-Series**
4. Scroll down to see:
   - Standard plots (thrust, Pc, mdot, etc.)
   - **🔥 Ablative Geometry Evolution** section with 4 new plots
   - Performance degradation summary

### Export Data

- Click **Download CSV** to get full dataframe including ablative columns
- Use custom plot builder to create specific views of recession/L*/geometry

---

## Validation

Test script confirms integration:

```bash
python examples/pintle_engine/test_time_varying_ablation.py
```

**Expected output:**
- ✅ Simulation completes successfully
- ✅ L* evolves over time
- ✅ Cumulative recession tracked
- ✅ Throat multiplier ~1.2-2.0x
- ✅ Performance degradation calculated
- ✅ Plots saved

---

## Files Modified

**UI Integration:**
- `examples/pintle_engine/ui_app.py`
  - `compute_timeseries_dataframe()` - Auto-detect ablative tracking
  - `plot_time_series_results()` - Add 4 new plots
  - `config_editor()` - Add geometry mode selector + ablative controls

**Backend (Already Complete):**
- `pintle_models/runner.py` - `evaluate_arrays_with_time()` method
- `pintle_pipeline/ablative_geometry.py` - Physics module
- `pintle_pipeline/config_schemas.py` - New config parameters

**Tests:**
- `examples/pintle_engine/test_time_varying_ablation.py` - Full integration test

**Documentation:**
- `examples/pintle_engine/ABLATIVE_GEOMETRY_IMPLEMENTATION.md`
- `examples/pintle_engine/QUICKSTART_ABLATIVE.md`
- `examples/pintle_engine/COOLING_THRUST_COUPLING.md`

---

## Summary

✅ **Ablative geometry evolution fully integrated into UI**
✅ **Automatic detection and method selection**
✅ **4 new plots for geometry tracking**
✅ **Performance degradation summary**
✅ **New config controls (throat multiplier, char layer, tracking toggle)**
✅ **L* override bug fixed with mode selector**
✅ **Complete test coverage**

**The UI now provides a complete, production-ready interface for ablative engine design with time-varying geometry tracking!** 🚀🔥

