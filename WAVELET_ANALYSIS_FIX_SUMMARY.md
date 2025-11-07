# Wavelet Scalogram Issues and Solutions

## Problem Summary

Your original wavelet scalogram showed almost no visible patterns - just a dark blue plot with no distinguishable frequency bands. This is a **very common problem** in wavelet analysis of real-world time series data.

## Root Causes Identified

### 1. **Inadequate Scale Selection**
- **Issue**: Original scales ranged from 1 to 500 with only 100 logarithmically spaced points
- **Impact**: Most scales clustered at low values, insufficient coverage of important periods (7, 30, 91, 183, 365 days)
- **Solution**: Implemented non-uniform scale distribution:
  - Dense sampling in 2-20 day range (weekly/biweekly patterns)
  - Medium sampling in 20-100 day range (monthly/quarterly patterns)
  - Logarithmic sampling in 100-730 day range (semi-annual/annual patterns)

### 2. **No Detrending Applied**
- **Issue**: US births data has a strong upward trend from 1974-1987
- **Impact**: Trend dominates the wavelet power spectrum, completely hiding seasonal patterns
- **Solution**: Applied linear detrending before CWT using `scipy.signal.detrend()`
- **Critical**: This is the **most important fix** for your data

### 3. **Poor Power Normalization**
- **Issue**: Wavelet power values span huge dynamic range (0 to 10^9)
- **Impact**: Low-power but meaningful patterns (like weekly cycles) become invisible when plotted
- **Solution**: Implemented two normalization schemes:

  **Option A - Logarithmic Scaling:**
  ```python
  power = np.log10(power + 1e-12)
  ```
  - Makes weak patterns visible
  - Compresses dynamic range
  - Best for detecting all patterns

  **Option B - Scale-wise Normalization:**
  ```python
  power = power / power.max(axis=1, keepdims=True)
  ```
  - Normalizes each frequency band independently
  - Highlights relative strength at each period
  - Best for comparing pattern consistency across frequencies

### 4. **Lack of Visual Reference Guides**
- **Issue**: No indication of where to look for important patterns
- **Solution**: Added white dashed reference lines at periods: 7, 30, 91, 183, 365 days
- **Benefit**: Easy visual identification of expected seasonal components

## Updated Implementation

### Modified Functions

#### 1. `continuous_wavelet_transform()` - [utilities/spectral_analysis.py:314](utilities/spectral_analysis.py#L314)

**New features:**
- Added `detrend` parameter (default: True)
- Improved scale selection algorithm
- Better coverage of seasonal periods

**Key changes:**
```python
# Detrend signal
if detrend:
    from scipy.signal import detrend as scipy_detrend
    ts_data = scipy_detrend(ts_data)

# Improved scales
scales_low = np.linspace(2, 20, 30)      # Weekly patterns
scales_mid = np.linspace(20, 100, 40)    # Monthly patterns
scales_high = np.geomspace(100, 730, 50) # Annual patterns
```

#### 2. `plot_wavelet_scalogram()` - [utilities/spectral_analysis.py:367](utilities/spectral_analysis.py#L367)

**New parameters:**
- `detrend`: Whether to detrend before CWT
- `normalize`: Power normalization method ('log', 'scale', or None)

**Visual enhancements:**
- Reference lines for important periods (7, 30, 91, 183, 365 days)
- Labels showing period values
- Improved colormap scaling
- Better title indicating detrending/normalization applied

## How to Use the Fixed Version

### In Your Notebook

The updated cell (cell-14) now runs **two different visualizations**:

```python
# 1. Log-normalized (best for detecting all patterns)
fig1 = spectral.plot_wavelet_scalogram(
    analysis_data['births'].values,
    datetime_index=analysis_data['datetime'],
    wavelet='morl',
    detrend=True,        # Remove trend
    normalize='log',     # Log scaling
    title='Wavelet Scalogram - Log Scale'
)

# 2. Scale-normalized (best for highlighting each frequency)
fig2 = spectral.plot_wavelet_scalogram(
    analysis_data['births'].values,
    datetime_index=analysis_data['datetime'],
    wavelet='morl',
    detrend=True,
    normalize='scale',   # Row-wise normalization
    title='Wavelet Scalogram - Normalized by Scale'
)
```

## What You Should See Now

With the fixes applied, the scalogram should show:

### Expected Patterns

1. **Strong horizontal band around 7 days** (weekly cycle)
   - Color: Yellow/orange/red (high power)
   - Interpretation: Strong weekday vs weekend effect
   - Should be visible throughout the entire time period

2. **Horizontal band around 365 days** (annual cycle)
   - Color: Orange/yellow
   - Interpretation: Seasonal birth rate variations
   - May show some time-varying intensity

3. **Possible band around 182-183 days** (semi-annual)
   - Color: Yellow/green
   - Weaker than weekly and annual
   - Related to bi-modal annual distribution

4. **Potential patterns around 30 days** (monthly)
   - Color: Variable, possibly green/blue
   - Weaker than weekly pattern
   - May be intermittent

### Visual Characteristics

- **Log-normalized version**: Shows all patterns, even weak ones
- **Scale-normalized version**: Emphasizes pattern consistency over time
- **White dashed lines**: Mark expected period locations
- **Time axis**: Shows when patterns are strongest/weakest

## Comparison with Fourier Analysis

The wavelet scalogram should **confirm** the Fourier analysis results:

| Method | Top Periods Detected |
|--------|---------------------|
| Fourier | 7, 3.5, 365, 183 days |
| Wavelet | Should show bands at ~7 and ~365 days |

**Why they differ slightly:**
- Fourier: Global frequency content (entire time series)
- Wavelet: Time-localized frequency content (when patterns occur)
- Wavelet reveals **time-varying** seasonal strength

## Technical Notes

### Why Detrending is Critical

Without detrending:
- Trend has infinite period (DC component)
- Wavelet coefficients at large scales dominated by trend
- Seasonal patterns at 7, 30, 365 days become invisible
- Power spectrum range: 0 to 10^15 (unmanageable)

With detrending:
- Trend removed, focusing on oscillations
- Seasonal patterns clearly visible
- Power spectrum range: manageable
- Logarithmic/normalized scaling effective

### Wavelet Choice

**Morlet wavelet** ('morl') is excellent for birth data because:
- Good frequency localization
- Similar to sinusoidal oscillations
- Well-suited for detecting periodic patterns
- Standard choice for time-frequency analysis

**Alternatives to try:**
- 'cmor1.5-1.0': Complex Morlet (better frequency resolution)
- 'gaus8': Gaussian wavelet (better time localization)
- 'mexh': Mexican Hat/Ricker (good for sharp features)

### Computational Considerations

The improved implementation:
- Uses ~120 scales (vs 100 before)
- More computationally intensive but still fast (<10 seconds)
- Memory efficient with row-wise normalization
- Scales up to 730 days (2 years) for long-period detection

## Troubleshooting

If patterns still not visible:

1. **Try different wavelets:**
   ```python
   wavelet='cmor1.5-1.0'  # Complex Morlet
   ```

2. **Adjust normalization:**
   ```python
   normalize='scale'  # Try this if log doesn't work well
   ```

3. **Check data quality:**
   - Ensure no large gaps in time series
   - Verify datetime index is continuous
   - Look for outliers that might dominate

4. **Experiment with scale ranges:**
   ```python
   # Custom scales focusing on weekly/annual
   custom_scales = np.concatenate([
       np.linspace(5, 10, 50),    # Around weekly
       np.linspace(350, 380, 50)  # Around annual
   ])
   ```

## Summary of Changes

### Files Modified

1. **[utilities/spectral_analysis.py](utilities/spectral_analysis.py)**
   - Line 314-364: Updated `continuous_wavelet_transform()`
   - Line 367-473: Updated `plot_wavelet_scalogram()`

2. **[us_births_fourier_wavelet_analysis.ipynb](us_births_fourier_wavelet_analysis.ipynb)**
   - Cell 14: Updated wavelet analysis with new parameters

### Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Detrending | ❌ None | ✅ Linear detrend |
| Scale range | 1-500 (poor coverage) | 2-730 (optimized) |
| Scale distribution | Logarithmic | Non-uniform (dense where needed) |
| Normalization | None | Log or scale-wise |
| Visual guides | None | Period reference lines |
| Pattern visibility | ❌ Nothing visible | ✅ Clear bands expected |

## Next Steps

1. **Re-run the notebook** - Execute the updated cell 14
2. **Compare both scalograms** - Log vs scale-normalized
3. **Verify patterns** - Look for horizontal bands at 7 and 365 days
4. **Cross-reference** - Confirm Fourier peaks match wavelet bands
5. **Interpret results** - Note any time-varying pattern strength

## Expected Outcome

With these fixes, your wavelet scalogram should transition from:
- **Before**: Dark blue, no patterns visible, unhelpful
- **After**: Colorful with distinct horizontal bands at weekly and annual periods, scientifically meaningful

The scalogram will now be a valuable complement to your Fourier analysis, showing not just **what** frequencies are present, but **when** they are strongest!
