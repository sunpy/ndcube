"""
======================================================
Analyzing a dynamic spectrum with log-spaced frequency
======================================================

This example shows how to create and analyze an `~ndcube.NDCube` for a
synthetic solar radio dynamic spectrum, where the time axis has an irregular
cadence and the frequency axis is logarithmically spaced.

The example uses a broad metric-range frequency grid and an irregular cadence
to exercise non-uniform world coordinates without needing an external data
file.
"""

import numpy as np
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from matplotlib import pyplot as plt

import astropy.units as u
from astropy.modeling import models
from astropy.time import Time

from ndcube import NDCube


def build_dynspec_wcs(time_offsets_s, frequencies_hz):
    """
    Build a 2D gWCS for dynamic-spectrum data stored as (frequency, time).
    """
    time_model = models.Tabular1D(
        points=np.arange(len(time_offsets_s)),
        lookup_table=time_offsets_s,
        method='linear',
        bounds_error=False,
    )
    freq_model = models.Tabular1D(
        points=np.arange(len(frequencies_hz)),
        lookup_table=frequencies_hz,
        method='linear',
        bounds_error=False,
    )

    time_frame = cf.TemporalFrame(
        axes_order=(0,),
        unit=u.s,
        reference_frame=Time("2024-03-23T00:03:23"),
    )
    freq_frame = cf.SpectralFrame(axes_order=(1,), unit=u.Hz, axes_names=('frequency',))

    transform = time_model & freq_model
    output_frame = cf.CompositeFrame([time_frame, freq_frame])
    detector_frame = cf.CoordinateFrame(
        name="detector",
        naxes=2,
        axes_order=(0, 1),
        axes_type=("pixel", "pixel"),
        unit=(u.pix, u.pix),
    )

    dynspec_wcs = gwcs_wcs.WCS(
        forward_transform=transform,
        output_frame=output_frame,
        input_frame=detector_frame,
    )
    dynspec_wcs.array_shape = (len(frequencies_hz), len(time_offsets_s))
    return dynspec_wcs

##############################################################################
# Build synthetic dynamic spectrum data
# -------------------------------------
# The frequency axis is logarithmically spaced between ~4 and ~978 MHz.
# The time axis has an irregular cadence drawn from a normal distribution
# centred on ~14 s.
# We inject a simulated Type III solar radio burst, a narrowband feature that
# drifts from high to low frequency over time, on top of a background noise
# floor.
#
# We store the data as shape ``(n_freq, n_time)`` so that frequency varies along
# rows (the Y axis) and time along columns (the X axis), matching the standard
# radio-astronomy display convention.

rng = np.random.default_rng(42)
n_freq, n_time = 64, 100

# Log-spaced frequencies for this synthetic metric-range example.
freqs_hz = np.logspace(np.log10(3.992e6), np.log10(978.572e6), n_freq)

# Irregular time offsets (seconds since observation start, median ~14 s cadence)
dt_s = np.cumsum(np.abs(rng.normal(14, 3, n_time)))
dt_s -= dt_s[0]

# Background flux density (exponential noise floor), shape (n_freq, n_time)
data = rng.exponential(1e-15, (n_freq, n_time))

# Inject a Type III burst: emission that drifts from ~400 MHz to ~50 MHz
drift_rate_hz_per_s = -5e6
burst_start_s = dt_s[35]
for t_idx in range(n_time):
    f_center = 400e6 + drift_rate_hz_per_s * (dt_s[t_idx] - burst_start_s)
    if freqs_hz[0] < f_center < freqs_hz[-1]:
        f_idx = int(np.argmin(np.abs(freqs_hz - f_center)))
        half_width = max(1, n_freq // 15)
        lo, hi = max(0, f_idx - half_width), min(n_freq, f_idx + half_width)
        data[lo:hi, t_idx] *= 30

##############################################################################
# Build the gWCS using ``Tabular1D`` lookup-table transforms
# ----------------------------------------------------------
# Because neither axis is uniformly spaced in physical coordinates, we use
# `~astropy.modeling.models.Tabular1D` to map pixel indices to world values.
#
# Axis ordering follows the ndcube/FITS convention: pixel axis 0 maps to the
# *last* numpy array axis, so for shape ``(n_freq, n_time)``:
#
# * **pixel axis 0** -> time (array axis 1, X axis when plotted)
# * **pixel axis 1** -> frequency (array axis 0, Y axis when plotted)

dynspec_wcs = build_dynspec_wcs(dt_s, freqs_hz)

##############################################################################
# Create the NDCube
# -----------------

dynspec_cube = NDCube(data, wcs=dynspec_wcs, unit=u.W / u.m**2 / u.Hz)
print(dynspec_cube)

##############################################################################
# The ``array_axis_physical_types`` property confirms the mapping: array axis 0
# (rows, Y) carries frequency and array axis 1 (columns, X) carries time.

print(dynspec_cube.array_axis_physical_types)

##############################################################################
# Plot the full dynamic spectrum
# ------------------------------
# Time appears on the X axis and frequency on the Y axis.

dynspec_cube.plot()
plt.gca().set_title("Synthetic dynamic spectrum")

##############################################################################
# Crop a useful time-frequency window
# -----------------------------------
# ``crop_by_values`` accepts world-coordinate `~astropy.units.Quantity` objects
# in world-axis order, which is ``(time, frequency)`` for this WCS.
# Pass ``None`` to leave an axis unconstrained.
# The method returns the smallest pixel bounding box that spans the requested
# range, so the outermost channels may lie just outside the nominal bounds.

windowed = dynspec_cube.crop_by_values(
    [200 * u.s, 10e6 * u.Hz],
    [800 * u.s, 500e6 * u.Hz],
)
print("Windowed shape (200-800 s, 10-500 MHz):", windowed.shape)

windowed.plot()
plt.gca().set_title("Windowed: 200-800 s, 10-500 MHz")

##############################################################################
# Rebin frequency channels
# ------------------------
# ``rebin`` bins contiguous pixels together.  The bin shape follows numpy
# (array) axis ordering, so ``(3, 1)`` averages triplets of frequency rows while
# leaving time samples unchanged.

rebinned = windowed.rebin((3, 1))
print("Frequency-rebinned shape:", rebinned.shape)

rebinned.plot()
plt.gca().set_title("Frequency rebinned")

##############################################################################
# Resample the time axis to a denser grid
# ---------------------------------------
# Here we use linear interpolation to add one new time sample between each pair
# of original time samples in the cropped cube.  The data shape changes from
# ``(n_freq, n_time)`` to ``(n_freq, 2 * n_time - 1)`` and we build a matching
# WCS from the new lookup table.

old_times_s = np.array([
    windowed.wcs.low_level_wcs.pixel_to_world_values(time_idx, 0)[0]
    for time_idx in range(windowed.shape[1])
])
windowed_freqs_hz = np.array([
    windowed.wcs.low_level_wcs.pixel_to_world_values(0, freq_idx)[1]
    for freq_idx in range(windowed.shape[0])
])
new_times_s = np.linspace(old_times_s[0], old_times_s[-1], 2 * len(old_times_s) - 1)
resampled_data = np.array([
    np.interp(new_times_s, old_times_s, frequency_row)
    for frequency_row in windowed.data
])

resampled_wcs = build_dynspec_wcs(new_times_s, windowed_freqs_hz)
resampled = NDCube(resampled_data, wcs=resampled_wcs, unit=windowed.unit)
print("Time-resampled shape:", resampled.shape)

resampled.plot()
plt.gca().set_title("Time resampled")

plt.show()
