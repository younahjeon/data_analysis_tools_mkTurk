"""
Waveform Characterization by Depth
====================================
Computes and plots waveform features (spike duration, peak/trough ratio,
spike height) as a function of recording depth, aligned to trial-averaged
raw LFP and z-scored PSTH across all channels on the Neuropixels probe.
Inspired by Zhang Li A, Li Peichao, Callaway Edward M (2024) 
High-Resolution Laminar Identification in Macaque Primary Visual Cortex Using Neuropixels Probes 
eLife 13:RP97290 https://doi.org/10.7554/eLife.97290.2

For each recording session directory the script:
  1. Reads the channel map and probe metadata.
  2. Loads the pre-processed LFP and computes a trial-averaged, CAR-corrected
     response locked to stimulus onset.
  3. Loads per-channel waveform features extracted by a separate pipeline.
  4. Identifies the stimulus file that elicited the strongest response.
  5. Finds the closest Paxinos marmoset atlas slice for the AP coordinate.
  6. Saves a 5-panel figure to the session plot directory and to the shared
     laminar-characterization directory.

Usage
-----
    python waveform_characterization_bydepth.py <monkey> <date>

Arguments
---------
    monkey   : subject identifier string, e.g. 'Pogo'
    date     : recording date string matching the data directory name
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import math
import os
import pickle
import socket
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from natsort import os_sorted
from pathlib import Path
from scipy.signal import find_peaks

from SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX import readMeta
from utils.utils_ephys import *
from utils.utils_meta import *

# ---------------------------------------------------------------------------
# Machine-specific data paths
# ---------------------------------------------------------------------------
host = socket.gethostname()
if 'rc.zi.columbia.edu' in host:
    engram_path = Path('/mnt/smb/locker/issa-locker')
elif 'DESKTOP' in host:
    engram_path = Path('Z:/')
elif 'Younah' in host:
    engram_path = Path('/Volumes/issa-locker/')

base_data_path    = engram_path / 'Data'
base_save_out_path = engram_path / 'users' / 'Younah' / 'ephys'

# ---------------------------------------------------------------------------
# Paxinos marmoset atlas slices — keyed by AP coordinate (mm)
# ---------------------------------------------------------------------------
_atlas_dir = (engram_path / 'users' / 'Younah' / 'code' / 'Paxinos_MarmosetAtlas').as_posix()
ATLAS_IMG_MAP = {
    -1.0: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-112.png',
    -0.7: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-111.png',
    -0.5: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-110.png',
     0.0: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-108.png',
     0.3: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-107.png',
     0.5: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-106.png',
     0.8: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-105.png',
     1.0: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-104.png',
     2.0: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-100.png',
     2.3: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-099.png',
     2.5: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-098.png',
     2.8: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-097.png',
     3.0: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-096.png',
     3.3: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-095.png',
     5.3: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-087.png',
     6.0: f'{_atlas_dir}/Paxinos_MarmosetAtlas_diagrams only-084.png',
}
_atlas_ap_coords = np.array(sorted(ATLAS_IMG_MAP.keys()))

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
monkey = sys.argv[1]
date   = sys.argv[2]

data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(
    base_data_path, monkey, date, base_save_out_path
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def smooth_data_np_average(arr, span):
    """Smooth arr with a centred running average of half-width span.

    NaN positions (from edge effects) are filled with the nearest valid value.
    """
    smoothed = np.array([np.average(arr[val - span:val + span + 1]) for val in range(len(arr))])
    non_nan_ind = np.argwhere(~np.isnan(smoothed))
    for ind, val in enumerate(smoothed):
        if math.isnan(val):
            smoothed[ind] = smoothed[non_nan_ind[np.argmin(non_nan_ind - ind)]]
    return smoothed

def make_probe(ax, ml, dv, ang, dep, n_resp):
    """Draw a probe schematic on a brain-atlas axes.

    Renders the full-length probe outline (black) and highlights the span
    of active recording sites (red).  Coordinates are in mm × 100 to match
    the atlas image pixel scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (already set up in atlas image units).
    ml : float
        Mediolateral entry-point coordinate (mm).
    dv : float
        Dorsoventral entry-point coordinate (mm).
    ang : float
        Insertion angle from horizontal (degrees).
    dep : float
        Insertion depth (µm).
    n_resp : int
        Number of active recording sites to highlight.

    Returns
    -------
    h : matplotlib.patches.Rectangle
        The probe-outline patch.
    """
    width_probe = 150 / 1000 * 100   # 150 µm shank → display units
    len_probe   = 1 * 10 * 100        # 1 cm probe   → display units
    width_site  = 10 / 1000 * 100     # 10 µm site   → display units

    dep_d = dep / 1000 * 100
    ml_d  = ml * 100
    dv_d  = dv * 100
    tip_x = ml_d - dep_d * math.cos(ang * math.pi / 180)
    tip_y = dv_d - dep_d * math.sin(ang * math.pi / 180) - width_probe / 2

    h = ax.add_patch(Rectangle(
        (tip_x, tip_y), len_probe, width_probe,
        angle=ang, color='k', fill=None, linewidth=2,
    ))
    ax.add_patch(Rectangle(
        (tip_x, tip_y), n_resp * width_site, width_probe,
        angle=ang, color='r', alpha=0.5, linewidth=2,
    ))
    h.set_visible(True)
    return h


def make_laminar_figure(avg_lfp, spike_dur_all, pt_ratio_all, spike_height_all,
                        psth_to_plot, chanmap, img, max_depth):
    """Build the 5-panel laminar characterisation figure.

    Layout (left → right):
      Panel 0 — Spike duration (ms) vs depth, with smoothed trend line.
      Panel 1 — Peak/trough amplitude ratio vs depth.
      Panel 2 — Spike height (µV) vs depth.
      Panel 3 — Raw LFP traces vs depth (inserted as inset axes).
      Panel 4 — PSTH z-score heatmap vs depth (inserted as inset axes).
      Panel 5 — Paxinos atlas slice with probe placement overlay.

    The following module-level variables must be set before calling:
        brain_boundary    (int)   — channel index of the brain surface
        chanmap_type      (str)   — 'linear', 'checkerboard', or 'long'
        Fs                (float) — LFP sampling rate (Hz)
        scenefile_max     (str)   — path of the chosen stimulus file
        stim_dur_scenefile (float) — stimulus duration of that file (s)
        ml_coord, dv_coord, ang  — probe coordinates for the atlas panel

    Parameters
    ----------
    avg_lfp : ndarray, shape (384, n_samples)
        CAR-corrected trial-averaged LFP sorted by depth (shallow → deep).
    spike_dur_all : ndarray, shape (384, n_spikes)
        Per-channel spike durations in seconds.
    pt_ratio_all : ndarray, shape (384, n_spikes)
        Per-channel peak-to-trough amplitude ratios.
    spike_height_all : ndarray, shape (384, n_spikes)
        Per-channel spike heights in volts.
    psth_to_plot : ndarray, shape (384, n_bins)
        Z-scored PSTH, channels sorted by depth.
    chanmap : ndarray, shape (n_chans, 2)
        Channel map — column 0: hardware index, column 1: depth rank.
    img : ndarray
        RGBA atlas image loaded via cv2.
    max_depth : float
        Depth of the shallowest channel below the brain surface (µm).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    matplotlib.rcParams.update({'font.size': 15, 'figure.facecolor': (1, 1, 1)})

    N_CHANS  = 384
    y_delta  = 300   # µm padding above/below depth axis
    dist_site = 20 if (chanmap[1, 1] - chanmap[0, 1]) != 1 else 10

    depth_vals = max_depth - (np.arange(N_CHANS) + 1) * dist_site   # µm per channel
    ylim_top   = max_depth + y_delta
    ylim_bot   = depth_vals[-1] - y_delta if depth_vals[-1] < 0 else -y_delta

    fig, ax = plt.subplots(1, 5, figsize=(40, 16))
    cmap = matplotlib.colormaps['tab20']

    def _set_ylim(axis):
        axis.set_ylim([ylim_top, ylim_bot])

    def _draw_depth_panel(axis, data_per_chan, color, xlabel):
        """Scatter + smoothed line + dashed peak-depth markers on one panel."""
        axis.scatter(data_per_chan, depth_vals, c=color, alpha=0.5)
        smoothed = smooth_data_np_average(data_per_chan, 10)
        peaks, _ = find_peaks(smoothed, distance=20)
        for p in peaks:
            axis.axhline(y=depth_vals[p], c='r', linestyle='--')
        twin = axis.twinx()
        _set_ylim(twin)
        twin.set_yticks(depth_vals[peaks])
        twin.set_yticklabels([str(p) for p in peaks], c='r')
        axis.plot(smoothed, depth_vals, c=color, linewidth=2)
        _set_ylim(axis)
        axis.set_xlabel(xlabel)

    # --- Panel 0: spike duration ---
    spike_dur_mean = np.nanmean(spike_dur_all[np.argsort(chanmap[:, 1]), :], axis=1) * 1000
    ax[0].axvline(x=0, c='lime', linestyle='--')
    ax[0].set_xticks([spike_dur_mean.min(), spike_dur_mean.max()])
    ax[0].set_xticklabels(
        [f'{spike_dur_mean.min():.2f}', f'{spike_dur_mean.max():.2f}'], color='lime'
    )
    _draw_depth_panel(ax[0], spike_dur_mean, 'lime', 'Spike Duration (ms)')
    ax[0].set_ylabel('Depth (µm)')
    ax[0].patch.set_facecolor('None')

    # --- Panel 1: peak/trough ratio ---
    pt_mean = np.nanmean(pt_ratio_all[np.argsort(chanmap[:, 1]), :], axis=1)
    inf_idx = np.where(pt_mean == np.inf)[0]
    for i in inf_idx:   # interpolate over any inf values
        pt_mean[i] = np.nanmean((pt_mean[i - 1], pt_mean[i + 1]))
    ax[1].axvline(x=-1, c='k', linestyle='--')
    _draw_depth_panel(ax[1], pt_mean, cmap(0), 'Spike Peak/Trough Ratio')

    # --- Panel 2: spike height ---
    spike_h_mean = np.nanmean(spike_height_all[np.argsort(chanmap[:, 1]), :], axis=1) * 1e6
    _draw_depth_panel(ax[2], spike_h_mean, cmap(0), 'Spike Height (µV)')

    # --- Panel 3: placeholder — LFP is overlaid below via inset axes ---
    ax[3].set_axis_off()

    # --- Panel 4: brain atlas with probe placement ---
    ax[4].set_ylim([0, 18.7 * 100])
    ax[4].set_xlim(4 * 100, 1150)
    atlas_crop = (600, 100, 1580, 2200)   # (xmin, ymin, xmax, ymax) in image pixels
    w = ax[4].get_xlim()[1] - ax[4].get_xlim()[0]
    h = ax[4].get_ylim()[1] - ax[4].get_ylim()[0]
    axin = ax[4].inset_axes(
        [ax[4].get_xlim()[0] + 10, ax[4].get_ylim()[0] + 30, w, h],
        transform=ax[4].transData, zorder=-3,
    )
    axin.imshow(img[atlas_crop[1]:atlas_crop[3], atlas_crop[0]:atlas_crop[2], :])
    axin.set_axis_off()
    ax[4].scatter(ml_coord * 100, dv_coord * 100, c='k')
    n_resp = 384 if (chanmap[:, 0] == chanmap[:, 1]).all() else 384 * 2
    make_probe(ax[4], ml_coord, dv_coord, 90 - ang, max_depth, n_resp)

    fig.tight_layout()
    fig.supylabel('Depth (µm)', x=-0.02, fontsize=30, fontweight='bold')

    # --- Inset: Raw LFP traces aligned to the depth axis ---
    pos0 = ax[0].get_position()
    pos1 = ax[1].get_position()
    pos2 = ax[2].get_position()
    ywidth   = pos0.y1 - pos0.y0
    newy0    = pos1.y0 + ywidth / (max_depth + y_delta * 2) * y_delta
    if depth_vals[-1] < 0:
        newywidth = ywidth / abs(depth_vals[-1] - y_delta - (max_depth + y_delta)) * N_CHANS * dist_site
    else:
        newywidth = ywidth / (max_depth + y_delta * 2) * N_CHANS * dist_site

    lfp_x0 = pos2.x0 + pos1.x0 - pos0.x0
    ax_lfp  = fig.add_axes([lfp_x0, newy0, pos0.x1 - pos0.x0, newywidth])
    ax_lfp.autoscale(enable=True, tight=True)

    t_lfp_b, t_lfp_a, t_lfp_s = 0.4, 0.1, 0.3
    t_lfp_win = int((t_lfp_b + t_lfp_a + t_lfp_s) * Fs)
    t_axis = np.linspace(-t_lfp_b, t_lfp_a + t_lfp_s, num=t_lfp_win)
    for i in range(N_CHANS):
        color = 'r' if i == brain_boundary else 'k'
        lw    = 1.0 if i == brain_boundary else 0.5
        ax_lfp.plot(t_axis, -avg_lfp[i, :] * 500 + max_depth - (i + 1) * dist_site,
                    linewidth=lw, color=color)
    ax_lfp.set_title('Raw LFP')
    ax_lfp.invert_yaxis()
    ax_lfp.set_xlabel('Time (s)')
    ax_lfp.set_yticklabels('')
    ax_lfp.plot([0, 0], ax_lfp.get_ylim(), 'k-')

    # --- Inset: z-scored PSTH heatmap aligned to the depth axis ---
    ax_psth = fig.add_axes([pos0.x0, newy0, pos0.x1 - pos0.x0, newywidth])
    ax_psth.autoscale(enable=True, tight=True)
    psth_data = np.repeat(psth_to_plot, 2, axis=0) if chanmap_type == 'long' else psth_to_plot
    ax_psth = sns.heatmap(psth_data, yticklabels='', cbar=None, vmin=-5, vmax=5)
    ax_psth.invert_yaxis()
    ax_psth.set_zorder(-2)

    xmin_p, xmax_p = ax_psth.get_xlim()
    n_bins_p   = int(xmax_p - xmin_p)
    t_psth_b   = 0.1
    bw_psth    = 0.01
    bin_edges  = np.hstack((
        np.arange(-t_psth_b, stim_dur_scenefile - 0.05, bw_psth),
        stim_dur_scenefile - 0.05,
    ))
    xticks_p   = np.arange(n_bins_p + 1)
    xlabels_p  = [''] * (n_bins_p + 1)
    for k in range(int((n_bins_p + 1) / 5)):
        xlabels_p[k * 5] = round(bin_edges[k * 5], 2)
    xlabels_p[-1] = round(bin_edges[-1], 2)

    ax3 = ax_psth.twiny()
    ax3.set_xticks(xticks_p)
    ax3.set_xticklabels(xlabels_p)
    ax3.set_xlabel('Time (s)')
    ax_psth.set_xticks([])
    ax_psth.set_xticklabels('')
    ymin_p, ymax_p = ax_psth.get_ylim()
    ax_psth.plot([t_psth_b / bw_psth] * 2, [ymin_p, ymax_p], 'k--')
    ax_psth.set_title(Path(scenefile_max).stem)

    return fig


# ---------------------------------------------------------------------------
# Main loop — one iteration per session directory
# ---------------------------------------------------------------------------

for n, (data_path, save_out_path, plot_save_out_path) in enumerate(
    zip(data_path_list, save_out_path_list, plot_save_out_path_list)
):
    print(data_path)

    # ------------------------------------------------------------------
    # 1. Channel map + probe configuration
    # ------------------------------------------------------------------
    chanmap, imroTbl = get_chanmap(data_path)
    chanmap_type = get_chanmap_type(chanmap, imroTbl)
    if chanmap_type == 'checkerboard':
        chanmap[:, 1] = chanmap[:, 0]

    # ------------------------------------------------------------------
    # 2. Recording coordinates and insertion depth
    # ------------------------------------------------------------------
    ap_coord, dv_coord, ml_coord, ang, hang, depth_start = get_coords_sess(
        base_data_path, monkey, date
    )
    max_depth = depth_start

    # ------------------------------------------------------------------
    # 3. Load LFP and compute trial-averaged, CAR-corrected response
    # ------------------------------------------------------------------
    stim_info_sess = pickle.load(open(save_out_path / 'stim_info_sess', 'rb'))

    LFP_dir = data_path / 'LFP'
    try:
        x = np.load(LFP_dir / 'lfp_mat.npz')
    except Exception:
        files_found = os_sorted(LFP_dir.glob('lfp_mat_5*.npz'))
        if files_found:
            x = np.load(files_found[0])
        else:
            print(f'  No LFP file found in {LFP_dir}, skipping.')
            continue

    lfp_allchs = x['lfp_allchs']
    t  = x['t']
    Fs = x['Fs']
    n_chans = lfp_allchs.shape[0]

    t_trs     = np.sort(np.hstack([stim_info_sess[s]['t_on'] for s in stim_info_sess]))
    t_trs     = t_trs[~np.isnan(t_trs)]
    n_trs_max = np.argmin(np.abs(t_trs - t[-1]))

    t_before, t_after, stim_dur = 0.4, 0.1, 0.3
    t_win = int((t_before + t_after + stim_dur) * Fs)
    data_bystim = np.full((n_chans, t_win, n_trs_max + 1), np.nan)
    for i, on_t in enumerate(t_trs[:n_trs_max + 1]):
        tidx     = np.argmin(np.abs(t - on_t))
        samp_win = [tidx - t_before * Fs, tidx + (stim_dur + t_after) * Fs]
        if (tidx - t_before * Fs > 0
                and (samp_win[1] - samp_win[0] == t_win)
                and samp_win[1] < lfp_allchs.shape[1]):
            data_bystim[:, :, i] = lfp_allchs[:, int(samp_win[0]):int(samp_win[1])]

    avg_lfp = np.nanmean(data_bystim, axis=2)
    avg_lfp -= np.nanmean(avg_lfp, axis=0)   # common-average reference

    depth_order = np.argsort(chanmap[:, 1])
    avg_lfp    = avg_lfp[depth_order]
    lfp_allchs = lfp_allchs[depth_order]

    # ------------------------------------------------------------------
    # 4. Load pre-computed waveform features
    # ------------------------------------------------------------------
    wf_features      = np.load(save_out_path / 'wf_features.npz')
    spike_dur_all    = wf_features['spike_dur']
    spike_height_all = wf_features['spike_height']
    pt_ratio_all     = wf_features['pt_ratio']

    try:
        brain_boundary = np.load(save_out_path / 'brain_boundary.npy')
    except Exception:
        brain_boundary = 383

    # Normalise to a plain int (0-d or 1-d numpy arrays both accepted)
    if isinstance(brain_boundary, np.ndarray):
        brain_boundary = int(brain_boundary.flat[0]) if brain_boundary.size > 0 else 383
    else:
        brain_boundary = int(brain_boundary)

    # ------------------------------------------------------------------
    # 5. Load PSTH metadata and rank scenefiles by response strength
    # ------------------------------------------------------------------
    psth_scenefile_meta = pickle.load(open(save_out_path / 'psth_scenefile_meta', 'rb'))
    sess = np.array(list(psth_scenefile_meta.keys()))

    max_resp_all = []
    for s in sess:
        f    = os_sorted(save_out_path.glob('psth_' + Path(s).stem))[0]
        psth = pickle.load(open(f, 'rb'))[:, :35]
        mu   = np.nanmean(psth, axis=1, keepdims=True)
        sd   = np.nanstd(psth,  axis=1, keepdims=True)
        z    = (psth - mu) / sd
        max_resp_all.append(np.sum(np.nanmean(z, axis=0)[15:25]))

    max_resp_all = np.array(max_resp_all)
    valid_mask   = ~np.isnan(max_resp_all)
    sess         = sess[valid_mask]
    max_resp_all = max_resp_all[valid_mask]
    scenefile_ordered = sess[np.argsort(max_resp_all)[::-1]]

    # Pick the highest-response file with a short (< 1 s) stimulus
    scenefile_max = None
    for s in scenefile_ordered:
        if psth_scenefile_meta[s]['stim_dur'] < 1:
            scenefile_max      = s
            stim_dur_scenefile = psth_scenefile_meta[s]['stim_dur']
            break
    if scenefile_max is None:
        print('  No short-stimulus scenefile found; skipping session.')
        continue

    # ------------------------------------------------------------------
    # 6. Load and z-score the PSTH for the chosen scenefile
    # ------------------------------------------------------------------
    psth_path = os_sorted(save_out_path.glob('psth_' + Path(scenefile_max).stem))[0]
    mean_psth = pickle.load(open(psth_path, 'rb'))[np.argsort(chanmap[:, 1]), :]

    psth_meta  = psth_scenefile_meta[scenefile_max]
    bw_psth    = 0.01
    n_bins_keep = round((psth_meta['t_before'] + psth_meta['stim_dur'] - 0.05) / bw_psth)
    mean_psth  = mean_psth[:, :n_bins_keep]
    mu_p = np.nanmean(mean_psth, axis=1, keepdims=True)
    sd_p = np.nanstd(mean_psth,  axis=1, keepdims=True)
    zscored = (mean_psth - mu_p) / sd_p

    # ------------------------------------------------------------------
    # 7. Load the closest Paxinos atlas slice for the AP coordinate
    # ------------------------------------------------------------------
    atlas_index = _atlas_ap_coords[np.argmin(np.abs(ap_coord - _atlas_ap_coords))]
    img = cv2.cvtColor(
        cv2.imread(ATLAS_IMG_MAP[atlas_index], cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA,
    )

    # ------------------------------------------------------------------
    # 8. Build the figure and save
    # ------------------------------------------------------------------
    fig = make_laminar_figure(
        avg_lfp, spike_dur_all, pt_ratio_all, spike_height_all,
        zscored, chanmap, img, max_depth,
    )

    ang_deg  = round(90 - ang)
    hang_deg = round(hang)
    title = (
        f'{date}\n'
        f'AP: {ap_coord}  DV: {dv_coord}  ML: {ml_coord}  ang: {ang_deg}'
        + (f'  hang: {hang_deg}' if 'HAng' in data_path.name else '')
        + f'  dep: {max_depth}'
    )
    fig.suptitle(title, y=1.2, fontsize=30, fontweight='bold')
    fig.text(0, 1.02, '\n'.join(Path(s).name for s in sess), fontsize=25)
    fig.text(0.9, 1.0, f'chanmap: {chanmap_type}', fontsize=25)

    fig.savefig(plot_save_out_path / 'waveform_features.png', bbox_inches='tight')

    stem = (
        f'{date}_ap_{ap_coord}_dv_{dv_coord}_ml_{ml_coord}_ang_{ang_deg}'
        + (f'_hang_{hang_deg}' if 'HAng' in data_path.name else '')
        + f'_dep_{max_depth}.png'
    )
    fig.savefig(
        base_save_out_path / f'{monkey}_plots' / 'laminar_characterization' / stem,
        bbox_inches='tight',
    )
