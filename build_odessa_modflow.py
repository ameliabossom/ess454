"""Build a conceptual MODFLOW 6 model for the Odessa Subarea.

The model is intentionally transparent: hydrogeologic units, recharge,
pumping, nitrate source strength, and geophysical proxy properties are all
defined in this script.  It is a screening model for comparing rock controls
on flow, not a calibrated regulatory model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import flopy
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_WS = PROJECT_ROOT / "model" / "odessa_modflow6"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DEFAULT_MF6_EXE = Path(
    r"C:\Users\ameli\Documents\Codex\2026-04-30"
    r"\set-up-my-computer-to-run\bin\mf6.exe"
)

MODEL_NAME = "odessa_rock_properties"
GWF_NAME = "odessa_gwf"
GWT_NAME = "odessa_nitrate"

NROW = 40
NCOL = 60
NLAY = 8
DELR = 1_000.0
DELC = 1_000.0
LX_KM = NCOL * DELR / 1_000.0
LY_KM = NROW * DELC / 1_000.0

WEST_HEAD_M = 480.0
EAST_HEAD_M = 430.0
BACKGROUND_NITRATE_MG_L = 2.0
SOURCE_NITRATE_MG_L = 25.0
BACKGROUND_RECHARGE_M_D = 2.0e-5
IRRIGATED_RECHARGE_M_D = 2.0e-4
PUMPING_PER_WELL_M3_D = -3_000.0
NUMBER_OF_PUMPING_CELLS = 10

ODESSA_LAT = 47.3332
ODESSA_LON = -118.6908
NORTH_LAT = 47.51
SOUTH_LAT = 47.15
WEST_LON = -119.09
EAST_LON = -118.29


@dataclass(frozen=True)
class LayerProperty:
    name: str
    short_name: str
    kh_m_d: float
    kv_m_d: float
    porosity: float
    resistivity_ohm_m: float
    thermal_conductivity_w_m_k: float
    volumetric_heat_capacity_mj_m3_k: float
    color: str
    note: str


LAYERS = [
    LayerProperty(
        "Unconsolidated-deposit aquifer",
        "Overburden",
        30.0,
        3.0,
        0.25,
        45.0,
        1.7,
        2.4,
        "#eadf7a",
        "Loess, flood deposits, alluvium; relatively high pore-space flow.",
    ),
    LayerProperty(
        "Upper confining unit",
        "Confining",
        1.0e-3,
        1.0e-5,
        0.12,
        25.0,
        1.5,
        2.5,
        "#b48a60",
        "Fine interbeds and weathered material limiting vertical leakage.",
    ),
    LayerProperty(
        "Saddle Mountains Basalt aquifer",
        "Saddle Mtn.",
        0.73,
        7.3e-5,
        0.08,
        180.0,
        1.8,
        2.7,
        "#d45835",
        "Youngest CRBG basalt aquifer; more discontinuous than deeper units.",
    ),
    LayerProperty(
        "Mabton/interflow confining unit",
        "Mabton",
        5.0e-4,
        5.0e-6,
        0.10,
        30.0,
        1.5,
        2.5,
        "#9d734d",
        "Sedimentary interbed and lower-permeability flow interiors.",
    ),
    LayerProperty(
        "Wanapum Basalt aquifer",
        "Wanapum",
        1.58,
        1.58e-4,
        0.07,
        220.0,
        1.9,
        2.75,
        "#e69b83",
        "Important Odessa aquifer; flow concentrated in interflow zones.",
    ),
    LayerProperty(
        "Vantage/interflow confining unit",
        "Vantage",
        5.0e-4,
        5.0e-6,
        0.10,
        35.0,
        1.5,
        2.5,
        "#8d6848",
        "Regional interbed/confining interval between Wanapum and Grande Ronde.",
    ),
    LayerProperty(
        "Grande Ronde Basalt aquifer",
        "Grande Ronde",
        1.49,
        1.49e-4,
        0.06,
        260.0,
        2.0,
        2.8,
        "#eec0ac",
        "Deep, laterally extensive CRBG aquifer targeted by pumping wells.",
    ),
    LayerProperty(
        "Pre-Miocene older bedrock",
        "Older bedrock",
        1.0e-4,
        1.0e-6,
        0.02,
        1_000.0,
        2.8,
        2.6,
        "#6f7d7e",
        "Basement confining unit below the regional basalt aquifer system.",
    ),
]


def odessa_xy_km() -> tuple[float, float]:
    """Convert Odessa latitude/longitude to the model coordinate system."""
    x_km = (ODESSA_LON - WEST_LON) / (EAST_LON - WEST_LON) * LX_KM
    y_km = (ODESSA_LAT - SOUTH_LAT) / (NORTH_LAT - SOUTH_LAT) * LY_KM
    return x_km, y_km


def row_from_y_km(y_km: float) -> int:
    return int(np.clip(NROW - 1 - math.floor(y_km), 0, NROW - 1))


def col_from_x_km(x_km: float) -> int:
    return int(np.clip(math.floor(x_km), 0, NCOL - 1))


def build_grid() -> dict[str, np.ndarray]:
    col = np.arange(NCOL)
    row = np.arange(NROW)
    x_km = (col + 0.5) * DELR / 1_000.0
    y_km = (NROW - row - 0.5) * DELC / 1_000.0
    x2d, y2d = np.meshgrid(x_km, y_km)
    xnorm = x2d / LX_KM
    ynorm = y2d / LY_KM

    # A smooth topographic surface and draped strata, inspired by the supplied
    # Odessa B-B' style section but scaled to the user-provided elevation range.
    top = (
        500.0
        - 55.0 * xnorm
        + 16.0 * np.sin(2.0 * np.pi * (xnorm - 0.12))
        + 9.0 * np.cos(2.0 * np.pi * (ynorm - 0.25))
    )

    basin_center = np.exp(-((xnorm - 0.55) / 0.20) ** 2)
    east_thickening = 1.0 / (1.0 + np.exp(-(xnorm - 0.58) / 0.06))
    fold_warp = 15.0 * np.sin(np.pi * xnorm) + 8.0 * np.cos(2.0 * np.pi * ynorm)

    th_overburden = 28.0 + 45.0 * basin_center + 10.0 * east_thickening
    th_upper_confining = 22.0 + 13.0 * basin_center
    th_saddle = 70.0 + 35.0 * east_thickening + 12.0 * np.sin(np.pi * xnorm)
    th_mabton = 28.0 + 10.0 * basin_center
    th_wanapum = 150.0 + 45.0 * np.sin(np.pi * xnorm) + 18.0 * east_thickening
    th_vantage = 30.0 + 12.0 * basin_center

    botm = np.zeros((NLAY, NROW, NCOL), dtype=float)
    botm[0] = top - th_overburden
    botm[1] = botm[0] - th_upper_confining
    botm[2] = botm[1] - th_saddle
    botm[3] = botm[2] - th_mabton
    botm[4] = botm[3] - th_wanapum
    botm[5] = botm[4] - th_vantage

    older_bedrock_top = (
        -545.0
        + 55.0 * np.exp(-((xnorm - 0.08) / 0.09) ** 2)
        + 42.0 * np.exp(-((xnorm - 0.93) / 0.10) ** 2)
        - 30.0 * basin_center
        + fold_warp
    )
    older_bedrock_top = np.minimum(older_bedrock_top, botm[5] - 65.0)
    older_bedrock_top = np.maximum(older_bedrock_top, -590.0)
    botm[6] = older_bedrock_top
    botm[7] = -600.0

    return {
        "top": top,
        "botm": botm,
        "x_km": x_km,
        "y_km": y_km,
        "x2d_km": x2d,
        "y2d_km": y2d,
    }


def build_stresses(grid: dict[str, np.ndarray]) -> dict[str, np.ndarray | list]:
    x2d = grid["x2d_km"]
    y2d = grid["y2d_km"]
    odessa_x, odessa_y = odessa_xy_km()

    source_zone = (
        ((x2d - odessa_x) / 13.5) ** 2 + ((y2d - odessa_y) / 8.5) ** 2
    ) <= 1.0
    recharge = np.where(source_zone, IRRIGATED_RECHARGE_M_D, BACKGROUND_RECHARGE_M_D)
    recharge_conc = np.where(
        source_zone, SOURCE_NITRATE_MG_L, BACKGROUND_NITRATE_MG_L
    )

    pump_offsets = [
        (-4.0, -3.0),
        (-2.0, -4.0),
        (0.0, -4.5),
        (2.5, -3.5),
        (4.5, -1.5),
        (-4.5, 1.0),
        (-2.0, 2.8),
        (0.5, 3.2),
        (2.8, 2.0),
        (4.5, 0.5),
    ]
    wells = []
    used_cells = set()
    for dx, dy in pump_offsets:
        col = col_from_x_km(odessa_x + dx)
        row = row_from_y_km(odessa_y + dy)
        cell = (6, row, col)
        if cell in used_cells:
            continue
        used_cells.add(cell)
        wells.append((6, row, col, PUMPING_PER_WELL_M3_D))

    if len(wells) != NUMBER_OF_PUMPING_CELLS:
        raise ValueError("Expected ten unique pumping cells near Odessa.")

    return {
        "source_zone": source_zone,
        "recharge": recharge,
        "recharge_conc": recharge_conc,
        "wells": wells,
    }


def build_property_arrays(
    grid: dict[str, np.ndarray], stresses: dict[str, np.ndarray | list]
) -> dict[str, np.ndarray]:
    x2d = grid["x2d_km"]
    y2d = grid["y2d_km"]
    odessa_x, odessa_y = odessa_xy_km()

    kh = np.zeros((NLAY, NROW, NCOL), dtype=float)
    kv = np.zeros_like(kh)
    porosity = np.zeros_like(kh)
    resistivity = np.zeros_like(kh)
    thermal_k = np.zeros_like(kh)
    heat_capacity = np.zeros_like(kh)

    fracture_corridor = np.exp(-((x2d - (odessa_x + 7.0)) / 2.8) ** 2) * np.exp(
        -((y2d - odessa_y) / 14.0) ** 2
    )
    source_zone = stresses["source_zone"]
    nitrate_factor = np.where(source_zone, 0.78, 1.0)

    for k, layer in enumerate(LAYERS):
        kh[k] = layer.kh_m_d
        kv[k] = layer.kv_m_d
        porosity[k] = layer.porosity
        resistivity[k] = layer.resistivity_ohm_m
        thermal_k[k] = layer.thermal_conductivity_w_m_k
        heat_capacity[k] = layer.volumetric_heat_capacity_mj_m3_k

    for k in (2, 4, 6):
        kh[k] *= 1.0 + 3.5 * fracture_corridor
        kv[k] *= 1.0 + 9.0 * fracture_corridor
        resistivity[k] *= 1.0 / (1.0 + 1.5 * fracture_corridor)

    for k in range(NLAY):
        resistivity[k] *= nitrate_factor

    return {
        "kh": kh,
        "kv": kv,
        "porosity": porosity,
        "resistivity": resistivity,
        "electrical_conductivity_s_m": 1.0 / resistivity,
        "thermal_k": thermal_k,
        "heat_capacity": heat_capacity,
        "fracture_corridor": fracture_corridor,
    }


def build_simulation(mf6_exe: Path) -> tuple[flopy.mf6.MFSimulation, dict]:
    grid = build_grid()
    stresses = build_stresses(grid)
    props = build_property_arrays(grid, stresses)

    MODEL_WS.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(
        sim_name=MODEL_NAME,
        exe_name=str(mf6_exe),
        version="mf6",
        sim_ws=str(MODEL_WS),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=1,
        perioddata=[(36_500.0, 160, 1.0)],
    )

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=GWF_NAME,
        save_flows=True,
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=DELC,
        top=grid["top"],
        botm=grid["botm"],
        idomain=1,
    )

    xfrac = (np.arange(NCOL) + 0.5) / NCOL
    start_2d = WEST_HEAD_M + (EAST_HEAD_M - WEST_HEAD_M) * xfrac
    strt = np.repeat(start_2d[np.newaxis, np.newaxis, :], NLAY * NROW, axis=0)
    strt = strt.reshape((NLAY, NROW, NCOL))
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=props["kh"],
        k33=props["kv"],
    )

    chd_spd = []
    for k in range(NLAY):
        for row in range(NROW):
            chd_spd.append(((k, row, 0), WEST_HEAD_M))
            chd_spd.append(((k, row, NCOL - 1), EAST_HEAD_M))
    flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chd_spd),
        stress_period_data={0: chd_spd},
        save_flows=True,
        pname="CHD-1",
    )

    with contextlib.redirect_stdout(io.StringIO()):
        flopy.mf6.ModflowGwfrcha(
            gwf,
            recharge=stresses["recharge"],
            auxiliary=["CONCENTRATION"],
            aux=[stresses["recharge_conc"]],
            save_flows=True,
            pname="RCH-1",
        )
    flopy.mf6.ModflowGwfwel(
        gwf,
        maxbound=len(stresses["wells"]),
        stress_period_data={0: stresses["wells"]},
        save_flows=True,
        pname="WEL-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{GWF_NAME}.hds",
        budget_filerecord=f"{GWF_NAME}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    gwt = flopy.mf6.ModflowGwt(sim, modelname=GWT_NAME, save_flows=True)
    flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=DELC,
        top=grid["top"],
        botm=grid["botm"],
        idomain=1,
    )
    flopy.mf6.ModflowGwtic(
        gwt, strt=np.full((NLAY, NROW, NCOL), BACKGROUND_NITRATE_MG_L)
    )
    flopy.mf6.ModflowGwtmst(gwt, porosity=props["porosity"])
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
    flopy.mf6.ModflowGwtdsp(
        gwt, xt3d_off=True, alh=120.0, alv=4.0, ath1=20.0, atv=1.0
    )
    flopy.mf6.ModflowGwtssm(
        gwt,
        sources=[("RCH-1", "AUX", "CONCENTRATION")],
        save_flows=True,
    )
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord=f"{GWT_NAME}.ucn",
        budget_filerecord=f"{GWT_NAME}.cbc",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    )

    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=GWF_NAME,
        exgmnameb=GWT_NAME,
    )

    ims_flow = flopy.mf6.ModflowIms(
        sim,
        filename="flow.ims",
        pname="FLOW_IMS",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_maximum=200,
        outer_dvclose=1.0e-4,
        inner_maximum=500,
        inner_dvclose=1.0e-4,
        rcloserecord="1.0e-2 STRICT",
        linear_acceleration="BICGSTAB",
    )
    sim.register_ims_package(ims_flow, [GWF_NAME])

    ims_transport = flopy.mf6.ModflowIms(
        sim,
        filename="transport.ims",
        pname="TRANSPORT_IMS",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_maximum=200,
        outer_dvclose=1.0e-5,
        inner_maximum=500,
        inner_dvclose=1.0e-5,
        rcloserecord="1.0e-6 STRICT",
        linear_acceleration="BICGSTAB",
    )
    sim.register_ims_package(ims_transport, [GWT_NAME])

    data = {"grid": grid, "stresses": stresses, "props": props}
    return sim, data


def write_tables(data: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLE_DIR / "rock_properties.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(LAYERS[0]).keys()))
        writer.writeheader()
        for layer in LAYERS:
            writer.writerow(asdict(layer))

    with (TABLE_DIR / "pumping_wells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["well", "layer", "row", "column", "x_km", "y_km", "q_m3_d"]
        )
        writer.writeheader()
        for i, (lay, row, col, q) in enumerate(data["stresses"]["wells"], start=1):
            writer.writerow(
                {
                    "well": i,
                    "layer": LAYERS[lay].short_name,
                    "row": row + 1,
                    "column": col + 1,
                    "x_km": (col + 0.5) * DELR / 1_000.0,
                    "y_km": (NROW - row - 0.5) * DELC / 1_000.0,
                    "q_m3_d": q,
                }
            )


def run_simulation(sim: flopy.mf6.MFSimulation) -> None:
    sim.write_simulation(silent=True)
    success, log = sim.run_simulation(silent=True, report=True)
    run_log = MODEL_WS / "mf6_run.log"
    run_log.write_text("\n".join(log), encoding="utf-8")
    if not success:
        tail = "\n".join(log[-80:])
        raise RuntimeError(f"MODFLOW 6 did not terminate normally:\n{tail}")


def layer_boundaries_for_row(grid: dict[str, np.ndarray], row: int) -> list[np.ndarray]:
    boundaries = [grid["top"][row, :]]
    boundaries.extend(grid["botm"][k, row, :] for k in range(NLAY))
    return boundaries


def plot_lithology_cross_section(grid: dict[str, np.ndarray], row: int) -> Path:
    out = FIGURE_DIR / "odessa_lithology_cross_section.png"
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    x = grid["x_km"]
    boundaries = layer_boundaries_for_row(grid, row)

    for k, layer in enumerate(LAYERS):
        for j in range(NCOL):
            ax.add_patch(
                Rectangle(
                    (j, boundaries[k + 1][j]),
                    1.0,
                    boundaries[k][j] - boundaries[k + 1][j],
                    facecolor=layer.color,
                    edgecolor="none",
                )
            )
        ax.plot(x, boundaries[k + 1], color="black", linewidth=0.35, alpha=0.45)

    odessa_x, _ = odessa_xy_km()
    ax.axvline(odessa_x, color="black", linestyle="--", linewidth=1.0)
    ax.text(odessa_x + 0.4, 470, "Odessa", fontsize=9, va="top")
    ax.set_xlim(0, LX_KM)
    ax.set_ylim(-610, 525)
    ax.set_xlabel("East-west distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Conceptual Odessa hydrogeologic cross section")
    ax.grid(True, color="white", linewidth=0.3, alpha=0.45)
    legend_handles = [Patch(facecolor=layer.color, label=layer.short_name) for layer in LAYERS]
    ax.legend(
        handles=legend_handles,
        ncol=4,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
    )
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def draw_numeric_section(
    ax: plt.Axes,
    grid: dict[str, np.ndarray],
    values: np.ndarray,
    row: int,
    cmap: str,
    norm: mcolors.Normalize,
    title: str,
    cbar_label: str,
    contour_values: np.ndarray | None = None,
    contour_labels: bool = True,
) -> None:
    boundaries = layer_boundaries_for_row(grid, row)
    cm = plt.get_cmap(cmap)

    for k in range(NLAY):
        for j in range(NCOL):
            ax.add_patch(
                Rectangle(
                    (j, boundaries[k + 1][j]),
                    1.0,
                    boundaries[k][j] - boundaries[k + 1][j],
                    facecolor=cm(norm(values[k, row, j])),
                    edgecolor="none",
                )
            )
        ax.plot(grid["x_km"], boundaries[k + 1], color="black", linewidth=0.25, alpha=0.35)

    if contour_values is not None:
        zmid = np.zeros((NLAY, NCOL), dtype=float)
        for k in range(NLAY):
            zmid[k] = 0.5 * (boundaries[k] + boundaries[k + 1])
        xx = np.repeat(grid["x_km"][np.newaxis, :], NLAY, axis=0)
        cs = ax.contour(xx, zmid, contour_values[:, row, :], colors="black", linewidths=0.55)
        if contour_labels:
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(cbar_label)
    ax.set_xlim(0, LX_KM)
    ax.set_ylim(-610, 525)
    ax.set_xlabel("East-west distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title)


def plot_hydraulic_cross_section(
    grid: dict[str, np.ndarray],
    props: dict[str, np.ndarray],
    head: np.ndarray,
    row: int,
) -> Path:
    out = FIGURE_DIR / "odessa_hydraulic_conductivity_and_heads.png"
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    draw_numeric_section(
        ax,
        grid,
        props["kh"],
        row,
        "viridis",
        mcolors.LogNorm(vmin=1.0e-4, vmax=35.0),
        "Hydraulic conductivity and simulated head contours",
        "Horizontal K (m/day, log scale)",
        contour_values=head,
    )
    odessa_x, _ = odessa_xy_km()
    ax.axvline(odessa_x, color="white", linestyle="--", linewidth=0.9)
    ax.text(odessa_x + 0.4, 470, "Odessa", color="white", fontsize=9, va="top")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_electro_thermal_cross_section(
    grid: dict[str, np.ndarray],
    props: dict[str, np.ndarray],
    row: int,
) -> Path:
    out = FIGURE_DIR / "odessa_electrical_thermal_cross_sections.png"
    top3d = np.zeros((NLAY, NROW, NCOL), dtype=float)
    botm = grid["botm"]
    top3d[0] = grid["top"]
    for k in range(1, NLAY):
        top3d[k] = botm[k - 1]
    zmid = 0.5 * (top3d + botm)
    depth_below_land = np.maximum(top3d[0] - zmid, 0.0)
    temperature_c = 11.0 + 0.028 * depth_below_land

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), constrained_layout=True)
    draw_numeric_section(
        axes[0],
        grid,
        props["resistivity"],
        row,
        "cividis",
        mcolors.LogNorm(vmin=15.0, vmax=1_100.0),
        "Conceptual electrical resistivity by rock type, fracture zone, and nitrate source",
        "Resistivity (ohm m, log scale)",
    )
    draw_numeric_section(
        axes[1],
        grid,
        temperature_c,
        row,
        "inferno",
        mcolors.Normalize(vmin=10.0, vmax=44.0),
        "Conceptual temperature field from geothermal gradient",
        "Temperature (deg C)",
        contour_values=props["thermal_k"],
        contour_labels=False,
    )
    axes[1].text(
        0.5,
        -575,
        "Black contours show thermal conductivity boundaries from lithology.",
        fontsize=8,
        color="white",
    )
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_planview(
    grid: dict[str, np.ndarray],
    stresses: dict[str, np.ndarray | list],
    head: np.ndarray,
    conc: np.ndarray,
) -> Path:
    out = FIGURE_DIR / "odessa_planview_grande_ronde_heads_nitrate.png"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    extent = [0, LX_KM, 0, LY_KM]
    deep_layer = 6
    shallow_layer = 0

    im0 = axes[0].imshow(
        head[deep_layer],
        origin="upper",
        extent=extent,
        cmap="Blues",
        aspect="equal",
    )
    axes[0].contour(
        grid["x2d_km"],
        grid["y2d_km"],
        head[deep_layer],
        colors="black",
        linewidths=0.45,
        levels=np.arange(410, 491, 5),
    )
    plt.colorbar(im0, ax=axes[0], pad=0.01, label="Head (m)")
    axes[0].set_title("Grande Ronde simulated head")

    im1 = axes[1].imshow(
        conc[shallow_layer],
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=SOURCE_NITRATE_MG_L,
        aspect="equal",
    )
    axes[1].contour(
        grid["x2d_km"],
        grid["y2d_km"],
        stresses["source_zone"].astype(float),
        colors="cyan",
        linewidths=1.1,
        levels=[0.5],
    )
    plt.colorbar(im1, ax=axes[1], pad=0.01, label="Nitrate-N (mg/L)")
    axes[1].set_title("Shallow nitrate after 100 years")

    im2 = axes[2].imshow(
        conc[deep_layer],
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=5.0,
        aspect="equal",
    )
    axes[2].contour(
        grid["x2d_km"],
        grid["y2d_km"],
        stresses["source_zone"].astype(float),
        colors="cyan",
        linewidths=1.1,
        levels=[0.5],
    )
    plt.colorbar(im2, ax=axes[2], pad=0.01, label="Nitrate-N (mg/L)")
    axes[2].set_title("Grande Ronde nitrate, enhanced scale")

    for ax in axes:
        for lay, row, col, _ in stresses["wells"]:
            ax.plot(
                (col + 0.5) * DELR / 1_000.0,
                (NROW - row - 0.5) * DELC / 1_000.0,
                marker="v",
                color="white",
                markeredgecolor="black",
                markersize=7,
            )
        odessa_x, odessa_y = odessa_xy_km()
        ax.plot(odessa_x, odessa_y, marker="*", color="gold", markeredgecolor="black", markersize=11)
        ax.set_xlim(0, LX_KM)
        ax.set_ylim(0, LY_KM)
        ax.set_xlabel("East-west distance (km)")
        ax.set_ylabel("North-south distance (km)")
        ax.grid(color="white", linewidth=0.35, alpha=0.45)

    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_nitrate_cross_section(
    grid: dict[str, np.ndarray],
    conc: np.ndarray,
    row: int,
) -> Path:
    out = FIGURE_DIR / "odessa_nitrate_cross_section.png"
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    draw_numeric_section(
        ax,
        grid,
        conc,
        row,
        "magma",
        mcolors.Normalize(vmin=0.0, vmax=SOURCE_NITRATE_MG_L),
        "Simulated nitrate-N concentration after 100 years; leakage is mostly shallow",
        "Nitrate-N (mg/L)",
    )
    odessa_x, _ = odessa_xy_km()
    ax.axvline(odessa_x, color="white", linestyle="--", linewidth=0.9)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_summary(
    data: dict,
    head: np.ndarray,
    conc: np.ndarray,
    figures: list[Path],
) -> None:
    recharge = data["stresses"]["recharge"]
    source_zone = data["stresses"]["source_zone"]
    area_m2 = DELR * DELC
    summary = {
        "model": MODEL_NAME,
        "grid": {
            "nlay": NLAY,
            "nrow": NROW,
            "ncol": NCOL,
            "cell_size_m": [DELR, DELC],
            "domain_km": [LX_KM, LY_KM],
            "elevation_range_m": [500, -600],
        },
        "odessa_model_xy_km": odessa_xy_km(),
        "boundary_heads_m": {"west": WEST_HEAD_M, "east": EAST_HEAD_M},
        "recharge_m3_d": {
            "total": float(np.sum(recharge) * area_m2),
            "irrigated_source_zone": float(np.sum(recharge[source_zone]) * area_m2),
            "background_outside_source": float(np.sum(recharge[~source_zone]) * area_m2),
        },
        "pumping_m3_d": {
            "per_well": PUMPING_PER_WELL_M3_D,
            "number_of_cells": len(data["stresses"]["wells"]),
            "total": float(sum(w[-1] for w in data["stresses"]["wells"])),
        },
        "head_stats_m": {
            "min": float(np.nanmin(head)),
            "mean": float(np.nanmean(head)),
            "max": float(np.nanmax(head)),
        },
        "nitrate_stats_mg_l": {
            "min": float(np.nanmin(conc)),
            "mean": float(np.nanmean(conc)),
            "max": float(np.nanmax(conc)),
        },
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figures],
    }
    (OUTPUT_DIR / "model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def make_figures(data: dict, sim: flopy.mf6.MFSimulation) -> list[Path]:
    gwf = sim.get_model(GWF_NAME)
    gwt = sim.get_model(GWT_NAME)
    head = gwf.output.head().get_data()
    conc = gwt.output.concentration().get_data()
    row = row_from_y_km(odessa_xy_km()[1])
    figures = [
        plot_lithology_cross_section(data["grid"], row),
        plot_hydraulic_cross_section(data["grid"], data["props"], head, row),
        plot_electro_thermal_cross_section(data["grid"], data["props"], row),
        plot_planview(data["grid"], data["stresses"], head, conc),
        plot_nitrate_cross_section(data["grid"], conc, row),
    ]
    save_summary(data, head, conc, figures)
    return figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the conceptual Odessa MODFLOW 6 model."
    )
    parser.add_argument(
        "--mf6-exe",
        type=Path,
        default=DEFAULT_MF6_EXE,
        help="Path to mf6 executable.",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Write model files and tables without running MODFLOW.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing MODFLOW output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mf6_exe.exists():
        raise FileNotFoundError(f"MODFLOW 6 executable not found: {args.mf6_exe}")

    with contextlib.redirect_stdout(io.StringIO()):
        sim, data = build_simulation(args.mf6_exe)
    write_tables(data)
    if args.plot_only:
        figures = make_figures(data, sim)
        print(f"Figures regenerated from existing outputs: {FIGURE_DIR}")
        for figure in figures:
            print(f" - {figure}")
        return

    sim.write_simulation(silent=True)
    if args.write_only:
        print(f"Wrote MODFLOW 6 input files to {MODEL_WS}")
        return

    run_simulation(sim)
    figures = make_figures(data, sim)
    print(f"MODFLOW 6 model completed: {MODEL_WS}")
    print(f"Figures written to: {FIGURE_DIR}")
    for figure in figures:
        print(f" - {figure}")


if __name__ == "__main__":
    main()
