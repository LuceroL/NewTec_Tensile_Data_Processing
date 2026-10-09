import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tensile Data Analysis", layout="wide")
st.title("Tensile Stress–Strain Analysis")
st.write("Upload tensile-test CSV files, configure the analysis and plots, then export analyzed data, metrics, and figures.")


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def figure_png(fig, dpi=300):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def optional_number(label, key, default=""):
    raw = st.text_input(label, value=default, key=key)
    if not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        st.warning(f"{label}: enter a numeric value or leave blank for automatic scaling.")
        return None


def axis_controls(prefix, defaults):
    cols = st.columns(4)
    labels = ["X minimum", "X maximum", "Y minimum", "Y maximum"]
    suffixes = ["xmin", "xmax", "ymin", "ymax"]
    values = []
    for col, label, suffix, default in zip(cols, labels, suffixes, defaults):
        with col:
            values.append(optional_number(label, f"{prefix}_{suffix}", default))
    return values


def apply_axis_limits(ax, limits):
    xmin, xmax, ymin, ymax = limits
    if xmin is not None and xmax is not None and xmin >= xmax:
        st.warning(f"{ax.get_title() or 'Plot'}: X minimum must be less than X maximum; automatic X limits retained.")
    else:
        if xmin is not None:
            ax.set_xlim(left=xmin)
        if xmax is not None:
            ax.set_xlim(right=xmax)
    if ymin is not None and ymax is not None and ymin >= ymax:
        st.warning(f"{ax.get_title() or 'Plot'}: Y minimum must be less than Y maximum; automatic Y limits retained.")
    else:
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)


def add_right_legend(fig, ax, font_size):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=font_size)
        fig.subplots_adjust(right=0.68)


def analyze_frame(frame, strain_column, stress_column, difference_points):
    data = pd.DataFrame({
        "Engineering Strain (%)": pd.to_numeric(frame[strain_column], errors="coerce"),
        "Engineering Stress (MPa)": pd.to_numeric(frame[stress_column], errors="coerce"),
    }).dropna()
    data["Engineering Strain"] = data["Engineering Strain (%)"] / 100.0
    valid = (
        np.isfinite(data["Engineering Strain"])
        & np.isfinite(data["Engineering Stress (MPa)"])
        & (data["Engineering Strain"] > -1)
    )
    data = data.loc[valid].sort_values("Engineering Strain")
    data = data.drop_duplicates(subset="Engineering Strain", keep="first").reset_index(drop=True)
    if len(data) < 5:
        return None, {"found": False, "note": "Fewer than five valid unique data points.", "strain": np.nan, "strain_pct": np.nan, "stress": np.nan}

    data["True Strain"] = np.log1p(data["Engineering Strain"])
    data["True Strain (%)"] = data["True Strain"] * 100.0
    data["True Stress (MPa)"] = data["Engineering Stress (MPa)"] * (1 + data["Engineering Strain"])
    data["Hardening Rate (MPa)"] = np.nan
    data["Hardening Rate Strain"] = np.nan
    data["Hardening Rate True Stress (MPa)"] = np.nan
    data["Considere Difference (MPa)"] = np.nan
    data["Estimated Necking Point"] = False

    n = len(data)
    dp = int(difference_points)
    neck = {"found": False, "note": "", "strain": np.nan, "strain_pct": np.nan, "stress": np.nan}
    if n < dp:
        neck["note"] = f"Not calculated: {n} points available, but the window is {dp} points."
        return data, neck

    half_window = dp // 2
    true_strain = data["True Strain"].to_numpy()
    true_stress = data["True Stress (MPa)"].to_numpy()
    delta_strain = true_strain[dp - 1:] - true_strain[:n - dp + 1]
    delta_stress = true_stress[dp - 1:] - true_stress[:n - dp + 1]
    valid_diff = np.isfinite(delta_strain) & np.isfinite(delta_stress) & (delta_strain > 0)
    hardening_rate = np.full(delta_strain.shape, np.nan, dtype=float)
    hardening_rate[valid_diff] = delta_stress[valid_diff] / delta_strain[valid_diff]
    midpoint_strain = true_strain[half_window:n - half_window]
    midpoint_stress = np.interp(midpoint_strain, true_strain, true_stress)

    row_indices = np.arange(half_window, n - half_window)
    data.loc[row_indices, "Hardening Rate (MPa)"] = hardening_rate
    data.loc[row_indices, "Hardening Rate Strain"] = midpoint_strain
    data.loc[row_indices, "Hardening Rate True Stress (MPa)"] = midpoint_stress
    difference = hardening_rate - midpoint_stress
    data.loc[row_indices, "Considere Difference (MPa)"] = difference

    valid_indices = np.where(np.isfinite(difference))[0]
    if len(valid_indices) > 1:
        valid_difference = difference[valid_indices]
        crossings = np.where((valid_difference[:-1] > 0) & (valid_difference[1:] <= 0))[0]
        if len(crossings):
            j = valid_indices[crossings[0]]
            denominator = difference[j] - difference[j + 1]
            if denominator != 0 and np.isfinite(denominator):
                fraction = difference[j] / denominator
                neck_strain = midpoint_strain[j] + fraction * (midpoint_strain[j + 1] - midpoint_strain[j])
                neck_stress = midpoint_stress[j] + fraction * (midpoint_stress[j + 1] - midpoint_stress[j])
                neck.update({
                    "found": True,
                    "note": "First positive-to-nonpositive Considère crossing.",
                    "strain": float(neck_strain),
                    "strain_pct": float(neck_strain * 100),
                    "stress": float(neck_stress),
                })
                nearest_index = int(np.argmin(np.abs(true_strain - neck_strain)))
                data.loc[nearest_index, "Estimated Necking Point"] = True
            else:
                neck["note"] = "Crossing could not be interpolated."
        else:
            neck["note"] = "No positive-to-nonpositive Considère crossing found."
    else:
        neck["note"] = "No valid finite-difference values available."
    return data, neck


with st.sidebar:
    st.header("Input and settings")
    uploads = st.file_uploader("Upload CSV file(s)", type=["csv", "txt"], accept_multiple_files=True)
    difference_points = st.number_input(
        "Difference points (odd integer)", min_value=3, max_value=100001,
        value=501, step=2,
        help="Centered finite-difference window size. Larger windows usually reduce noise but can obscure local changes."
    )
    if int(difference_points) % 2 == 0:
        st.error("Difference points must be an odd integer.")
        st.stop()
    plot_width = st.slider("Plot width (inches)", 7, 16, 10)
    plot_height = st.slider("Plot height (inches)", 4, 12, 7)
    legend_font_size = st.slider("Legend font size", 6, 14, 8)
    export_dpi = st.selectbox("PNG export resolution", [150, 300, 600], index=1)
    show_grid = st.checkbox("Show grid", value=True)

if not uploads:
    st.info("Upload one or more CSV files in the sidebar to begin.")
    st.markdown("Expected columns are **Strain (%)** and **Stress (MPa)**. If your column names differ, map them below after uploading.")
    st.stop()

frames = {}
read_errors = []
for upload in uploads:
    try:
        frame = pd.read_csv(upload, sep=None, engine="python", on_bad_lines="warn")
        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.shape[1] < 2:
            read_errors.append(f"{upload.name}: fewer than two columns found.")
        else:
            frames[upload.name] = frame
    except Exception as exc:
        read_errors.append(f"{upload.name}: {exc}")
for error in read_errors:
    st.error(f"Could not read file — {error}")
if not frames:
    st.stop()

first_columns = list(next(iter(frames.values())).columns)
strain_default = first_columns.index("Strain (%)") if "Strain (%)" in first_columns else 0
stress_default = first_columns.index("Stress (MPa)") if "Stress (MPa)" in first_columns else min(1, len(first_columns) - 1)
st.subheader("Column mapping")
col1, col2 = st.columns(2)
with col1:
    strain_column = st.selectbox("Engineering strain (%) column", first_columns, index=strain_default)
with col2:
    stress_column = st.selectbox("Engineering stress (MPa) column", first_columns, index=stress_default)

results = []
warnings = []
colors = plt.cm.tab10.colors
for index, (filename, frame) in enumerate(frames.items()):
    if strain_column not in frame.columns or stress_column not in frame.columns:
        warnings.append(f"{filename}: selected columns are not present in this file; skipped.")
        continue
    data, neck = analyze_frame(frame, strain_column, stress_column, int(difference_points))
    if data is None:
        warnings.append(f"{filename}: {neck['note']} Skipped.")
        continue
    data.insert(0, "File", filename)
    if not neck["found"]:
        warnings.append(f"{filename}: {neck['note']}")
    results.append({
        "filename": filename,
        "label": Path(filename).stem,
        "data": data,
        "neck": neck,
        "color": colors[index % len(colors)],
    })

if warnings:
    with st.expander("Data warnings", expanded=True):
        for warning in warnings:
            st.warning(warning)
if not results:
    st.error("No files could be analyzed. Check the files and column mapping.")
    st.stop()

metric_rows = []
for result in results:
    data, neck = result["data"], result["neck"]
    metric_rows.append({
        "File": result["filename"],
        "Valid unique data points": len(data),
        "Difference points": int(difference_points),
        "Maximum engineering stress (MPa)": data["Engineering Stress (MPa)"].max(),
        "Maximum true stress (MPa)": data["True Stress (MPa)"].max(),
        "Estimated necking found": neck["found"],
        "Estimated necking true strain (decimal)": neck["strain"],
        "Estimated necking true strain (%)": neck["strain_pct"],
        "Estimated necking true stress (MPa)": neck["stress"],
        "Necking note": neck["note"],
    })
metrics = pd.DataFrame(metric_rows)
st.subheader("Analysis metrics")
st.dataframe(metrics, use_container_width=True, hide_index=True)
st.download_button("Download metrics CSV", to_csv_bytes(metrics), "tensile_analysis_metrics.csv", "text/csv")

st.subheader("Plot axes and labels")
with st.expander("Engineering stress–strain", expanded=True):
    eng_xlabel = st.text_input("X-axis label", "Engineering Strain (%)", key="eng_xlabel")
    eng_ylabel = st.text_input("Y-axis label", "Engineering Stress (MPa)", key="eng_ylabel")
    eng_limits = axis_controls("eng", ("0", "", "0", ""))
with st.expander("True stress–strain", expanded=True):
    true_xlabel = st.text_input("X-axis label", "True Strain (%)", key="true_xlabel")
    true_ylabel = st.text_input("Y-axis label", "True Stress (MPa)", key="true_ylabel")
    true_limits = axis_controls("true", ("0", "", "0", ""))
with st.expander("True stress and hardening rate", expanded=True):
    hard_xlabel = st.text_input("X-axis label", "True Strain (decimal)", key="hard_xlabel")
    hard_ylabel = st.text_input("Y-axis label", "Stress (MPa) / Hardening Rate (MPa)", key="hard_ylabel")
    hard_limits = axis_controls("hard", ("0.01", "", "0", ""))

fig_eng, ax_eng = plt.subplots(figsize=(plot_width, plot_height))
fig_true, ax_true = plt.subplots(figsize=(plot_width, plot_height))
fig_hard, ax_hard = plt.subplots(figsize=(plot_width, plot_height))
for result in results:
    data, label, color, neck = result["data"], result["label"], result["color"], result["neck"]
    ax_eng.plot(data["Engineering Strain (%)"], data["Engineering Stress (MPa)"], color=color, linewidth=1.5, label=label)
    ax_true.plot(data["True Strain (%)"], data["True Stress (MPa)"], color=color, linewidth=1.5, label=label)
    ax_hard.plot(data["True Strain"], data["True Stress (MPa)"], color=color, linewidth=1.5, label=f"{label} True Stress")
    hardening_data = data.dropna(subset=["Hardening Rate Strain", "Hardening Rate (MPa)"])
    if not hardening_data.empty:
        ax_hard.plot(hardening_data["Hardening Rate Strain"], hardening_data["Hardening Rate (MPa)"], color=color, linestyle="--", linewidth=1.2, label=f"{label} Hardening Rate")
    if neck["found"]:
        ax_hard.scatter(neck["strain"], neck["stress"], color=color, edgecolor="black", zorder=5, label=f"{label} Estimated Necking")

for axis in (ax_eng, ax_true, ax_hard):
    if show_grid:
        axis.grid(True, alpha=0.3)
ax_eng.set_xlabel(eng_xlabel)
ax_eng.set_ylabel(eng_ylabel)
apply_axis_limits(ax_eng, eng_limits)
add_right_legend(fig_eng, ax_eng, legend_font_size)
fig_eng.tight_layout()
ax_true.set_xlabel(true_xlabel)
ax_true.set_ylabel(true_ylabel)
apply_axis_limits(ax_true, true_limits)
add_right_legend(fig_true, ax_true, legend_font_size)
fig_true.tight_layout()
ax_hard.set_xlabel(hard_xlabel)
ax_hard.set_ylabel(hard_ylabel)
apply_axis_limits(ax_hard, hard_limits)
add_right_legend(fig_hard, ax_hard, legend_font_size)
fig_hard.tight_layout()

st.subheader("Plots and PNG exports")
tab1, tab2, tab3 = st.tabs(["Engineering stress–strain", "True stress–strain", "Hardening rate / Considère"])
with tab1:
    st.pyplot(fig_eng, use_container_width=True)
    st.download_button("Download engineering plot PNG", figure_png(fig_eng, export_dpi), "Engineering_Stress_Strain.png", "image/png")
with tab2:
    st.pyplot(fig_true, use_container_width=True)
    st.download_button("Download true stress plot PNG", figure_png(fig_true, export_dpi), "True_Stress_Strain.png", "image/png")
with tab3:
    st.pyplot(fig_hard, use_container_width=True)
    st.download_button("Download hardening-rate plot PNG", figure_png(fig_hard, export_dpi), "True_Stress_Hardening_Rate.png", "image/png")

st.subheader("Analyzed data exports")
st.caption("The output contains engineering and true values, hardening rate, the Considère difference, and a flag for the nearest data row to the estimated necking point.")
combined_data = pd.concat([result["data"] for result in results], ignore_index=True)
st.download_button("Download all analyzed data as CSV", to_csv_bytes(combined_data), "all_tensile_data_analyzed.csv", "text/csv")
for result in results:
    safe_name = Path(result["filename"]).stem
    with st.expander(f"Preview and export: {result['filename']}"):
        st.dataframe(result["data"].head(20), use_container_width=True, hide_index=True)
        st.download_button(f"Download {safe_name} analyzed CSV", to_csv_bytes(result["data"]), f"{safe_name}_analyzed.csv", "text/csv", key=f"csv_{safe_name}")

st.subheader("Complete results package")
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("tensile_analysis_metrics.csv", to_csv_bytes(metrics))
    archive.writestr("all_tensile_data_analyzed.csv", to_csv_bytes(combined_data))
    archive.writestr("Engineering_Stress_Strain.png", figure_png(fig_eng, export_dpi))
    archive.writestr("True_Stress_Strain.png", figure_png(fig_true, export_dpi))
    archive.writestr("True_Stress_Hardening_Rate.png", figure_png(fig_hard, export_dpi))
    for result in results:
        archive.writestr(f"individual_csvs/{Path(result['filename']).stem}_analyzed.csv", to_csv_bytes(result["data"]))
zip_buffer.seek(0)
st.download_button("Download ZIP: CSVs + plots + metrics", zip_buffer.getvalue(), "tensile_analysis_results.zip", "application/zip", type="primary")

st.caption("Scientific note: engineering-to-true conversion assumes uniform deformation and is generally not valid after necking. The Considère intersection is an estimate that depends on the selected finite-difference window; compare results across window sizes before interpreting it.")

# Close Matplotlib figures after Streamlit has rendered and serialized the export data.
plt.close(fig_eng)
plt.close(fig_true)
plt.close(fig_hard)
