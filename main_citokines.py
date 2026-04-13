#!/usr/bin/env python3

import re

import numpy as np
import pandas as pd
import pathlib as pl

from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

STIMULI = ["RPMI", "LPS", "Bexcero", "HBVaxpro", "Poly_C"]


def _find(predicate, container, start=0):
    for i, item in enumerate(container[start:], start):
        if predicate(item):
            return i
    return None


def _read_table(filepath, header):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    start = _find(lambda line: line.strip() == header, lines)
    assert start is not None, f"Header '{header}' no encontrado en {filepath}"
    end = _find(lambda line: line.strip() == "", lines, start)
    assert end is not None, f"Fin de tabla no encontrado después de {header}"

    df = pd.read_csv(
        pd.io.common.StringIO("".join(lines[start + 1 : end])),
        index_col=0,
        dtype=str,
    )
    df.index = df.index.astype(str).str.strip()
    return df


def _load_data(filepath):
    count = _read_table(filepath, '"DataType:","Count"')
    result = _read_table(filepath, '"DataType:","Obs Concentration"')
    return count, result


def _join_data(*dfs):
    return pd.concat(dfs, axis=0, verify_integrity=False)


def _remove_calibration_values_location_and_events(df):
    pattern = r"^(Background|Standard)\d*$"
    df = df[~df.index.str.match(pattern, na=False)].copy()
    df = df.drop(columns=["Location"], errors="ignore")
    return df


def _process_replicates(df):
    data_dict = {}
    for cytokine in df.columns:
        long = df[cytokine].reset_index()
        long.columns = ["sample", "value"]
        grouped = long.groupby("sample")["value"].apply(list)

        rows = []
        for sample, vals in grouped.items():
            rep1 = vals[0] if len(vals) > 0 else np.nan
            rep2 = vals[1] if len(vals) > 1 else np.nan
            rows.append({"sample": sample, "replicate_1": rep1, "replicate_2": rep2})

        df_cyt = pd.DataFrame(rows).set_index("sample")
        data_dict[cytokine] = df_cyt
    return data_dict


def _duplicate_replicate_if_any_is_nan(data_dict):
    for _, df in data_dict.items():
        for idx in df.index:
            r1, r2 = df.loc[idx, ["replicate_1", "replicate_2"]]
            if pd.isna(r1) and not pd.isna(r2):
                df.loc[idx, "replicate_1"] = r2
            elif pd.isna(r2) and not pd.isna(r1):
                df.loc[idx, "replicate_2"] = r1
    return data_dict


def _clamp_value(x):
    if isinstance(x, str):
        x = x.strip()
        x = x.replace(",", ".")
        if re.match(r"^<\s*\d", x):
            val = float(re.sub(r"^<\s*", "", x))
            return val * 0.5
        elif re.match(r"^>\s*\d", x):
            val = float(re.sub(r"^>\s*", "", x))
            return val * 2.0
        elif re.match(r"^\*\d*", x):
            return float(re.sub(r"^\*\s*", "", x))
        elif re.match(r"^OOR <", x):
            return 0.5
        elif re.match(r"^OOR >", x):
            return 10000.0
    if isinstance(x, (int, float)) and x <= 0:
        return 0.5
    return x


def _clamp_values(data_dict):
    for _, df in data_dict.items():
        df["replicate_1"] = df["replicate_1"].map(_clamp_value)
        df["replicate_2"] = df["replicate_2"].map(_clamp_value)
    return data_dict


def _convert_to_numeric(data_dict):
    for _, df in data_dict.items():
        df["replicate_1"] = pd.to_numeric(df["replicate_1"], errors="coerce")
        df["replicate_2"] = pd.to_numeric(df["replicate_2"], errors="coerce")
    return data_dict


def _compute_mean_results_by_stimulus(count_dict, result_dict, metadata, stimulus):
    individuals = metadata[metadata["Estimulacion"] == stimulus].copy()

    placebo_codes = (
        individuals[individuals["Vacunado_Placebo"] == "Placebo"]["Codigo_citoquinas"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    vaccinated_codes = (
        individuals[individuals["Vacunado_Placebo"] == "Vacunado"]["Codigo_citoquinas"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    result_placebo_dict = {}
    result_vaccinated_dict = {}

    for cytokine in result_dict.keys():
        common_p = [c for c in placebo_codes if c in count_dict[cytokine].index]
        common_v = [c for c in vaccinated_codes if c in count_dict[cytokine].index]

        count_p = count_dict[cytokine].loc[common_p] if common_p else pd.DataFrame()
        result_p = result_dict[cytokine].loc[common_p] if common_p else pd.DataFrame()
        count_v = count_dict[cytokine].loc[common_v] if common_v else pd.DataFrame()
        result_v = result_dict[cytokine].loc[common_v] if common_v else pd.DataFrame()

        def _compute_means(count_df, result_df):
            if result_df.empty:
                return pd.DataFrame(
                    columns=["replicate_1", "replicate_2", "mean"], index=[]
                )
            res = result_df.copy()
            means = []
            for sample in res.index:
                c1, c2 = count_df.loc[sample, ["replicate_1", "replicate_2"]]
                r1, r2 = res.loc[sample, ["replicate_1", "replicate_2"]]
                valid = []
                if c1 > 35:
                    valid.append(r1)
                if c2 > 35:
                    valid.append(r2)
                means.append(np.nan if not valid else np.mean(valid))
            res["mean"] = means
            return res

        rp = _compute_means(count_p, result_p)
        rv = _compute_means(count_v, result_v)

        rp = rp.dropna(subset=["mean"]).copy()
        rv = rv.dropna(subset=["mean"]).copy()

        result_placebo_dict[cytokine] = rp
        result_vaccinated_dict[cytokine] = rv

    return result_placebo_dict, result_vaccinated_dict


def _compute_paired_vs_rpmi(all_results, metadata):
    paired_records = []

    for group_name in ["Placebo", "Vacunado"]:
        for stim in [s for s in STIMULI if s != "RPMI"]:
            rp_dict = all_results["RPMI"][group_name]
            stim_dict = all_results[stim][group_name]

            assert set(rp_dict.keys()) == set(stim_dict.keys())

            for cytokine in rp_dict.keys():
                df_rpmi = rp_dict[cytokine]
                df_stim = stim_dict[cytokine]

                rpmi_map = {}
                for code, row in df_rpmi.iterrows():
                    patient = metadata.loc[
                        metadata["Codigo_citoquinas"] == code, "ID_paciente"
                    ]
                    if not patient.empty:
                        pat = str(patient.iloc[0])
                        rpmi_map[pat] = row["mean"]

                stim_map = {}
                for code, row in df_stim.iterrows():
                    patient = metadata.loc[
                        metadata["Codigo_citoquinas"] == code, "ID_paciente"
                    ]
                    if not patient.empty:
                        pat = str(patient.iloc[0])
                        stim_map[pat] = row["mean"]

                # Encontrar pacientes que tienen ambos (RPMI y estímulo)
                common_patients = set(rpmi_map.keys()) & set(stim_map.keys())

                paired_rpmi = []
                paired_stim = []
                for pat in common_patients:
                    val_r = rpmi_map[pat]
                    val_s = stim_map[pat]
                    if pd.notna(val_r) and pd.notna(val_s):
                        paired_rpmi.append(val_r)
                        paired_stim.append(val_s)

                n_pairs = len(paired_rpmi)
                if n_pairs < 3:  # mínimo razonable
                    continue

                paired_rpmi = np.array(paired_rpmi)
                paired_stim = np.array(paired_stim)

                # Test estadístico
                _, p_value = wilcoxon(paired_rpmi, paired_stim, alternative="two-sided")

                paired_records.append(
                    {
                        "Group": group_name,
                        "Stimulus": stim,
                        "Cytokine": cytokine,
                        "n_pairs": n_pairs,
                        "p_value": p_value,
                    }
                )

    # Crear DataFrame y ajuste FDR
    paired_df = pd.DataFrame(paired_records)

    if not paired_df.empty:
        for grp in paired_df["Group"].unique():
            mask = paired_df["Group"] == grp
            pvals = paired_df.loc[mask, "p_value"].values
            if len(pvals) > 1:
                paired_df.loc[mask, "p_value_adjusted"] = multipletests(
                    pvals, method="fdr_bh"
                )[1]
            else:
                paired_df.loc[mask, "p_value_adjusted"] = paired_df.loc[mask, "p_value"]

        paired_df = paired_df.sort_values(["Group", "Stimulus", "p_value"])

    # Mostrar resultados
    print("p-values paired (Stimulus vs RPMI):")
    print(paired_df.round(4))

    significant = paired_df[paired_df["p_value"] < 0.05]
    if not significant.empty:
        print(f"{len(significant)} comparaciones significativas (p < 0.05):")
        print(significant.round(4))
    else:
        print("Ninguna comparación alcanza p < 0.05")

    return paired_df


def _compute_group_comparison_on_foldchange(all_results, metadata):
    records = []

    for stim in [s for s in STIMULI if s != "RPMI"]:
        rp_dict = all_results["RPMI"]
        stim_dict = all_results[stim]

        assert set(rp_dict.keys()) == set(stim_dict.keys())

        for cytokine in rp_dict["Placebo"].keys():
            # Obtener datos para Placebo y Vacunado
            df_rpmi_p = rp_dict["Placebo"][cytokine]
            df_stim_p = stim_dict["Placebo"][cytokine]
            df_rpmi_v = rp_dict["Vacunado"][cytokine]
            df_stim_v = stim_dict["Vacunado"][cytokine]

            # Crear mapa paciente -> valor para cada condición
            def _get_patient_map(df):
                patient_map = {}
                for code, row in df.iterrows():
                    pat_series = metadata.loc[
                        metadata["Codigo_citoquinas"] == code, "ID_paciente"
                    ]
                    if not pat_series.empty:
                        pat = str(pat_series.iloc[0])
                        patient_map[pat] = row["mean"]
                return patient_map

            rpmi_p = _get_patient_map(df_rpmi_p)
            stim_p = _get_patient_map(df_stim_p)
            rpmi_v = _get_patient_map(df_rpmi_v)
            stim_v = _get_patient_map(df_stim_v)

            # Calcular fold-change relativo por paciente
            fc_placebo = []
            fc_vaccinated = []

            # Para Placebo
            for pat in set(rpmi_p.keys()) & set(stim_p.keys()):
                r = rpmi_p[pat]
                s = stim_p[pat]
                if pd.notna(r) and pd.notna(s) and r > 0:
                    fc_placebo.append((s - r) / r)

            # Para Vacunado
            for pat in set(rpmi_v.keys()) & set(stim_v.keys()):
                r = rpmi_v[pat]
                s = stim_v[pat]
                if pd.notna(r) and pd.notna(s) and r > 0:
                    fc_vaccinated.append((s - r) / r)

            if len(fc_placebo) < 3 or len(fc_vaccinated) < 3:
                continue

            # Test estadístico
            _, p_value = mannwhitneyu(
                fc_placebo, fc_vaccinated, alternative="two-sided"
            )

            records.append(
                {
                    "Stimulus": stim,
                    "Cytokine": cytokine,
                    "n_placebo": len(fc_placebo),
                    "n_vaccinated": len(fc_vaccinated),
                    "p_value": p_value,
                }
            )

    # Crear DataFrame y ajuste FDR
    df = pd.DataFrame(records)

    if not df.empty:
        pvals = df["p_value"].values
        df["p_value_adjusted"] = multipletests(pvals, method="fdr_bh")[1]

        df = df.sort_values(["Stimulus", "p_value"])

    print("Comparación entre grupos (Vacunado vs Placebo) sobre fold-change relativo:")
    print(df.round(4))

    significant = df[df["p_value"] < 0.05]
    if not significant.empty:
        print(f"{len(significant)} comparaciones significativas (p < 0.05)")
        print(significant.round(4))

    return df


if __name__ == "__main__":
    script_dir = pl.Path(__file__).parent

    count_1, result_1 = _load_data(script_dir / "data" / "citokines_1.csv")
    count_2, result_2 = _load_data(script_dir / "data" / "citokines_2.csv")
    count_3, result_3 = _load_data(script_dir / "data" / "citokines_3.csv")

    count = _join_data(count_1, count_2, count_3)
    result = _join_data(result_1, result_2, result_3)

    count = _remove_calibration_values_location_and_events(count)
    result = _remove_calibration_values_location_and_events(result)

    count_dict = _process_replicates(count)
    result_dict = _process_replicates(result)

    count_dict = _duplicate_replicate_if_any_is_nan(count_dict)
    result_dict = _duplicate_replicate_if_any_is_nan(result_dict)

    count_dict = _clamp_values(count_dict)
    result_dict = _clamp_values(result_dict)

    count_dict = _convert_to_numeric(count_dict)
    result_dict = _convert_to_numeric(result_dict)

    metadata = pd.read_csv(script_dir / "data" / "metadata_v2.csv", index_col=0)

    all_results = {}
    for stimulus in STIMULI:
        rp, rv = _compute_mean_results_by_stimulus(
            count_dict, result_dict, metadata, stimulus
        )
        all_results[stimulus] = {"Placebo": rp, "Vacunado": rv}

    _compute_paired_vs_rpmi(all_results, metadata)
    _compute_group_comparison_on_foldchange(all_results, metadata)
