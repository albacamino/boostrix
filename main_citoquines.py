#!/usr/bin/env python3

import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import pathlib as pl

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

TIMES_STDDEV = 1
STIMULI = ["RPMI", "LPS", "Bexcero", "HBVaxpro", "Poly_C"]


def _find(predicate, container, start=0):
    for i, item in enumerate(container[start:], start):
        if predicate(item):
            return i
    return None


def _read_table(filepath, header):
    with open(filepath) as f:
        lines = f.readlines()

    start = _find(lambda line: line.strip() == header, lines)
    assert start is not None
    end = _find(lambda line: line.strip() == "", lines, start)
    assert end is not None

    return pd.read_csv(
        pd.io.common.StringIO("".join(lines[start + 1 : end])), index_col=1
    )


def _load_data(filepath):
    count = _read_table(filepath, '"DataType:","Count"')
    result = _read_table(filepath, '"DataType:","Result"')
    return count, result


def _join_data(df1, df2, df3):
    return pd.concat([df1, df2, df3])


def _remove_calibration_values_location_and_events(df):
    pattern = r"^(Background|Standard)\d+$"
    filtered_df = df[~df.index.str.match(pattern)]
    filtered_df = filtered_df.drop(
        columns=["Location", "Total Events"], errors="ignore"
    )
    return filtered_df


def _remove_nans_in_both_replicates(df):
    filtered_data = {}

    for cytokine in df.columns.tolist():
        filtered_indices = []
        for sample in df.index.unique():
            replicates = df.loc[sample, cytokine]
            if pd.isna(replicates).all():
                continue
            filtered_indices.append(sample)
        cytokine_rows = []
        for sample in filtered_indices:
            replicates = df.loc[sample, cytokine]
            cytokine_rows.append([sample, replicates.iloc[0], replicates.iloc[1]])
        filtered_data[cytokine] = pd.DataFrame(
            cytokine_rows, columns=[cytokine, "replicate_1", "replicate_2"]
        ).set_index(cytokine)

    return filtered_data


def _duplicate_replicate_if_any_is_nan(df_dict):
    updated_data = {}

    for cytokine, df in df_dict.items():
        updated_rows = []
        for sample in df.index:
            replicate_1 = df.loc[sample, "replicate_1"]
            replicate_2 = df.loc[sample, "replicate_2"]

            if pd.isna(replicate_1) and not pd.isna(replicate_2):
                updated_rows.append([sample, replicate_2, replicate_2])
            elif pd.isna(replicate_2) and not pd.isna(replicate_1):
                updated_rows.append([sample, replicate_1, replicate_1])
            else:
                updated_rows.append([sample, replicate_1, replicate_2])

        updated_df = pd.DataFrame(
            updated_rows, columns=[cytokine, "replicate_1", "replicate_2"]
        ).set_index(cytokine)
        updated_data[cytokine] = updated_df

    return updated_data


def _clamp_value(x):
    if isinstance(x, str):
        x = x.strip()
        if re.match(r"^<\s*\d+(\.\d+)?$", x):
            return float(x.replace("<", "").strip())
        elif re.match(r"^>\s*\d+(\.\d+)?$", x):
            return float(x.replace(">", "").strip())
    return x


def _clamp_values(df_dict):
    clamped_data = {}

    for cytokine, df in df_dict.items():
        clamped_rows = []
        for sample in df.index:
            replicate_1 = df.loc[sample, "replicate_1"]
            replicate_2 = df.loc[sample, "replicate_2"]

            replicate_1 = _clamp_value(replicate_1)
            replicate_2 = _clamp_value(replicate_2)

            clamped_rows.append([sample, replicate_1, replicate_2])

        clamped_df = pd.DataFrame(
            clamped_rows, columns=[cytokine, "replicate_1", "replicate_2"]
        ).set_index(cytokine)
        clamped_data[cytokine] = clamped_df

    return clamped_data


def _convert_to_numeric(df_dict):
    for cytokine in df_dict.keys():
        df_dict[cytokine]["replicate_1"] = pd.to_numeric(
            df_dict[cytokine]["replicate_1"]
        )
        df_dict[cytokine]["replicate_2"] = pd.to_numeric(
            df_dict[cytokine]["replicate_2"]
        )
    return df_dict


def _get_common_codes(count, result, codes, cytokine):
    common_codes = set(codes)
    common_codes = common_codes.intersection(
        set(count[cytokine].index.astype(str).str.strip())
    )
    common_codes = common_codes.intersection(
        set(result[cytokine].index.astype(str).str.strip())
    )
    return sorted(common_codes)


def _filter_data_by_common_codes(count, result, common_codes, cytokine):
    return count[cytokine].loc[
        count[cytokine].index.astype(str).str.strip().isin(common_codes)
    ], result[cytokine].loc[
        result[cytokine].index.astype(str).str.strip().isin(common_codes)
    ]


def _compute_means(
    count_placebo, result_placebo, count_vaccinated, result_vaccinated, cytokine
):
    def _compute_mean(row_count, row_result):
        rep1_count = row_count["replicate_1"]
        rep2_count = row_count["replicate_2"]
        rep1_result = row_result["replicate_1"]
        rep2_result = row_result["replicate_2"]

        valid_results = []
        if rep1_count > 35:
            valid_results.append(rep1_result)
        if rep2_count > 35:
            valid_results.append(rep2_result)

        return (
            float("nan")
            if not valid_results
            else sum(valid_results) / len(valid_results)
        )

    def _compute_for_group(count_dict, result_dict, cytokine):
        result_df = result_dict.copy()
        means = []
        for sample in count_dict.index:
            row_count = count_dict.loc[sample]
            row_result = result_dict.loc[sample]
            mean_value = _compute_mean(row_count, row_result)
            means.append(mean_value)
        result_df["mean"] = means
        return result_df

    return _compute_for_group(
        count_placebo, result_placebo, cytokine
    ), _compute_for_group(count_vaccinated, result_vaccinated, cytokine)


def _reduce_nas(result_placebo, result_vaccinated):
    def _fill_means(df):
        new_df = df.copy()

        valid_means = new_df["mean"].dropna()
        mean_of_valids = valid_means.mean()
        std_of_valids = valid_means.std()
        lower = mean_of_valids - TIMES_STDDEV * std_of_valids
        upper = mean_of_valids + TIMES_STDDEV * std_of_valids

        for sample in new_df.index:
            if pd.isna(new_df.loc[sample, "mean"]):
                rep1 = new_df.loc[sample, "replicate_1"]
                rep2 = new_df.loc[sample, "replicate_2"]

                in1 = lower <= rep1 <= upper
                in2 = lower <= rep2 <= upper

                if in1 and in2:
                    new_df.loc[sample, "mean"] = (rep1 + rep2) / 2
                elif in1:
                    new_df.loc[sample, "mean"] = rep1
                elif in2:
                    new_df.loc[sample, "mean"] = rep2

        return new_df

    return _fill_means(result_placebo), _fill_means(result_vaccinated)


def _remove_nas(result_placebo, result_vaccinated):
    return result_placebo.dropna(subset=["mean"]).copy(), result_vaccinated.dropna(
        subset=["mean"]
    ).copy()


def _compute_mean_results_by_stimulus(count, result, metadata, stimulus):
    individuals = metadata[metadata["Estimulacion"] == stimulus]

    placebo = individuals[individuals["Vacunado_Placebo"] == "Placebo"]
    vaccinated = individuals[individuals["Vacunado_Placebo"] == "Vacunado"]

    placebo_codes = placebo["Codigo_citoquinas"].astype(str).str.strip().tolist()
    vaccinated_codes = vaccinated["Codigo_citoquinas"].astype(str).str.strip().tolist()

    ret_placebo, ret_vaccinated = {}, {}

    for cytokine, df in result.items():
        common_placebo_codes = _get_common_codes(count, result, placebo_codes, cytokine)
        common_vaccinated_codes = _get_common_codes(
            count, result, vaccinated_codes, cytokine
        )

        count_placebo, result_placebo = _filter_data_by_common_codes(
            count, result, common_placebo_codes, cytokine
        )
        count_vaccinated, result_vaccinated = _filter_data_by_common_codes(
            count, result, common_vaccinated_codes, cytokine
        )

        result_placebo, result_vaccinated = _compute_means(
            count_placebo, result_placebo, count_vaccinated, result_vaccinated, cytokine
        )
        result_placebo, result_vaccinated = _reduce_nas(
            result_placebo, result_vaccinated
        )
        result_placebo, result_vaccinated = _remove_nas(
            result_placebo, result_vaccinated
        )

        ret_placebo[cytokine] = result_placebo
        ret_vaccinated[cytokine] = result_vaccinated

    return ret_placebo, ret_vaccinated


def _compute_pvalues(result_placebo, result_vaccinated):
    pvalues = {}

    for cytokine in result_placebo.keys():
        data_placebo = result_placebo[cytokine]["mean"].values
        data_vaccinated = result_vaccinated[cytokine]["mean"].values

        if len(data_placebo) < 2 or len(data_vaccinated) < 2:
            pvalues[cytokine] = float("nan")
            continue

        pvalues[cytokine] = mannwhitneyu(
            data_placebo, data_vaccinated, alternative="two-sided"
        )[1]

    pvals = list(pvalues.values())
    adjusted_pvalue = multipletests(pvals, method="bonferroni")[1]

    return pd.DataFrame(
        {
            "Cytokine": list(pvalues.keys()),
            "p-value": pvals,
            "p-value_adjusted": adjusted_pvalue,
        },
    )


if __name__ == "__main__":
    script_dir = pl.Path(__file__).parent

    count_1, result_1 = _load_data(script_dir / "datos" / "New_Batch_22.csv")
    count_2, result_2 = _load_data(script_dir / "datos" / "New_Batch_23.csv")
    count_3, result_3 = _load_data(script_dir / "datos" / "New_Batch_25.csv")

    count = _join_data(count_1, count_2, count_3)
    result = _join_data(result_1, result_2, result_3)

    count = _remove_calibration_values_location_and_events(count)
    result = _remove_calibration_values_location_and_events(result)

    count = _remove_nans_in_both_replicates(count)
    result = _remove_nans_in_both_replicates(result)

    count = _duplicate_replicate_if_any_is_nan(count)
    result = _duplicate_replicate_if_any_is_nan(result)

    count = _clamp_values(count)
    result = _clamp_values(result)

    count = _convert_to_numeric(count)
    result = _convert_to_numeric(result)

    metadata = pd.read_csv(script_dir / "datos" / "metadata_boostrix_modificado.csv", index_col=0)

    for stimulus in STIMULI:
        result_placebo, result_vaccinated = _compute_mean_results_by_stimulus(
            count, result, metadata, stimulus
        )
        print(result_placebo)
        print(result_vaccinated)

        pvalues = _compute_pvalues(result_placebo, result_vaccinated)
        significant = pvalues[pvalues["p-value"] < 0.05]

        if not significant.empty:
            print(f"Significant cytokines for stimulus {stimulus}:")
            print(significant)
            print("\n\n", end="")

        
        folder = f"efectos_heterologos/boxplots_citoquinas"
        os.makedirs(folder, exist_ok=True)

        # Crear diccionario {citoquina: pvalue} para escribirlo en cada figura
        pvals_dict = dict(zip(pvalues["Cytokine"], pvalues["p-value"]))
  
        for cytokine in result_placebo.keys():

            df_p = result_placebo[cytokine]     # placebo dataframe
            df_v = result_vaccinated[cytokine]  # vacunado dataframe

            values_placebo = df_p["mean"].values
            values_vaccinated = df_v["mean"].values

            plt.figure(figsize=(6, 4))
            plt.boxplot(
                [values_placebo, values_vaccinated],
                labels=["Placebo", "Vaccinated"],
                showmeans=True
            )

            plt.ylabel("Concentration")

            # Escribir el p-value encima del plot
            pval = pvals_dict[cytokine]
            if pd.notna(pval):
                plt.text(
                    1.5,                 # posición x centrada
                    max(max(values_placebo), max(values_vaccinated)) * 1.05,
                    f"p = {pval:.3e}",
                    ha="center"
                )

            plt.tight_layout()
            plt.savefig(f"{folder}/{cytokine}_{stimulus}.png", dpi=200)
            plt.close()