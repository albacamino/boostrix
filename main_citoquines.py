#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
import subprocess

from pathlib import Path

def _load_data(filepath):
    start = False
    data = []

    with open(filepath) as f:
        for line in f:
            if start:
                if line.strip() == "":
                    break
                data.append(line)
            if line.strip() == '"DataType:","Avg Result"':
                start = True

    # Join lines and read as csv
    csv_text = "".join(data)
    result = pd.read_csv(pd.io.common.StringIO(csv_text), index_col=0)
    return result

def clean_value(x):
    if isinstance(x, str):
        x = x.strip()
        # Valores menores al límite -> mitad del límite
        if re.match(r"^<\s*\d+(\.\d+)?$", x):
            return float(x.replace("<", "").strip()) / 2
        # Valores mayores al límite -> límite
        elif re.match(r"^>\s*\d+(\.\d+)?$", x):
            return float(x.replace(">", "").strip())
    return x

def _join_data(df1, df2, df3, metadata):
    # Add Plaque column
    df1["Plaque"] = 1
    df2["Plaque"] = 2
    df3["Plaque"] = 3

     # Join citoquines dataframe
    citoquines = pd.concat([df1, df2, df3], axis=0)

    # Ensure index are strings
    metadata["Codigo_citoquinas"] = metadata["Codigo_citoquinas"].astype(str).str.strip()
    citoquines.index = citoquines.index.astype(str).str.strip()

    common_index = citoquines.index.intersection(metadata["Codigo_citoquinas"])
  
    # Reorder both dataframes to have the same order
    citoquines = citoquines.loc[common_index]
    metadata_filter = metadata.set_index("Codigo_citoquinas").loc[common_index]

    columns_add = ["Vacunado_Placebo", "código lactante", "Estimulacion"]

    # Add columns to cytokynes dataframe
    for col in columns_add:
        citoquines[col] = metadata_filter[col].values

    citoquines = citoquines.applymap(clean_value)
    citoquines = citoquines.apply(pd.to_numeric, errors="ignore")

    return citoquines
def _compute_pvalues(cytokynes, stimulus, cytokyne):
   
    results = []
    # Filter rows by stimulus
    subset = cytokynes[cytokynes["Estimulacion"] == stimulus]

    # Exract interest columns: cytokyne + group
    areas = subset[[cytokyne, "Vacunado_Placebo"]].dropna()

    vac = areas[areas["Vacunado_Placebo"] == "Vacunado"][cytokyne]
    plac = areas[areas["Vacunado_Placebo"] == "Placebo"][cytokyne]

    # Normality test
    normal_vac = stats.shapiro(vac).pvalue > 0.05
    normal_plac = stats.shapiro(plac).pvalue > 0.05


    # Select test
    if normal_vac and normal_plac:
        test_type = "t-test"
        _, pval = stats.ttest_ind(vac, plac, equal_var=False)
    else:
        test_type = "Mann-Whitney"
        _, pval = stats.mannwhitneyu(vac, plac, alternative="two-sided")

    results.append({
        "Stimulus": stimulus,
        "Cytokine": cytokyne,
        "Test": test_type,
        "p_value": pval
    })

    return results

if __name__== "__main__":

    script_dir = Path(__file__).parent

    df1 = _load_data(script_dir / "datos" / "New_Batch_22.csv")
    df2 = _load_data(script_dir / "datos" / "New_Batch_23.csv")
    df3 = _load_data(script_dir / "datos" / "New_Batch_25.csv")
    metadata = pd.read_csv(script_dir / "datos" / "metadata_boostrix_modificado.csv", index_col=0, dtype=str)

    citoquines = _join_data(df1, df2, df3, metadata)
    citoquines.to_csv(script_dir / "datos" / "results_citoquines_nonNa.csv", index=True)

    input_file = script_dir / "datos" / "results_citoquines_nonNa.csv"
    output_file = script_dir / "datos" / "citoquinas_combat.csv"
    
    subprocess.run(
    ["Rscript", "correct_batch_effect.R", str(input_file), str(output_file)],
    check=True
    )
    
    citoquines = pd.read_csv(output_file)
    for stimulus in ['RPMI' 'LPS' 'Bexcero' 'HBVaxpro' 'Poly_C']:
        for cytokyne in  ["IFNalpha","IFNgamma","IL10","IL12p70","IL1beta","IL21","IL6","IL8","IP10","MCP1","MIP1alpha","RANTES","TNFalpha"]:
            print(_compute_pvalues(cytokynes, stimulus, cytokyne))
