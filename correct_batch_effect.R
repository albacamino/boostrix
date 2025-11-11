#!/usr/bin/env Rscript
list.of.packages <- c("sva", "readr", "dpylr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

lapply(list.of.packages, library, character.only = TRUE)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[1]
output_file <- args[2]

df <- read_csv(input_file)

# Asegurar que los factores están bien definidos
df$Plaque <- as.factor(df$Plaque)
df$Vacunado_Placebo <- as.factor(df$Vacunado_Placebo)
df$Estimulacion <- as.factor(df$Estimulacion)
expr <- df %>%
  select(-Vacunado_Placebo, -Estimulacion, -Plaque, -Sample, -`código lactante`)

# Modelo biológico
mod <- model.matrix(~ Vacunado_Placebo + Estimulacion, data = df)

# Aplicar ComBat
combat_data <- ComBat(
  dat = t(as.matrix(expr)),
  batch = df$Plaque,
  mod = mod,
  par.prior = TRUE,
  prior.plots = FALSE
)

combat_corrected <- as.data.frame(t(combat_data))

# Restaurar nombres de columnas y filas
colnames(combat_corrected) <- colnames(expr)
rownames(combat_corrected) <- df$Sample

# --- Añadir columnas originales ---
combat_corrected$Sample <- df$Sample
combat_corrected$Plaque <- df$Plaque
combat_corrected$Vacunado_Placebo <- df$Vacunado_Placebo
combat_corrected$Estimulacion <- df$Estimulacion
combat_corrected$`código lactante` <- df$`código lactante`

combat_corrected <- combat_corrected %>%
  select(Sample, Plaque, Vacunado_Placebo, Estimulacion, `código lactante`, everything())


write.csv(combat_corrected,output_file, col.names = TRUE, row.names = FALSE)
