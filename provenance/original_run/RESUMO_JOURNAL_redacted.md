# Journal — EEG × demografia em duas coortes — 2026-11-20T17:17:35

**What Does Resting-State EEG Add to Demographics in Alzheimer's Disease? Incremental Value, Confounding, and Transportability Across Two Cohorts**

Modo: completo | SPEC fcdd59491d8d1aa8, congelada em 2026-11-20T10:33:16 | rótulos gregos lidos pela primeira vez em 2026-11-20T10:33:16 | duração 3 min

ds004504 v1.0.9 (doi:10.18112/openneuro.ds004504.v1.0.9); DOI na pasta: doi:10.18112/openneuro.ds004504.v1.0.9; 93 de 93 arquivos conferidos por SHA-256

python 3.11.9 | numpy 2.4.6 | pandas 3.0.6 | scipy 1.17.1 | scikit-learn 1.9.1 | xgboost 3.2.0 | mne 1.13.2 | matplotlib 3.11.2

## Leituras pela regra da SPEC

| Pergunta | Leitura |
|---|---|
| RQ1a | não demonstrado |
| RQ1b | acrescenta |
| RQ1c | heterogêneo ou incerto |
| RQ1 | acrescenta só no grego |
| RQ2a | sobrevive em parte |
| RQ2b | sobrevive |
| RQ3a | transporta |
| RQ3b | transporta |

## Resultados

| RQ | Item | Valor | Onde entra | Nota |
|---|---|---|---|---|
| Métodos | Correspondência BioSemi → 10-20 | posições do próprio arquivo; ângulo médio 3.5°, máximo 6.9° | Methods; Suppl. Table S1 |  |
| Métodos | BrainLat: participantes com EEG analisável | 67 de 81 (35 AD, 32 HC); 7425 épocas | Methods; Table 1 |  |
| Métodos | Grego: arquivos com marcas 'boundary' | 56 de 65; mediana 4 por arquivo | Methods (se relevante) |  |
| Métodos | Grego: participantes com EEG analisável | 65 de 65 (36 AD, 29 HC); 13282 épocas | Methods; Table 1 |  |
| Métodos | BrainLat: sem EEG analisável | 14: [subject IDs redacted] | Methods |  |
| Métodos | Mudança de domínio (controles, \|SMD\| > 0,8 entre coortes) | Rel_Theta_mean, Rel_Gamma_mean, Spectral_Entropy, PLV_Theta_mean, PLV_Alpha_mean, Corr_Broadband_mean | Results (RQ3); Suppl. Table S2 |  |
| Base | BrainLat: AUC do EEG (RF, LOSO) | 0.681 [0.545, 0.804] | Results; Table 2 |  |
| Base | BrainLat: métricas com limiar 0.5 (RF, LOSO) | sens 0.714, espec 0.562, acurácia balanceada 0.638, acurácia 0.642, F1 0.676, MCC 0.280 | Table 2 | descritivo; comparável aos estudos do ds004504 |
| Base | Grego: AUC do EEG (RF, LOSO) | 0.869 [0.767, 0.947] | Results; Table 2 |  |
| Base | Grego: métricas com limiar 0.5 (RF, LOSO) | sens 0.722, espec 0.966, acurácia balanceada 0.844, acurácia 0.831, F1 0.825, MCC 0.694 | Table 2 | descritivo; comparável aos estudos do ds004504 |
| RQ1a | BrainLat: escore EEG além de idade, sexo e escolaridade | χ²(1) = 3.59; p (bootstrap nulo) = 0.094 [106 réplicas]; p ingênuo = 0.058; OR/DP = 2.00 [0.98, 4.74] (IC perfilado, descritivo); MLE; n = 67 | Results; Table 2 | leitura: não demonstrado |
| RQ1a | BrainLat: verificação com sítio (idade, sexo, escolaridade, sítio) | OR/DP = 2.06 [0.99, 5.10]; p = 0.055; MLE | Results (uma frase) | verificação; fora da regra de leitura |
| RQ1a | BrainLat: AUC Age | 0.699 [0.571, 0.816] | Table 2 | logística, leave-one-out |
| RQ1a | BrainLat: AUC Age + sex | 0.682 [0.552, 0.802] | Table 2 | logística, leave-one-out |
| RQ1a | BrainLat: AUC Age + sex + education | 0.862 [0.761, 0.944] | Table 2 | logística, leave-one-out |
| RQ1a | BrainLat: AUC EEG score (RF, LOSO) | 0.681 [0.545, 0.804] | Table 2 | LOSO |
| RQ1a | BrainLat: AUC Demographics + EEG score | 0.876 [0.780, 0.949] | Table 2 | logística, leave-one-out |
| RQ1a | BrainLat: ΔAUC Demographics + EEG score − demographics | 0.014 [-0.023, 0.060] | Results | descritivo, sem teste |
| RQ1a | BrainLat: ΔAUC EEG score (RF, LOSO) − demographics | -0.180 [-0.342, -0.023] | Results | descritivo, sem teste |
| RQ1b | Referência para o E-value (BrainLat) | OR/DP do EEG 1.94 (idade + sexo) → 2.00 (+ escolaridade); escolaridade OR/DP 0.19 para AD | Discussion | o quanto a escolaridade mudou o efeito onde foi medida |
| RQ1b | Grego: escore EEG além de idade e sexo (unilateral) | χ²(1) = 31.44; p unilateral (bootstrap nulo) = 0.005 [199 réplicas]; p unilateral ingênuo = < 0.001; OR/DP = 18.63 [4.82, 134.35] (IC perfilado, descritivo); MLE; n = 65 | Results; Table 2 | leitura: acrescenta |
| RQ1b | Grego: E-value (escolaridade não medida) | 8.10 (OR); 3.82 (limite do IC) | Results; Discussion |  |
| RQ1b | Grego: AUC Age | 0.519 [0.378, 0.667] | Table 2 | logística, leave-one-out |
| RQ1b | Grego: AUC Age + sex | 0.618 [0.486, 0.757] | Table 2 | logística, leave-one-out |
| RQ1b | Grego: AUC EEG score (RF, LOSO) | 0.869 [0.767, 0.947] | Table 2 | LOSO |
| RQ1b | Grego: AUC Demographics + EEG score | 0.842 [0.734, 0.930] | Table 2 | logística, leave-one-out |
| RQ1b | Grego: ΔAUC Demographics + EEG score − demographics | 0.224 [0.070, 0.375] | Results | descritivo, sem teste |
| RQ1b | Grego: ΔAUC EEG score (RF, LOSO) − demographics | 0.251 [0.087, 0.416] | Results | descritivo, sem teste |
| RQ1c | Conjunto: escore EEG além das covariáveis de cada coorte (efeito comum) | OR/DP = 4.46 [2.36, 9.66]; χ²(1) = 27.25, p = < 0.001; MLE; n = 132 | Results; Fig. 2 |  |
| RQ1c | Heterogeneidade (coorte × EEG) | χ²(1) = 7.78, p = 0.005 | Results | leitura: heterogêneo ou incerto |
| RQ2a | BrainLat: AUC nos pares 1:1 por escolaridade (±2 anos) | 0.698 [0.515, 0.862]; 15 pares; desempates: mediana 0.711, faixa 0.524–0.867 | Results; Table 2 |  |
| RQ2a | BrainLat: concordância nos pares comparáveis (escolaridade ±2 anos) | 0.678 [0.503, 0.834]; 199 pares | Results; Table 2 |  |
| RQ2a | BrainLat: AUC com overlap weights (idade, sexo, escolaridade) | 0.667 [0.453, 0.836]; ESS AD 17.2, HC 18.2 | Results; Table 2 |  |
| RQ2a | BrainLat: AUC ajustada por covariáveis (Janes & Pepe; idade, sexo, escolaridade) | 0.719 [0.546, 0.863] | Results; Table 2 | verificação; fora da regra de leitura |
| RQ2b | Grego: AUC ajustada por covariáveis (Janes & Pepe; idade, sexo) | 0.860 [0.752, 0.944] | Results; Table 2 | verificação; fora da regra de leitura |
| RQ2b | Grego: AUC com overlap weights (idade, sexo) | 0.865 [0.752, 0.948]; ESS AD 31.6, HC 26.9 | Results; Table 2 | leitura: sobrevive |
| RQ2b | Grego: concordância em pares do mesmo sexo e idade ±2 anos | 0.850 [0.688, 0.971]; 80 pares | Results; Table 2 |  |
| RQ3a | BrainLat → grego (modelo congelado): AUC | 0.855 [0.757, 0.937]; n = 65 (36 AD, 29 HC); inclinação de calibração 1.94; calibration-in-the-large 0.31; Brier 0.161 | Results; Table 2 | leitura: transporta |
| RQ3a | BrainLat → grego: métricas com limiar 0.5 | sens 0.667, espec 0.897, acurácia balanceada 0.782, acurácia 0.769, F1 0.762, MCC 0.568 | Table 2 | descritivo |
| RQ1 (independente) | Grego: escore do modelo do BrainLat além de idade e sexo | OR/DP = 9.34 [3.37, 38.18]; p unilateral < 0.001 (teste paramétrico válido: modelo independente) | Results (RQ1 e RQ3) |  |
| RQ3b | Grego → BrainLat (modelo congelado): AUC | 0.726 [0.596, 0.844]; n = 67 (35 AD, 32 HC); inclinação de calibração 1.15; calibration-in-the-large -0.40; Brier 0.214 | Results; Table 2 | leitura: transporta |
| RQ3b | Grego → BrainLat: métricas com limiar 0.5 | sens 0.914, espec 0.406, acurácia balanceada 0.660, acurácia 0.672, F1 0.744, MCC 0.376 | Table 2 | descritivo |
| RQ1 (independente) | BrainLat: escore do modelo do grego além de idade, sexo e escolaridade | OR/DP = 3.02 [1.41, 8.61]; p = 0.003 (teste paramétrico válido: modelo independente) | Results (RQ1 e RQ3) |  |
| RQ3a | BrainLat → grego, padronizado por coorte: AUC (secundária) | 0.867 [0.763, 0.951]; inclinação de calibração 3.24; Brier 0.187 | Results (uma frase) | fora da regra de leitura |
| RQ3b | Grego → BrainLat, padronizado por coorte: AUC (secundária) | 0.738 [0.609, 0.849]; inclinação de calibração 0.83; Brier 0.210 | Results (uma frase) | fora da regra de leitura |
| Exploratória | BrainLat: ρ parcial entre escore EEG e MoCA nos AD | 0.237 [-0.185, 0.614], p = 0.235; n = 29 | Results (exploratory) |  |
| Exploratória | BrainLat: ρ parcial entre escore EEG e IFS (secondary) nos AD | -0.443 [-0.682, -0.093], p = 0.013; n = 33 | Results (exploratory) |  |
| Exploratória | Grego: ρ parcial entre escore EEG e MMSE nos AD | 0.060 [-0.261, 0.356], p = 0.732; n = 36 | Results (exploratory) |  |
| Descritiva | Features além da demografia (OR por DP; BH dentro da coorte) | mesma direção em 10 de 11; q < 0,05 nas duas: Rel_Theta_mean, Razao_Theta_Alpha, Razao_Theta_Beta | Discussion; Suppl. Table S6 | descritiva |
| Sensibilidade | RF, 7 spectral features | RQ1a OR 2.45 [1.17, 6.20] (p 0.015); RQ1b OR 17.07 [4.69, 107.14] (p unil. < 0.001); BL→GR 0.879 [0.790, 0.955]; GR→BL 0.735 [0.602, 0.854] | Results; Suppl. Table S5 | muda a leitura de: RQ1a |
| Sensibilidade | RF, 10 features (no gamma) | RQ1a OR 1.85 [0.90, 4.30] (p 0.095); RQ1b OR 19.20 [4.93, 139.75] (p unil. < 0.001); BL→GR 0.845 [0.742, 0.929]; GR→BL 0.742 [0.613, 0.855] | Results; Suppl. Table S5 | muda a leitura de: nenhuma |
| Sensibilidade | RF, epochs with gamma RP > 0.15 removed | RQ1a OR 1.90 [0.94, 4.38] (p 0.075); RQ1b OR 18.29 [4.76, 129.24] (p unil. < 0.001); BL→GR 0.852 [0.753, 0.935]; GR→BL 0.717 [0.589, 0.836] | Results; Suppl. Table S5 | muda a leitura de: nenhuma |
| Sensibilidade | SVM, 11 features | RQ1a OR 1.47 [0.76, 3.02] (p 0.258); RQ1b OR 7.46 [3.09, 24.93] (p unil. < 0.001); BL→GR 0.830 [0.718, 0.920]; GR→BL 0.672 [0.533, 0.806] | Results; Suppl. Table S5 | muda a leitura de: nenhuma |
| Sensibilidade | XGBoost, 11 features | RQ1a OR 1.83 [0.92, 4.06] (p 0.085); RQ1b OR 19.03 [4.94, 134.22] (p unil. < 0.001); BL→GR 0.829 [0.725, 0.918]; GR→BL 0.703 [0.568, 0.828] | Results; Suppl. Table S5 | muda a leitura de: nenhuma |

## Alertas

- nenhum

## Material para submissão (seção 13)

| Item | Valor | Onde entra |
|---|---|---|
| Fluxo de participantes (fig0; F0) | BrainLat 81 com .set → 67 analisáveis (14 excluídos: EEG data file (.fdt) missing 14); grego 88 → 65 AD+HC (DFT 23 excluídos por desenho) → 65 analisáveis | Results (Fig. 1) |
| BrainLat: gravidade (MoCA), g de Hedges AD − HC | -2.54 [-3.27, -1.81]; AD 17.0 ± 3.4 (n = 29), HC 25.6 ± 3.2 (n = 23) | Table 1b; Limitations |
| BrainLat: gravidade (IFS), g de Hedges AD − HC | -1.64 [-2.28, -1.01]; AD 14.4 ± 4.9 (n = 33), HC 23.0 ± 5.6 (n = 20) | Table 1b; Limitations |
| Grego (ds004504): gravidade (MMSE), g de Hedges AD − HC | -3.61 [-4.40, -2.82]; AD 17.8 ± 4.5 (n = 36), HC 30.0 ± 0.0 (n = 29) | Table 1b; Limitations |
| Grego: AD por estágio do MMSE (Perneczky 2006) | severe (0–10): 3 (8.3%); moderate (11–20): 25 (69.4%); mild (21–25): 8 (22.2%); questionable (26–29): 0 (0.0%); none (30): 0 (0.0%) | Table 1b; Discussion (spectrum) |
| BrainLat: quem foi excluído (sem EEG analisável) | HC Chile: 14 | Methods; Limitations |

TRIPOD+AI: pronto no notebook: 26, escrever no manuscrito: 16, parcial: 8, não se aplica (dizer por quê): 2
