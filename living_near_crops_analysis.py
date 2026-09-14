# Crop proximity (survey primary, georef comparison), pesticide handling and PD severity: OLS with a residential proxy. Reproduces every table and figure in living_near_crops_pd_colombia.tex.
import numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
A = 'D:/Proyecto_TQ/article/'
df = pd.read_csv('D:/Proyecto_TQ/DatosGeoref20260708/DatosGeoref/20260603121727_GEOREF1916_DATA_2026-06-02_1223.csv', low_memory=False).replace({'UNK': np.nan, 'Unknown': np.nan, 'Not applicable': np.nan})
num = lambda c: pd.to_numeric(df[c], errors='coerce')

# Exposure proxies. Survey: "yes" to <500 m from a crop field, per life period. Georef: distance to nearest cultivated polygon (living_near_crops_georef_distances.py), NaN -> 1000 m cap; patients with no geocoded period are missing on the map proxy.
crop_q = [c for c in df.columns if 'cerca a un campo de cultivo' in c]; df[crop_q] = df[crop_q].apply(pd.to_numeric, errors='coerce'); df['survey_n'] = df[crop_q].sum(axis=1).where(df[crop_q].notna().any(axis=1))
df['crop_years'] = (df[crop_q].fillna(0).values * np.column_stack([np.full(len(df), 18), np.full(len(df), 8), np.full(len(df), 15), np.full(len(df), 20), (pd.to_numeric(df['Edad actual'], errors='coerce') - 60).clip(lower=0)])).sum(axis=1)
geo = pd.read_csv(A + 'georef_distances.csv'); geo['dist_m'] = geo['dist_m'].fillna(1000)
df = df.merge(geo.pivot(index='Record ID', columns='period', values='dist_m').add_prefix('geo_'), left_on='Record ID', right_index=True, how='left')
geo_cols = ['geo_0-17', 'geo_18-25', 'geo_26-40', 'geo_41-60', 'geo_61']; has_geo = df[geo_cols].notna().any(axis=1)
df['geo_n'] = (df[geo_cols] < 500).sum(axis=1).where(has_geo); df['geo_min_m'] = df[geo_cols].min(axis=1)
df['Z_survey'] = (df['survey_n'] >= 1).astype(float).where(df['survey_n'].notna()); df['survey_3plus'] = (df['survey_n'] >= 3).astype(float).where(df['survey_n'].notna()); df['Z_geo'] = (df['geo_n'] >= 3).astype(float).where(has_geo); df['any_geo'] = (df['geo_n'] >= 1).astype(float).where(has_geo)
df['survey_cat'] = pd.cut(df['survey_n'], [-1, 0, 1, 2, 5], labels=['0', '1', '2', '3+'])

# Pesticide handling, outcomes, controls (raw units), rural residence (DANE class) as the access-to-care proxy
df['pest_work'] = num('Durante su vida, ¿ha tenido un trabajo en el que le tocara aplicar o mezclar algún tipo de pesticida?'); df['pest_home'] = num('Durante su vida, ¿ha aplicado o mezclado algún tipo de pesticida en su casa o zonas de recreación?')
df['pesticide'] = ((df['pest_work'] == 1) | (df['pest_home'] == 1)).astype(int)
df['hy'] = num('Estadio de la enfermedad, Hoehn y Yahr'); df['updrs'] = num('Puntaje total de MDS-UPDRS III'); df['moca'] = num('Puntaje total MoCA')
df['age'] = num('Edad actual'); df['female'] = num('Sexo'); df['educ_years'] = num('Número total de años de educación'); df['onset'] = num('Edad de inicio de síntomas motores'); df['dx_age'] = num('Edad de diagnóstico de EP')
# Years since first symptom: age minus onset age when 0-20; else age minus diagnosis age when 0-20; else flagged and set to the median (onset and diagnosis ages carry entry errors, current age agrees with year of birth)
d_on, d_dx = df['age'] - df['onset'], df['age'] - df['dx_age']; df['dur_symptoms'] = d_on.where(d_on.between(0, 20), d_dx.where(d_dx.between(0, 20))); df['dur_flag'] = df['dur_symptoms'].isna().astype(int); df['dur_symptoms'] = df['dur_symptoms'].fillna(df['dur_symptoms'].median())
df['coffee_ever'] = num('¿Ha tomado café?'); df['coffee_cups'] = num('¿Cuánto café toma (o tomó)?').fillna(0)
df['smoke_ever'] = num('¿Ha fumado durante su vida?'); df['pack_index'] = num('Por favor, indique el IPA (índice de paquetes de cigarrillos)').fillna(0); df['levodopa'] = num('¿Actualmente está tratado con levodopa?')
# Rural residence: DANE MGN 2024 class of the geocoded current residence (living_near_crops_dane_clase.py): 1 = outside the cabecera municipal. Fallback when the address did not geocode: 1 if the vereda/corregimiento field is filled or the address names a vereda, finca, corregimiento, km marker, parcela, resguardo, hacienda or caserio; else 0 (cabecera). rural_geo = 1 when the class comes from the map.
df = df.merge(pd.read_csv(A + 'dane_clase_residencia.csv')[['Record ID', 'dane_clase']], on='Record ID', how='left'); df['rural_geo'] = df['dane_clase'].notna().astype(int)
addr = df['Dirección de residencia actual'].fillna('').str.upper().str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii')
txt_rural = addr.str.contains(r'VEREDA|FINCA|CORREGIMIENTO|KM|KILOMETRO|VIA|PARCELA|RESGUARDO|ZONA RURAL|HACIENDA|CASERIO', regex=True) | df['Lugar de residencia (vereda/corregimiento)'].notna()
df['rural'] = (df['dane_clase'] != '1. CABECERA').astype(float).where(df['dane_clase'].notna(), txt_rural.astype(float))
outs = ['hy', 'updrs', 'moca']; controls = ['age', 'female', 'educ_years', 'dur_symptoms', 'dur_flag', 'coffee_ever', 'coffee_cups', 'smoke_ever', 'pack_index', 'levodopa']
labels = {'Z_survey': 'Near crops: survey (1+ periods)', 'Z_geo': 'Near crops: georef (3+ periods)', 'survey_3plus': 'Near crops: survey (3+ periods)', 'any_geo': 'Near crops: georef (1+ periods)', 'geo_n': 'Periods near crops, georef (0--5)', 'pesticide': 'Pesticide application (self-report)', 'rural': 'Rural residence (DANE class)', 'hy': 'Hoehn \\& Yahr stage', 'updrs': 'MDS-UPDRS III', 'moca': 'MoCA', 'age': 'Age (years)', 'female': 'Female', 'educ_years': 'Education (years)', 'dur_symptoms': 'Years since first symptom', 'dur_flag': 'Duration imputed (flag)', 'crop_years': 'Years near crops, survey', 'coffee_ever': 'Ever drank coffee', 'coffee_cups': 'Coffee intake (0--4 scale)', 'smoke_ever': 'Ever smoked', 'pack_index': 'Pack-year index', 'levodopa': 'Levodopa treatment', 'survey_n': 'Periods near crops, survey (0--5)', 'geo_min_m': 'Min. distance to crops (m)', 'pest_work': 'Occupational pesticide application', 'pest_home': 'Domestic pesticide application', 'sp_1': 'Near crops in 1 period (survey)', 'sp_2': 'Near crops in 2 periods (survey)', 'sp_3plus': 'Near crops in 3+ periods (survey)'}

# Analytic sample: complete outcomes, survey answers and controls (duration never missing after the rule above)
ana = df.dropna(subset=['hy', 'updrs', 'Z_survey', 'rural'] + controls).copy()
print('analytic:', len(ana), 'of', len(df), '| no crop answers at all:', df['Z_survey'].isna().sum(), '| duration from onset:', d_on.loc[ana.index].between(0, 20).sum(), '| from diagnosis age:', (~d_on.between(0, 20) & d_dx.between(0, 20)).loc[ana.index].sum(), '| flagged:', ana['dur_flag'].sum(), '(negative onset-based:', (d_on.loc[ana.index] < 0).sum(), ')')
print('geocoded residences per period:', df[geo_cols].notna().sum().to_dict(), '| analytic patients without geocode:', ana['Z_geo'].isna().sum())
print('rural source: geocoded', ana['rural_geo'].sum(), 'fallback', (ana['rural_geo'] == 0).sum(), '| rural by source:', ana.groupby('rural_geo')['rural'].agg(['sum', 'count']).to_dict())
print('analytic shares:', ana[['Z_survey', 'survey_3plus', 'any_geo', 'Z_geo', 'pesticide', 'pest_work', 'pest_home', 'rural']].mean().round(2).to_dict())
print('survey x georef (geocoded patients):\n', pd.crosstab(ana['Z_survey'], ana['Z_geo'], margins=True))

# Table 1 descriptives, Table 2 balance by survey proxy
tab = lambda name, head, rows, note='': open(A + f'tables/{name}.tex', 'w', encoding='utf-8').write(head + rows + "\\hline\\hline\n" + note + "\\end{tabular}\n")
d = ana[['hy', 'updrs', 'moca', 'Z_survey', 'survey_3plus', 'Z_geo', 'any_geo', 'survey_n', 'crop_years', 'geo_n', 'geo_min_m', 'pesticide', 'pest_work', 'pest_home', 'rural'] + controls].describe().T
tab('tab_descriptives', "\\begin{tabular}{lccccc}\n\\hline\\hline\nVariable & N & Mean & St. Dev. & Min & Max \\\\\n\\hline\n", ''.join(f"{labels[v]} & {int(r['count'])} & {r['mean']:.2f} & {r['std']:.2f} & {r['min']:.1f} & {r['max']:.1f} \\\\\n" for v, r in d.iterrows()))
g0, g1 = ana[ana.Z_survey == 0], ana[ana.Z_survey == 1]
tab('tab_balance', f"\\begin{{tabular}}{{lcccc}}\n\\hline\\hline\nVariable & Not near crops (n={len(g0)}) & Near crops (n={len(g1)}) & Difference & p-value \\\\\n\\hline\n", ''.join(f"{labels[v]} & {g0[v].mean():.2f} & {g1[v].mean():.2f} & {g1[v].mean()-g0[v].mean():.2f} & {stats.ttest_ind(g1[v], g0[v], equal_var=False, nan_policy='omit').pvalue:.3f} \\\\\n" for v in controls + ['rural', 'pesticide'] + outs))

# Tables 3-4: OLS with HC1 SE. (1) survey; (2) + controls; (3) pesticide + controls; (4) survey + pesticide + controls; (5) + rural residence; (6) + georef, geocoded patients only
star = lambda p: '$^{***}$' if p < 0.01 else '$^{**}$' if p < 0.05 else '$^{*}$' if p < 0.1 else ''
ols = lambda y, x, d=ana: sm.OLS(d[y], sm.add_constant(d[x]), missing='drop').fit(cov_type='HC1')
specs = {1: ['Z_survey'], 2: ['Z_survey'] + controls, 3: ['pesticide'] + controls, 4: ['Z_survey', 'pesticide'] + controls, 5: ['Z_survey', 'pesticide', 'rural'] + controls, 6: ['Z_geo', 'Z_survey', 'pesticide', 'rural'] + controls}
fits = {(y, k): ols(y, x, ana.dropna(subset=x)) for y in outs for k, x in specs.items()}
for y in outs:
    ms = [fits[(y, k)] for k in specs]; print(f'\n== {labels[y]}, spec (5) ==\n', ms[4].summary().tables[1], f'\n== {labels[y]}, spec (6) ==\n', ms[5].summary().tables[1])
    body = ''.join(f"{labels.get(t, 'Constant')} & " + ' & '.join(f"{m.params[t]:.3f}{star(m.pvalues[t])}" if t in m.params else '' for m in ms) + " \\\\\n & " + ' & '.join(f"({m.bse[t]:.3f})" if t in m.params else '' for m in ms) + " \\\\\n" for t in ['Z_survey', 'pesticide', 'rural', 'Z_geo'] + controls + ['const'])
    foot = f"\\hline\nObservations & {' & '.join(str(int(m.nobs)) for m in ms)} \\\\\nR$^2$ & {' & '.join(f'{m.rsquared:.3f}' for m in ms)} \\\\\nAdjusted R$^2$ & {' & '.join(f'{m.rsquared_adj:.3f}' for m in ms)} \\\\\n"
    tab(f'tab_ols_{y}', f"\\begin{{tabular}}{{lcccccc}}\n\\hline\\hline\n & \\multicolumn{{6}}{{c}}{{\\textit{{Dependent variable:}}}} \\\\\n\\cline{{2-7}}\n & \\multicolumn{{6}}{{c}}{{{labels[y]}}} \\\\\n & (1) & (2) & (3) & (4) & (5) & (6) \\\\\n\\hline\n", body + foot, "\\textit{Note:} & \\multicolumn{6}{r}{$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01. Robust SE (HC1).} \\\\\n & \\multicolumn{6}{r}{Controls in columns (2)--(6). Column (6): patients with a geocoded residence.} \\\\\n")

# Figure 1: exposure coefficients, 95% CI, alpha by significance
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, y in zip(axes, outs):
    rows = [(f"({k}) {labels[t]}", fits[(y, k)], t) for k, x in specs.items() for t in x if t in ('Z_survey', 'Z_geo', 'pesticide')]
    for i, (lab, m, t) in enumerate(rows):
        ci = m.conf_int(); a = 1 if m.pvalues[t] < .01 else .75 if m.pvalues[t] < .05 else .5 if m.pvalues[t] < .1 else .25
        ax.errorbar(m.params[t], i, xerr=[[m.params[t] - ci.loc[t, 0]], [ci.loc[t, 1] - m.params[t]]], fmt='o', color='darkblue', alpha=a, capsize=3)
        ax.text(m.params[t], i + 0.25, f"{m.params[t]:.2f}" + '*' * (3 if m.pvalues[t] < .01 else 2 if m.pvalues[t] < .05 else 1 if m.pvalues[t] < .1 else 0), ha='center', fontsize=9, alpha=a)
    ax.axvline(0, color='black', ls='--'); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0].replace('\\&', '&') for r in rows], fontsize=9); ax.invert_yaxis(); ax.set_title(labels[y].replace('\\&', '&'), fontweight='bold'); ax.set_xlabel('Estimate (95% CI)')
fig.tight_layout(); fig.savefig(A + 'figures/coef_exposure.png', dpi=150); plt.close(fig)

# Tables 5-6: periods near crops (survey) vs never, raw Welch and adjusted dummies
ref = ana[ana['survey_cat'] == '0']; rows = ''
for cat in ['0', '1', '2', '3+']:
    g = ana[ana['survey_cat'] == cat]; p = lambda v: '' if cat == '0' else f"{stats.ttest_ind(g[v], ref[v], equal_var=False, nan_policy='omit').pvalue:.3f}"
    rows += f"{cat} & {len(g)} & {g['hy'].mean():.2f} & {p('hy')} & {g['updrs'].mean():.1f} & {p('updrs')} & {g['moca'].mean():.1f} & {p('moca')} & {g['pesticide'].mean():.2f} & {g['Z_geo'].mean():.2f} \\\\\n"
tab('tab_survey_periods', "\\begin{tabular}{lccccccccc}\n\\hline\\hline\nPeriods (survey) & N & H\\&Y & p vs 0 & UPDRS III & p vs 0 & MoCA & p vs 0 & Pesticide & Georef 3+ \\\\\n\\hline\n", rows)
dum = pd.get_dummies(ana['survey_cat'], prefix='sp', drop_first=True).astype(int).rename(columns={'sp_3+': 'sp_3plus'}); dcols = list(dum.columns); ana[dcols] = dum
cf = {y: ols(y, dcols + ['pesticide', 'rural'] + controls) for y in outs}
body = ''.join(f"{labels[t]} & " + ' & '.join(f"{cf[y].params[t]:.3f}{star(cf[y].pvalues[t])}" for y in outs) + " \\\\\n & " + ' & '.join(f"({cf[y].bse[t]:.3f})" for y in outs) + " \\\\\n" for t in dcols + ['pesticide', 'rural'] + controls)
tab('tab_survey_dummies', "\\begin{tabular}{lccc}\n\\hline\\hline\n & Hoehn \\& Yahr stage & MDS-UPDRS III & MoCA \\\\\n & (1) & (2) & (3) \\\\\n\\hline\n", body + f"\\hline\nObservations & {' & '.join(str(int(cf[y].nobs)) for y in outs)} \\\\\nAdjusted R$^2$ & {' & '.join(f'{cf[y].rsquared_adj:.3f}' for y in outs)} \\\\\n", "\\textit{Note:} & \\multicolumn{3}{r}{$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01. Robust SE (HC1).} \\\\\n & \\multicolumn{3}{r}{Reference: never near crops.} \\\\\n")

# Table 7: robustness of the survey estimate (column 5 as baseline)
base = ['Z_survey', 'pesticide', 'rural'] + controls
rob = {'Baseline, column (5)': {y: fits[(y, 5)] for y in outs}, 'Without levodopa': {y: ols(y, [c for c in base if c != 'levodopa']) for y in outs},
       f'Dropping {int(ana.dur_flag.sum())} patients with imputed duration': {y: ols(y, base, ana[ana.dur_flag == 0]) for y in outs},
       'Linear period count (per period)': {y: ols(y, ['survey_n'] + base[1:]) for y in outs},
       'Linear years near crops (per 10 years)': {y: ols(y, ['crop_years'] + base[1:]) for y in outs},
       'Geocoded residences only (DANE class)': {y: ols(y, base, ana[ana.rural_geo == 1]) for y in outs},
       'Ordered logit (latent scale)': {'hy': OrderedModel(ana['hy'], ana[base], distr='logit').fit(method='bfgs', disp=0)}}
term = lambda k: 'crop_years' if 'years' in k else 'survey_n' if 'Linear' in k else 'Z_survey'; scale = lambda k: 10 if 'years' in k else 1
cell = lambda m, t, s: f"{m.params[t] * s:.3f}{star(m.pvalues[t])} ({m.bse[t] * s:.3f})"
rows = ''.join(f"{k} & " + ' & '.join(cell(v[y], term(k), scale(k)) if y in v else '' for y in outs) + f" & {int(v['hy'].nobs)} \\\\\n" for k, v in rob.items())
tab('tab_robust', "\\begin{tabular}{lcccc}\n\\hline\\hline\nSpecification & Hoehn \\& Yahr stage & MDS-UPDRS III & MoCA & N \\\\\n\\hline\n", rows, "\\textit{Note:} & \\multicolumn{4}{r}{Survey proxy coefficient (SE). HC1 SE; ordered logit conventional.} \\\\\n")
for k, v in rob.items(): print(k, {y: (round(m.params[term(k)] * scale(k), 3), round(m.bse[term(k)] * scale(k), 3), round(m.pvalues[term(k)], 3)) for y, m in v.items()})

# Table 8: severity by georef periods, geocoded patients; map rules for the text
prof = ana.groupby('geo_n')[['hy', 'updrs', 'moca', 'pesticide']].agg(['mean', 'count'])
tab('tab_dose_geo_n', "\\begin{tabular}{lccccc}\n\\hline\\hline\nPeriods near crops (georef) & N & Mean H\\&Y & Mean UPDRS III & Mean MoCA & Pesticide share \\\\\n\\hline\n", ''.join(f"{int(n)} & {int(prof.loc[n, ('hy', 'count')])} & {prof.loc[n, ('hy', 'mean')]:.2f} & {prof.loc[n, ('updrs', 'mean')]:.1f} & {prof.loc[n, ('moca', 'mean')]:.1f} & {prof.loc[n, ('pesticide', 'mean')]:.2f} \\\\\n" for n in prof.index))
for z in ['any_geo', 'geo_n']: print('map rule', z, {y: (round((m := ols(y, [z] + base, ana.dropna(subset=[z]))).params[z], 3), round(m.bse[z], 3), round(m.pvalues[z], 3)) for y in outs})
for y in outs: print(y, 'Z_survey col 5 95% CI:', fits[(y, 5)].conf_int().loc['Z_survey'].round(2).tolist(), '| MDE 80%:', round(2.8 * fits[(y, 5)].bse['Z_survey'], 2))
ana[['Record ID', 'Z_survey', 'Z_geo', 'survey_cat', 'survey_n', 'geo_n', 'geo_min_m', 'pesticide', 'pest_work', 'pest_home', 'rural', 'hy', 'updrs', 'moca'] + controls].to_csv(A + 'analytic_sample.csv', index=False)
