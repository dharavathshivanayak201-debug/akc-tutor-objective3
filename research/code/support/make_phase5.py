import nbformat as n
from pathlib import Path
b=n.v4.new_notebook(); cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('''# AKC-Tutor Phase V — Blinded Pedagogical Evaluation & Statistical Ablation

**Input:** Phase IV `AKC_Tutor_Phase4_Results.zip` (96 responses / 24 researcher-authored scenarios / 4 variants).  
**Design:** Blind independent response ratings, scenario-paired comparisons, bootstrap confidence intervals, paired permutation tests, Holm correction, inter-rater agreement, reproducible export.

**Scientific boundary:** These are researcher-authored scenarios, not observations from actual matched students. Phase III behavioral proxies are not measured emotions. RoBERTa output is *predicted* emotion, not human-verified learner affect. This notebook does **not** fabricate human ratings, student outcomes, or statistical significance.

**Run:** Runtime → Run all. Upload the Phase IV ZIP when prompted. The notebook exports an anonymized rating packet. Have at least two independent raters complete the rating forms, then rerun the notebook and upload completed CSVs when prompted. It can also export a preparation-only ZIP immediately.''')
code('''import sys, subprocess, importlib.util
for package in ['numpy','pandas','scipy','matplotlib','scikit-learn','openpyxl']:
    module = 'sklearn' if package == 'scikit-learn' else package
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable,'-m','pip','install','-q',package])
import os, io, zipfile, json, hashlib, pathlib, shutil, textwrap
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from google.colab import files
from IPython.display import display
OUT=pathlib.Path('/content/AKC_Tutor_Phase5'); OUT.mkdir(exist_ok=True)
SEED=42; rng=np.random.default_rng(SEED)
print('Ready. Python:',sys.version.split()[0])''')
md('''## 1. Upload Phase IV results
Upload the **original** Phase IV ZIP. The notebook validates scenario pairing and preserves a SHA-256 fingerprint.''')
code('''print('Upload AKC_Tutor_Phase4_Results.zip')
uploaded=files.upload()
zips=[(k,v) for k,v in uploaded.items() if k.lower().endswith('.zip')]
if not zips: raise ValueError('Upload the Phase IV results ZIP to continue.')
name,raw=zips[0]; digest=hashlib.sha256(raw).hexdigest()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    names=set(z.namelist()); assert 'all_variant_responses.csv' in names, 'Missing response file'
    df=pd.read_csv(io.BytesIO(z.read('all_variant_responses.csv')))
    automated=pd.read_csv(io.BytesIO(z.read('all_variant_responses_scored.csv'))) if 'all_variant_responses_scored.csv' in names else None
required=['scenario_id','variant','skill','student_text','response']
assert set(required).issubset(df.columns),f'Missing columns: {set(required)-set(df.columns)}'
assert not df[required].isna().any().any(), 'Missing essential values'
assert not df.duplicated(['scenario_id','variant']).any(), 'Duplicate scenario/variant rows'
counts=df.groupby('scenario_id').variant.nunique(); assert counts.nunique()==1 and counts.iloc[0]==4, 'Each scenario needs four variants'
print('Rows:',len(df),'Scenarios:',df.scenario_id.nunique(),'Variants:',sorted(df.variant.unique()))
print('SHA256:',digest)
display(df.groupby('variant').size().rename('responses').to_frame())''')
md('''## 2. Create blinded rating packets

Ratings use a **1–5 ordinal scale**: 1 = poor, 2 = weak, 3 = adequate, 4 = good, 5 = excellent.

**Dimensions**
- **Correctness:** Factual/conceptual accuracy of the tutoring response
- **Pedagogical helpfulness:** Explanation quality, next step, and usefulness for the question
- **Knowledge appropriateness:** Suitability for the *scenario-described* learner level (not actual measured mastery)
- **Affective appropriateness:** Supportive and non-patronizing tone for the scenario; do not assume the predicted emotion is correct

Raters see the student's prompt, skill, and **scenario-described** knowledge level, but **not** variant, model identity, predicted emotion, or author-intended emotion. Each rater independently rates every response. Avoid communicating during rating. The packet randomizes order independently per rater.''')
code('''from uuid import uuid5, NAMESPACE_DNS
base=df[['scenario_id','variant','skill','student_text','response']].copy()
if 'scenario_knowledge_level' in df: base['scenario_knowledge_level']=df['scenario_knowledge_level']
else: base['scenario_knowledge_level']='not specified'
base['response_id']=base.apply(lambda r:'R'+hashlib.sha256((str(r.scenario_id)+'|'+str(r.variant)+'|'+str(r.response)).encode()).hexdigest()[:14],axis=1)
assert base.response_id.is_unique
key=base[['response_id','scenario_id','variant']].copy();key.to_csv(OUT/'PRIVATE_variant_key.csv',index=False)
columns=['response_id','scenario_id','skill','student_text','scenario_knowledge_level','response']
for rid in ['R1','R2']:
    packet=base[columns].sample(frac=1,random_state=SEED+(1 if rid=='R1' else 2)).reset_index(drop=True)
    packet.insert(0,'rater_id',rid)
    for dim in ['correctness','helpfulness','knowledge_appropriateness','affective_appropriateness']:
        packet[dim]=pd.Series([pd.NA]*len(packet),dtype='Int64')
    packet['notes']=''
    packet.to_csv(OUT/f'BLINDED_RATINGS_{rid}.csv',index=False)
    packet.to_excel(OUT/f'BLINDED_RATINGS_{rid}.xlsx',index=False)
print('Prepared 2 blinded forms with',len(base),'responses each.')
display(packet.head(3)[['response_id','scenario_id','skill','student_text','response']])''')
md('''## 3. Optional upload of completed ratings

When ratings are not yet available, simply **Cancel** the upload prompt; the notebook will export the preparation packet and stop statistical analysis. For analysis, upload both completed `BLINDED_RATINGS_R1.csv` and `BLINDED_RATINGS_R2.csv`. If you rated in Excel, save each file as CSV (UTF-8). No ratings are prefilled.''')
code('''print('Optional: upload completed rating CSV files for R1 and R2. Cancel if not available.')
try:
    rated_files=files.upload()
except Exception as e:
    print('Upload skipped:',str(e)); rated_files={}
needed={'BLINDED_RATINGS_R1.csv','BLINDED_RATINGS_R2.csv'}
provided={pathlib.Path(k).name:v for k,v in rated_files.items()}
ready=needed.issubset(provided)
if ready:
    ratings=pd.concat([pd.read_csv(io.BytesIO(provided[f])) for f in sorted(needed)],ignore_index=True)
    rating_cols=['correctness','helpfulness','knowledge_appropriateness','affective_appropriateness']
    assert set(['rater_id','response_id']+rating_cols).issubset(ratings.columns)
    assert len(ratings)==len(base)*2 and not ratings.duplicated(['rater_id','response_id']).any()
    assert set(ratings.response_id)==set(base.response_id)
    assert set(ratings.rater_id)=={'R1','R2'}
    for col in rating_cols:
        ratings[col]=pd.to_numeric(ratings[col],errors='coerce')
        assert ratings[col].notna().all() and ratings[col].between(1,5).all() and (ratings[col]%1==0).all(),f'Invalid/missing {col} ratings'
    print('Validated',len(ratings),'human rating rows')
else:
    print('No complete ratings supplied: preparation-only mode. No human evaluation or p-values will be fabricated.')''')
md('''## 4. Automated checks (descriptive only)
These are deterministic template-policy checks. They are **not** independent judgments of pedagogical quality.''')
code('''if automated is not None:
    check_cols=[c for c in ['grounded_topic','contains_followup_check','policy_conformity','words'] if c in automated.columns]
    auto_summary=automated.groupby('variant')[check_cols].mean(numeric_only=True)
    auto_summary.to_csv(OUT/'automated_checks_descriptive.csv')
    display(auto_summary)
else: print('No automated scoring CSV in Phase IV ZIP.')''')
md('''## 5. Independent ratings: summaries, agreement and paired comparisons

**Primary endpoint:** scenario-level mean helpfulness (average of two raters) comparing combined vs knowledge-only. Other dimensions and contrasts are secondary/exploratory.

Paired scenario bootstrap CIs and sign-flip permutation tests preserve within-scenario pairing. Holm correction is applied across the four pre-specified combined-vs-knowledge-only dimension tests. These statistics are exploratory with only 24 scenarios; do not interpret them as learning gains or external generalization.

Inter-rater agreement is reported as quadratic-weighted Cohen's kappa for each dimension; this is descriptive, and agreement may be unstable for narrow rating distributions.''')
code('''if ready:
    from sklearn.metrics import cohen_kappa_score
    r1=ratings[ratings.rater_id=='R1'].set_index('response_id')
    r2=ratings[ratings.rater_id=='R2'].set_index('response_id')
    agreement=[]
    for c in rating_cols:
        agreement.append({'dimension':c,'quadratic_weighted_kappa':cohen_kappa_score(r1.loc[base.response_id,c],r2.loc[base.response_id,c],weights='quadratic')})
    pd.DataFrame(agreement).to_csv(OUT/'inter_rater_agreement.csv',index=False)
    display(pd.DataFrame(agreement))
    mean_ratings=ratings.groupby('response_id')[rating_cols].mean().reset_index()
    joined=mean_ratings.merge(key,on='response_id',validate='one_to_one')
    summary=joined.groupby('variant')[rating_cols].agg(['mean','std','count'])
    summary.to_csv(OUT/'human_rating_summary.csv')
    display(summary)
else: print('Skipped: no completed independent ratings.')''')
code('''def paired_stats(a,b,n_boot=5000,n_perm=20000,seed=42):
    d=np.asarray(a,float)-np.asarray(b,float)
    if not np.isfinite(d).all(): raise ValueError('Non-finite scores')
    rng=np.random.default_rng(seed)
    obs=float(d.mean())
    boot=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(n_boot)])
    ci=np.quantile(boot,[.025,.975])
    signs=rng.choice([-1,1],size=(n_perm,len(d)))
    perm=(signs*d).mean(axis=1)
    p=(1+np.count_nonzero(np.abs(perm)>=abs(obs)))/(n_perm+1)
    return obs,float(ci[0]),float(ci[1]),float(p)
if ready:
    results=[]
    for idx,c in enumerate(rating_cols):
        pivot=joined.pivot(index='scenario_id',columns='variant',values=c)
        assert {'knowledge_plus_predicted_emotion','knowledge_only'}.issubset(pivot.columns),f'Expected knowledge_plus_predicted_emotion and knowledge_only; got {list(pivot.columns)}'
        d,lo,hi,p=paired_stats(pivot['knowledge_plus_predicted_emotion'],pivot['knowledge_only'],seed=SEED+idx)
        results.append({'dimension':c,'contrast':'combined minus knowledge_only','mean_difference':d,'ci95_low':lo,'ci95_high':hi,'permutation_p_uncorrected':p,'scenarios':len(pivot)})
    res=pd.DataFrame(results).sort_values('permutation_p_uncorrected').reset_index(drop=True)
    m=len(res)
    adjusted=np.maximum.accumulate([(m-i)*p for i,p in enumerate(res.permutation_p_uncorrected)])
    res['holm_adjusted_p']=np.minimum(1,adjusted)
    res.to_csv(OUT/'paired_human_comparisons.csv',index=False)
    display(res)
else: print('Skipped statistical comparisons: no ratings.')''')
md('''## 6. Plots, reproducibility and ZIP export''')
code('''if ready:
    means=joined.groupby('variant')[rating_cols].mean()
    fig,ax=plt.subplots(figsize=(11,5))
    means.T.plot(kind='bar',ax=ax,ylim=(1,5))
    ax.set_ylabel('Mean independent rating (1–5)');ax.set_title('Blinded pedagogical evaluation (2 raters)')
    ax.tick_params(axis='x',rotation=20);fig.tight_layout()
    fig.savefig(OUT/'human_rating_comparison.png',dpi=180);plt.show();plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    r=res.sort_values('dimension');x=np.arange(len(r))
    ax.errorbar(x,r.mean_difference,yerr=[r.mean_difference-r.ci95_low,r.ci95_high-r.mean_difference],fmt='o',capsize=4)
    ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(x,r.dimension,rotation=20)
    ax.set_ylabel('Combined – knowledge-only (mean rating)');fig.tight_layout()
    fig.savefig(OUT/'paired_difference_ci.png',dpi=180);plt.show();plt.close(fig)
metadata={'phase':'V','source_zip':name,'source_sha256':digest,'scenario_count':int(df.scenario_id.nunique()),'response_count':len(df),'variants':sorted(df.variant.unique()),'rating_status':'completed' if ready else 'not_provided','rater_count':2 if ready else 0,'primary_endpoint':'helpfulness combined vs knowledge_only','statistical_design':'scenario-paired bootstrap and sign-flip permutation','seed':SEED,'limitations':['researcher-authored scenarios','no matched learner-emotion observations','predicted emotions are not ground truth','no measured student learning gains']}
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
(OUT/'README.txt').write_text('AKC-Tutor Phase V. PRIVATE_variant_key.csv must NOT be shared with raters until rating complete. Blank rating packets are not human evidence. If ratings were supplied, see human_rating_summary.csv, paired_human_comparisons.csv, inter_rater_agreement.csv. Automated checks are not human quality ratings. No student learning gains measured.\\n')
zip_path='/content/AKC_Tutor_Phase5_Results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.iterdir()):
        if f.is_file(): z.write(f,arcname=f.name)
print('Saved',zip_path,'Status:',metadata['rating_status'])
files.download(zip_path)''')
md('''### What to do next

- **If preparation-only:** Download ZIP, distribute `BLINDED_RATINGS_R1.xlsx` and `BLINDED_RATINGS_R2.xlsx` to independent raters. **Do not** distribute `PRIVATE_variant_key.csv`. After rating, save each as CSV and rerun this notebook, uploading the original Phase IV ZIP plus completed rating CSVs when prompted.
- **If ratings were analyzed:** Upload `AKC_Tutor_Phase5_Results.zip` to ChatGPT for methodological review and IEEE paper planning.
- **Important:** Real learning-gain claims require an additional study with real learners and an appropriate pre/post or controlled design.''')
b.cells=cells;b.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
p='/mnt/data/AKC_Tutor_Phase5_Blinded_Evaluation_Colab.ipynb';n.write(b,p)
for i,c in enumerate(cells):
 if c.cell_type=='code': compile(c.source,f'cell_{i}','exec')
print(p,len(cells))
