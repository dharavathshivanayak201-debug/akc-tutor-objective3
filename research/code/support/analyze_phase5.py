import csv,zipfile,io,os,numpy as np,pandas as pd
from scipy.stats import binomtest
base='/mnt/data'; out=base+'/AKC_Tutor_Phase5_Analysis';os.makedirs(out,exist_ok=True)
cols=['correctness','helpfulness','knowledge_appropriateness','affective_appropriateness']
frames=[]
for name,rater in [('work1.txt','R1'),('work2.txt','R2')]:
 with open(base+'/'+name,encoding='utf-8-sig',newline='') as f: raw=list(csv.reader(f))
 good=[];bad=[]
 for i,r in enumerate(raw[1:],2):
  if len(r)==13 and i==75:r=r[:6]+[r[6]+','+r[7]]+r[8:]
  if len(r)!=12:bad.append((i,len(r)));continue
  r[0]=rater;good.append(r)
 df=pd.DataFrame(good,columns=raw[0]);
 for c in cols:df[c]=pd.to_numeric(df[c],errors='coerce')
 print(name,len(df),'invalid',bad,'unique IDs',df.response_id.nunique(),'null',df[cols].isna().sum().to_dict())
 frames.append(df)
with zipfile.ZipFile(base+'/AKC_Tutor_Phase5_Results.zip') as z:
 key=pd.read_csv(io.BytesIO(z.read('PRIVATE_variant_key.csv')))
 print('key columns',key.columns.tolist(),key.head(2).to_string(index=False))
for df in frames:
 print('matched key',df.response_id.isin(key.response_id).sum())
joined=frames[0].merge(frames[1],on='response_id',suffixes=('_R1','_R2'))
print('shared',len(joined),'score differences',sum((joined[c+'_R1']!=joined[c+'_R2']).sum() for c in cols))
combined=pd.concat(frames,ignore_index=True).merge(key[['response_id','variant']],on='response_id',how='left')
combined.to_csv(out+'/validated_ratings_provisional.csv',index=False)
summary=combined.groupby(['rater_id','variant'])[cols].agg(['mean','std','count']);summary.to_csv(out+'/variant_summary_by_rater.csv')
print(summary.to_string())
# use shared only; scenario paired
shared=frames[0][frames[0].response_id.isin(joined.response_id)].merge(key[['response_id','variant']],on='response_id')
print('variants',shared.variant.value_counts().to_dict())
for c in cols:
 p=shared.pivot(index='scenario_id',columns='variant',values=c).dropna(subset=['knowledge_only','knowledge_plus_predicted_emotion'])
 d=p['knowledge_plus_predicted_emotion']-p['knowledge_only'];rng=np.random.default_rng(42);bs=np.mean(rng.choice(d.to_numpy(),size=(20000,len(d)),replace=True),axis=1)
 # exact sign-flip randomization Monte Carlo
 flips=rng.choice([-1,1],size=(20000,len(d)));null=np.mean(flips*d.to_numpy(),axis=1)
 pv=(1+np.sum(np.abs(null)>=abs(d.mean())-1e-12))/(20001)
 print(c,'n',len(d),'delta',round(d.mean(),4),'CI',np.round(np.percentile(bs,[2.5,97.5]),4),'p',pv)
with open(out+'/VALIDATION_WARNING.txt','w') as f:f.write('PROVISIONAL ONLY. Both files initially labeled R1; corrected identifiers reflect user assertion, not independently verified provenance. All 95 shared ratings are identical on all 4 dimensions; inter-rater agreement cannot be treated as convincing independent validation without verification. work1.txt last record is truncated and excluded. Row 75 unquoted comma repaired. Do not claim confirmed independent pedagogical validation.\n')
with zipfile.ZipFile(base+'/AKC_Tutor_Phase5_Provisional_Analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in os.listdir(out):z.write(out+'/'+f,arcname=f)
