import csv,io,zipfile,os,json
import pandas as pd
from pathlib import Path
root=Path('/mnt/data');out=root/'AKC_Tutor_Phase5_Validation_Audit';out.mkdir(exist_ok=True)
fields=['correctness','helpfulness','knowledge_appropriateness','affective_appropriateness']
frames={}; issues=[]
for fname,rater in [('work1.txt','R1'),('work2.txt','R2')]:
 with open(root/fname,encoding='utf-8-sig',newline='') as f:raw=list(csv.reader(f))
 good=[]
 for lineno,row in enumerate(raw[1:],2):
  if len(row)==13 and lineno==75:
   row=row[:6]+[row[6]+','+row[7]]+row[8:];issues.append(f'{fname}: line 75 repaired by joining an unquoted comma in response text')
  if len(row)!=12:
   issues.append(f'{fname}: line {lineno} has {len(row)} fields, expected 12; excluded');continue
  row[0]=rater;good.append(row)
 df=pd.DataFrame(good,columns=raw[0]);
 for c in fields:df[c]=pd.to_numeric(df[c],errors='coerce')
 assert df.response_id.is_unique and df[fields].notna().all().all() and df[fields].ge(1).all().all() and df[fields].le(5).all().all()
 frames[rater]=df
 df.to_csv(out/f'REPAIRED_{rater}_complete_rows.csv',index=False)
with zipfile.ZipFile(root/'AKC_Tutor_Phase5_Results.zip') as z:key=pd.read_csv(io.BytesIO(z.read('PRIVATE_variant_key.csv')))
assert set(frames['R1'].response_id).issubset(set(key.response_id)) and set(frames['R2'].response_id)==set(key.response_id)
missing=set(frames['R2'].response_id)-set(frames['R1'].response_id)
assert len(missing)==1
missing_id=next(iter(missing));missing_info=frames['R2'].loc[frames['R2'].response_id==missing_id,['response_id','scenario_id','skill','student_text']]
missing_info.to_csv(out/'R1_MISSING_RESPONSE_TO_RECOVER.csv',index=False)
shared=frames['R1'].merge(frames['R2'],on='response_id',suffixes=('_R1','_R2'))
identical_scores=all((shared[c+'_R1']==shared[c+'_R2']).all() for c in fields)
identical_notes=(shared.notes_R1==shared.notes_R2).all()
with open(out/'VALIDATION_REPORT.txt','w') as f:
 f.write('AKC-Tutor Phase V validation audit\n\n')
 f.write('R1 complete records: 95/96; R2 complete records: 96/96.\n')
 f.write(f'R1 missing response ID: {missing_id}.\n')
 f.write(f'95 shared responses: all 380 ratings identical: {identical_scores}; all notes identical: {identical_notes}.\n')
 f.write('Both original files label their rater_id as R1. Repaired R2 identifier is based on the user statement, not independent provenance verification.\n')
 f.write('The same scores and comments across all shared rows are a material provenance concern. Do not report confirmed independent evaluation or inter-rater kappa until original independently saved rating workbooks and evaluation provenance are verified.\n')
 f.write('No scores have been imputed, invented, or modified. The R1 truncated row is excluded.\n\n')
 f.write('Repairs/issues:\n'+'\n'.join(issues)+'\n\n')
 f.write('To finalize: obtain R1 original full workbook with the missing response rated; obtain both original separately completed workbooks and evaluator confirmation of independent scoring. Then rerun reliability and scenario-paired analyses.\n')
print('missing',missing_id,'identical_scores',identical_scores,'notes',identical_notes)
with zipfile.ZipFile(root/'AKC_Tutor_Phase5_Validation_Audit.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in out.iterdir():z.write(p,p.name)
