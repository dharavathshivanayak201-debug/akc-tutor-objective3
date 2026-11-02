import nbformat as n
from textwrap import dedent
b=n.v4.new_notebook();b.metadata={'colab':{'provenance':[],'gpuType':'A100'},'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}}
def md(x):b.cells.append(n.v4.new_markdown_cell(dedent(x)))
def code(x):b.cells.append(n.v4.new_code_cell(dedent(x)))
md('''# AKC-Tutor — Phase IV: Adaptive Conversational Tutoring
**Automated Google Colab pipeline | Phase II RoBERTa + Phase III learner model | Honest, evidence-gated evaluation**

**What this experiment actually measures:** whether a controlled tutoring response generator follows pedagogical constraints under knowledge and emotion inputs, plus component ablations. It does **not** measure student learning gains or validate emotion labels on real learners. GoEmotions is Reddit text; Phase III educational data has no matched dialogue. **No synthetic pairing of student and emotion records is performed.**

**Before Run all:** select **Runtime → Change runtime type → GPU** (A100/L4 recommended). Upload the original `AKC_Tutor_Phase3_Results.zip` when prompted. Optionally upload your saved Phase II RoBERTa checkpoint ZIP; otherwise the notebook automatically re-trains the Phase II RoBERTa model on the official GoEmotions training split (3 epochs, potentially 20–90+ minutes). Phase II *results ZIP alone* contains predictions but **no weights** and cannot be used for inference.

This notebook never uses the held-out GoEmotions test split for model selection or prompt tuning. Model selection uses the validation split only. Phase IV evaluation scenarios are explicitly *researcher-authored*, not learner observations.''')
md('## 1. Dependencies and reproducibility')
code('''%pip -q install "transformers>=4.44,<5" "safetensors>=0.4" "scikit-learn>=1.6,<1.7" pandas matplotlib seaborn joblib tqdm
''')
code('''import os, json, io, sys, re, random, hashlib, zipfile, shutil, time, platform, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, joblib, sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',DEVICE,'GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
ROOT=Path('/content/AKC_Phase4');ROOT.mkdir(exist_ok=True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
MODELS=ROOT/'models';MODELS.mkdir(exist_ok=True)
MODEL_ID='FacebookAI/roberta-base'; MAX_LEN=128; EPOCHS=3; BATCH=16; ACCUM=2
assert DEVICE.type=='cuda', 'Please select a GPU runtime; Phase II checkpoint recovery requires GPU training.'
''')
md('''## 2. Upload Phase III results and optional Phase II model checkpoint
Only trusted archives from your own Colab experiments should be uploaded. **Never unpickle untrusted joblib files.**''')
code('''from google.colab import files
print('Upload AKC_Tutor_Phase3_Results.zip. Optionally also upload a Phase II model ZIP (config.json, model.safetensors, tokenizer files).')
uploaded=files.upload()
phase3_candidates=[k for k in uploaded if 'Phase3' in k and k.lower().endswith('.zip')]
if not phase3_candidates:
    raise FileNotFoundError('Phase III results ZIP is required. Re-run this cell and upload AKC_Tutor_Phase3_Results.zip.')
phase3_zip=ROOT/'phase3.zip';phase3_zip.write_bytes(uploaded[phase3_candidates[0]])
with zipfile.ZipFile(phase3_zip) as z:
    required={'knowledge_plus_behavior_proxy.joblib','knowledge_only.joblib','test_predictions.csv','run_metadata.json','feature_definitions.json'}
    assert required.issubset(set(z.namelist())), f'Missing Phase III artifacts: {required-set(z.namelist())}'
    meta3=json.loads(z.read('run_metadata.json'))
    assert meta3.get('true_affect_mode') is False, 'Unexpected Phase III mode: review dataset alignment manually.'
    features3=json.loads(z.read('feature_definitions.json'))
    fusion_model=joblib.load(io.BytesIO(z.read('knowledge_plus_behavior_proxy.joblib')))
    knowledge_model=joblib.load(io.BytesIO(z.read('knowledge_only.joblib')))
    p3test=pd.read_csv(io.BytesIO(z.read('test_predictions.csv')))
print('Phase III source:',meta3['dataset_source'],'held-out rows:',len(p3test))
print('Phase III model feature names:',fusion_model.feature_names_in_.tolist())
print('Phase III held-out learner IDs:',p3test.user_id.nunique())
phase2_candidates=[k for k in uploaded if k.lower().endswith('.zip') and k not in phase3_candidates]
''')
md('''## 3. Load Phase II RoBERTa checkpoint or recover it by re-training
If no checkpoint was uploaded, this cell downloads official GoEmotions train/dev, filters to exactly one label, and trains RoBERTa-base using Phase II hyperparameters. **The Phase II results ZIP is not a model checkpoint.**''')
code('''import urllib.request
EMODIR=MODELS/'emotion_roberta';EMODIR.mkdir(exist_ok=True)
checkpoint_loaded=False
if phase2_candidates:
    candidate=phase2_candidates[0]
    with zipfile.ZipFile(io.BytesIO(uploaded[candidate])) as z:
        safe=[x for x in z.namelist() if not x.startswith('/') and '..' not in Path(x).parts]
        if {'config.json','model.safetensors'}.issubset({Path(x).name for x in safe}):
            for name in safe:
                if name.endswith('/') or Path(name).name not in {'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'}: continue
                (EMODIR/Path(name).name).write_bytes(z.read(name))
            checkpoint_loaded=True
        else: print('Uploaded additional ZIP has no model.safetensors/config.json; re-training instead.')

BASE='https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/'
DATA=ROOT/'goemotions';DATA.mkdir(exist_ok=True)
for name in ['emotions.txt','train.tsv','dev.tsv']:
    target=DATA/name
    if not target.exists():
        urllib.request.urlretrieve(BASE+name,target)
labels=(DATA/'emotions.txt').read_text().splitlines()
assert len(labels)==28

def load_single_label(split):
    d=pd.read_csv(DATA/f'{split}.tsv',sep='\\t',header=None,names=['text','label_ids','comment_id'],dtype=str,keep_default_na=False)
    d=d[d.label_ids.str.fullmatch(r'\\d+')].copy()
    d['label']=d.label_ids.astype(int)
    return d.reset_index(drop=True)
train=load_single_label('train');dev=load_single_label('dev')
print('GoEmotions single-label sizes:',len(train),len(dev))
if checkpoint_loaded:
    tokenizer=AutoTokenizer.from_pretrained(str(EMODIR))
    emotion_model=AutoModelForSequenceClassification.from_pretrained(str(EMODIR)).to(DEVICE).eval()
    assert emotion_model.config.num_labels==28
    print('Loaded uploaded Phase II model checkpoint.')
else:
    print('No valid Phase II checkpoint: fine-tuning RoBERTa-base. This is compute intensive.')
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    class EmotionDS(Dataset):
        def __init__(self,d):
            self.enc=tokenizer(d.text.tolist(),truncation=True,padding='max_length',max_length=MAX_LEN)
            self.y=d.label.to_numpy()
        def __len__(self): return len(self.y)
        def __getitem__(self,i):
            return {**{k:torch.tensor(v[i]) for k,v in self.enc.items()},'labels':torch.tensor(int(self.y[i]))}
    tl=DataLoader(EmotionDS(train),batch_size=BATCH,shuffle=True,num_workers=0)
    vl=DataLoader(EmotionDS(dev),batch_size=BATCH*2,shuffle=False,num_workers=0)
    emotion_model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,num_labels=28,id2label={i:x for i,x in enumerate(labels)},label2id={x:i for i,x in enumerate(labels)}).to(DEVICE)
    optimizer=torch.optim.AdamW(emotion_model.parameters(),lr=2e-5,weight_decay=.01)
    steps=EPOCHS*((len(tl)+ACCUM-1)//ACCUM)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*steps),steps)
    best=-1;history=[]
    for epoch in range(1,EPOCHS+1):
        emotion_model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for i,batch in enumerate(tqdm(tl,desc=f'Train epoch {epoch}')):
            batch={k:v.to(DEVICE) for k,v in batch.items()}
            with torch.autocast(device_type='cuda',dtype=torch.float16): loss=emotion_model(**batch).loss
            (loss/ACCUM).backward();losses.append(float(loss.detach().cpu()))
            if (i+1)%ACCUM==0 or i+1==len(tl):
                torch.nn.utils.clip_grad_norm_(emotion_model.parameters(),1.0)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        emotion_model.eval();ys=[];pred=[]
        with torch.no_grad():
            for batch in tqdm(vl,desc='Validate',leave=False):
                ys.extend(batch['labels'].numpy().tolist())
                x={k:v.to(DEVICE) for k,v in batch.items() if k!='labels'}
                with torch.autocast(device_type='cuda',dtype=torch.float16): logits=emotion_model(**x).logits
                pred.extend(logits.argmax(-1).cpu().tolist())
        f=f1_score(ys,pred,average='macro',labels=list(range(28)),zero_division=0)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_macro_f1':f})
        print('Epoch',epoch,'validation Macro-F1',round(f,4))
        if f>best:
            best=f;emotion_model.save_pretrained(EMODIR,safe_serialization=True);tokenizer.save_pretrained(EMODIR)
    pd.DataFrame(history).to_csv(OUT/'emotion_training_history.csv',index=False)
    emotion_model=AutoModelForSequenceClassification.from_pretrained(EMODIR).to(DEVICE).eval()
    print('Recovered Phase II model; best dev Macro-F1',best)
(OUT/'emotion_checkpoint_provenance.json').write_text(json.dumps({'source':'uploaded_saved_phase2_checkpoint' if checkpoint_loaded else 'retrained_on_goemotions_train_dev_selection','model_id':MODEL_ID,'single_label':True,'phase2_test_not_used':True},indent=2))
''')
md('''## 4. Predict affect from dialogue, separately from behavioral proxies
**Predicted emotion is a model output, not a verified student emotion.** No phase III behavioral proxy is ever relabeled as affect.''')
code('''def emotion_infer(texts,batch_size=24):
    outputs=[]
    emotion_model.eval()
    for start in range(0,len(texts),batch_size):
        chunk=texts[start:start+batch_size]
        inp=tokenizer(chunk,return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN).to(DEVICE)
        with torch.inference_mode():
            logits=emotion_model(**inp).logits
            probs=logits.softmax(-1).cpu().numpy()
        for row in probs:
            idx=int(np.argmax(row));outputs.append({'pred_emotion':labels[idx],'emotion_confidence':float(row[idx]),'emotion_id':idx})
    return outputs
print(emotion_infer(['I am confused about how to solve this equation.','Thank you, I understand now!']))
''')
md('''## 5. Prepare learner-state samples from Phase III held-out predictions
Phase III outputs are **real educational response predictions**, but they are **not paired with the Phase IV dialogue scenarios**. We use their probability distribution only to create explicitly *illustrative* high/medium/low knowledge conditions. These conditions are **not represented as observations from the same learner who wrote a dialogue**.''')
code('''pcol='p_knowledge_plus_behavior_proxy'; assert pcol in p3test.columns
assert p3test[pcol].between(0,1).all()
q=p3test[pcol].quantile([.2,.5,.8]).to_dict()
STATE_LEVELS={'low':float(q[.2]),'medium':float(q[.5]),'high':float(q[.8])}
print('Illustrative knowledge probability levels from real Phase III test distribution:',STATE_LEVELS)
(OUT/'knowledge_state_provenance.json').write_text(json.dumps({'source':'Phase III held-out predicted probabilities','quantiles':STATE_LEVELS,'paired_with_dialogue':False,'note':'scenario conditions, not matched learner records'},indent=2))
''')
md('''## 6. Researcher-authored educational dialogue scenarios
These are controlled *scenario prompts* for testing system behavior, not real student dialogues and not evidence of learning outcomes. No GoEmotions text is falsely paired to a student ID.''')
code('''SCENARIOS=[
('linear equations','I do not understand why we move the 3 to the other side.','low','confusion'),
('linear equations','I tried twice and I keep getting the wrong answer. This is frustrating.','low','frustration'),
('linear equations','I think I can solve 2x + 3 = 11. Can I try a harder one?','high','confidence'),
('linear equations','Can you give me a hint rather than the answer?','medium','curiosity'),
('fractions','I am confused about adding fractions with different denominators.','low','confusion'),
('fractions','I keep making mistakes with common denominators and I am annoyed.','low','frustration'),
('fractions','I understand equivalent fractions now. What is next?','high','confidence'),
('fractions','Could you show one example of adding one half and one third?','medium','curiosity'),
('probability','I do not know how to calculate the probability of two independent events.','low','confusion'),
('probability','This probability problem is so frustrating. I have tried several times.','low','frustration'),
('probability','I know how independent events work. Give me a challenge.','high','confidence'),
('probability','Why do we multiply the probabilities?','medium','curiosity'),
('linear equations','I feel nervous about this equation test. I need a simple explanation.','low','nervousness'),
('fractions','I am worried that I will fail the fraction quiz.','low','nervousness'),
('probability','I am excited to learn a more difficult probability problem.','high','excitement'),
('linear equations','Thanks, that explanation was helpful.','high','gratitude'),
('fractions','Please repeat the previous step.','medium','neutral'),
('probability','What does sample space mean?','medium','curiosity'),
('linear equations','I am lost. Can we start from the basics?','low','confusion'),
('fractions','I can do this now, but I want to check my method.','high','confidence'),
('probability','I cannot figure this out. It makes me angry.','low','anger'),
('linear equations','I want to know what happens when there are brackets.','high','curiosity'),
('fractions','I have no idea what the numerator means.','low','confusion'),
('probability','Can you give me a short practice question?','medium','neutral'),
]
scenarios=pd.DataFrame(SCENARIOS,columns=['skill','student_text','scenario_knowledge_level','author_intended_affect'])
scenarios.insert(0,'scenario_id',[f'S{i:03d}' for i in range(1,len(scenarios)+1)])
assert scenarios.scenario_id.is_unique
scenarios['knowledge_probability']=scenarios.scenario_knowledge_level.map(STATE_LEVELS)
scenarios.to_csv(OUT/'researcher_authored_scenarios.csv',index=False)
print('Scenario count:',len(scenarios));display(scenarios.head())
''')
md('''## 7. Pedagogical response generator with explicit grounding and safeguards
A deterministic, auditable **conversational response policy**, not a fine-tuned LLM. It uses verified topic-specific micro-lessons, a knowledge-level action, and optional *predicted* emotion. This is a controlled Phase IV prototype; fluent generative-LLM tutoring and human educational evaluation remain future work.''')
code('''LESSONS={
 'linear equations':{'core':'To solve 2x + 3 = 11, subtract 3 from both sides, then divide both sides by 2.','hint':'First isolate the term containing x by subtracting 3 on both sides.','challenge':'Try solving 3(x + 2) = 18 and check by substitution.','check':'What operation undoes adding 3?'},
 'fractions':{'core':'To add 1/2 and 1/3, use denominator 6: 3/6 + 2/6 = 5/6.','hint':'Find the least common denominator of 2 and 3 before adding.','challenge':'Try 3/4 + 2/5 and simplify your result.','check':'What common denominator works for 1/2 and 1/3?'},
 'probability':{'core':'For independent events A and B, P(A and B) = P(A) × P(B).','hint':'Check whether the events are independent before multiplying their probabilities.','challenge':'A fair coin is tossed twice. What is the chance of two heads?','check':'If P(A)=1/2 and P(B)=1/3 and they are independent, what is P(A and B)?'},
}
ACK={
 'confusion':'It is understandable to need another explanation.',
 'frustration':'Repeated mistakes can feel frustrating; we can work through one step.',
 'annoyance':'We can slow down and focus on one step.',
 'anger':'We can approach this calmly, one step at a time.',
 'nervousness':'We can begin with a small, manageable step.',
 'fear':'We can start with a simpler example.',
 'sadness':'We can work through this together at a comfortable pace.',
 'curiosity':'That is a useful question to explore.',
 'excitement':'Great enthusiasm; let us extend the idea.',
 'gratitude':'You are welcome; let us check the idea.',
 'neutral':'Let us work through the concept.',
}
NEG={'confusion','frustration','annoyance','anger','nervousness','fear','sadness','grief','disappointment','embarrassment','remorse'}
def respond(skill, text, knowledge_prob=None, emotion=None, use_knowledge=True, use_emotion=True):
    lesson=LESSONS[skill]
    if not use_knowledge: level='medium'
    elif knowledge_prob<.45: level='low'
    elif knowledge_prob>.75: level='high'
    else: level='medium'
    if level=='low': content=lesson['core'];action='worked_example'
    elif level=='high': content=lesson['challenge'];action='challenge'
    else: content=lesson['hint'];action='hint'
    if use_emotion and emotion in NEG and level=='high':
        content=lesson['hint'];action='hint'  # reduce difficulty for predicted distress
    ack=(ACK.get(emotion,'Let us work through the concept.') if use_emotion else 'Let us work through the concept.')
    response=f'{ack} {content} {lesson["check"]}'
    return {'response':response,'action':action,'knowledge_level':level,'used_emotion':emotion if use_emotion else None}

pred=emotion_infer(scenarios.student_text.tolist())
scenarios['pred_emotion']=[p['pred_emotion'] for p in pred]
scenarios['emotion_confidence']=[p['emotion_confidence'] for p in pred]
scenarios['emotion_source']='RoBERTa prediction on researcher-authored text (not observed learner emotion)'
print('Example predicted emotion and response:')
r=scenarios.iloc[0];print(r.student_text,'=>',r.pred_emotion,respond(r.skill,r.student_text,r.knowledge_probability,r.pred_emotion)['response'])
''')
md('''## 8. Four model variants and controlled ablation
The same authored scenarios are used for all variants. This isolates the effect of toggling knowledge and predicted-emotion inputs; it does **not** estimate causal learning improvements.''')
code('''VARIANTS={'generic':(False,False),'knowledge_only':(True,False),'emotion_only':(False,True),'knowledge_plus_predicted_emotion':(True,True)}
records=[]
for _,s in scenarios.iterrows():
    for variant,(uk,ue) in VARIANTS.items():
        out=respond(s.skill,s.student_text,s.knowledge_probability,s.pred_emotion,uk,ue)
        records.append({'scenario_id':s.scenario_id,'variant':variant,'skill':s.skill,'student_text':s.student_text,'scenario_knowledge_level':s.scenario_knowledge_level,'knowledge_probability':s.knowledge_probability,'author_intended_affect':s.author_intended_affect,'pred_emotion':s.pred_emotion,'emotion_confidence':s.emotion_confidence,**out})
results=pd.DataFrame(records)
results.to_csv(OUT/'all_variant_responses.csv',index=False)
print('Generated responses:',len(results));display(results.head(8))
''')
md('''## 9. Evaluation: transparent policy checks, not invented human judgments
Automatic checks: (a) grounding in the pre-authored topic lesson, (b) explicit follow-up check, (c) action conformity to **implemented** rules, (d) response length. Since these rules are hard-coded, high scores measure **implementation consistency**, not pedagogical effectiveness. No BLEU/ROUGE against fabricated reference answers and no claims of student gains.''')
code('''def policy_expected_action(s,uk,ue):
    k=s.knowledge_probability if uk else .5
    lev='low' if k<.45 else ('high' if k>.75 else 'medium')
    if lev=='low': action='worked_example'
    elif lev=='high': action='challenge'
    else: action='hint'
    if ue and s.pred_emotion in NEG and lev=='high': action='hint'
    return action

def rubric(row):
    s=scenarios.set_index('scenario_id').loc[row.scenario_id]
    uk,ue=VARIANTS[row.variant]
    l=LESSONS[row.skill]
    expected=policy_expected_action(s,uk,ue)
    return pd.Series({
      'grounded_topic':int(any(x in row.response for x in [l['core'],l['hint'],l['challenge']])),
      'contains_followup_check':int(l['check'] in row.response),
      'policy_conformity':int(row.action==expected),
      'words':len(row.response.split()),
      'mentions_emotion_ack':int(ue and row.response.startswith(ACK.get(s.pred_emotion,'Let us work through the concept.'))),
      'knowledge_condition_used':int(uk),
      'predicted_emotion_condition_used':int(ue),
    })
rub=results.apply(rubric,axis=1)
results=pd.concat([results,rub],axis=1)
results.to_csv(OUT/'all_variant_responses_scored.csv',index=False)
summary=results.groupby('variant').agg(n=('scenario_id','size'),grounded_topic_rate=('grounded_topic','mean'),followup_check_rate=('contains_followup_check','mean'),policy_conformity_rate=('policy_conformity','mean'),mean_response_words=('words','mean'),emotion_ack_rate=('mentions_emotion_ack','mean')).reset_index()
summary.to_csv(OUT/'policy_check_summary.csv',index=False)
display(summary)
''')
md('''## 10. Robustness: label perturbation and emotion-domain shift checks
Measure how often the policy action changes if the predicted emotion is perturbed, and compare RoBERTa predicted labels with the **author-intended scenario labels only as a diagnostic**. This is **not a validated emotion-classification benchmark**.''')
code('''# Scenario-intended labels do not necessarily match the 28-class GoEmotions taxonomy.
# Do not compute 'accuracy' from non-equivalent labels such as confidence/frustration.
scenarios['goemotions_label_in_taxonomy']=scenarios.author_intended_affect.isin(labels)
valid=scenarios[scenarios.goemotions_label_in_taxonomy].copy()
scenario_match_rate=float((valid.author_intended_affect==valid.pred_emotion).mean()) if len(valid) else None
print('Diagnostic exact agreement on compatible author-intended labels:',scenario_match_rate,'n=',len(valid))
print('Not a real educational-domain accuracy estimate: scenario text and intent were authored by researchers.')
perturb=[]
for _,s in scenarios.iterrows():
    for new_emotion in ['neutral','confusion','annoyance','curiosity']:
        original=respond(s.skill,s.student_text,s.knowledge_probability,s.pred_emotion)
        altered=respond(s.skill,s.student_text,s.knowledge_probability,new_emotion)
        perturb.append({'scenario_id':s.scenario_id,'original_emotion':s.pred_emotion,'replacement_emotion':new_emotion,'action_changed':int(original['action']!=altered['action']),'response_changed':int(original['response']!=altered['response'])})
perturb=pd.DataFrame(perturb)
perturb.to_csv(OUT/'emotion_perturbation.csv',index=False)
print('Action sensitivity under counterfactual emotion input:',perturb.action_changed.mean())
print('Response sensitivity under counterfactual emotion input:',perturb.response_changed.mean())
''')
md('## 11. Charts and example dialogues')
code('''plt.figure(figsize=(10,4));plt.bar(summary.variant,summary.mean_response_words);plt.ylabel('Mean response words');plt.xticks(rotation=20,ha='right');plt.title('Response length by ablation');plt.tight_layout();plt.savefig(OUT/'response_length_ablation.png',dpi=160);plt.show()
actions=pd.crosstab(results.variant,results.action,normalize='index')
actions.to_csv(OUT/'action_distribution.csv')
ax=actions.plot(kind='bar',stacked=True,figsize=(10,5));ax.set_ylabel('Proportion');ax.set_title('Instructional actions by variant');plt.xticks(rotation=20,ha='right');plt.tight_layout();plt.savefig(OUT/'action_distribution.png',dpi=160);plt.show()
plt.figure(figsize=(8,4));scenarios.pred_emotion.value_counts().plot(kind='bar');plt.ylabel('Number of authored scenarios');plt.title('RoBERTa predictions on authored educational prompts');plt.tight_layout();plt.savefig(OUT/'scenario_predicted_emotions.png',dpi=160);plt.show()
for sid in scenarios.scenario_id.iloc[:3]:
    print('\\nSCENARIO',sid)
    for _,r in results[results.scenario_id==sid].iterrows(): print(' ',r.variant,':',r.response)
''')
md('''## 12. Export reproducibility metadata, model checkpoint, and downloadable results ZIP
Model checkpoint is exported separately because RoBERTa weights can be hundreds of MB. The small results ZIP contains predictions, policy checks, charts, metadata, and the Phase III model artifacts. Optionally archive RoBERTa weights to Google Drive.''')
code('''metadata={
 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
 'experiment':'AKC-Tutor Phase IV controlled response policy ablation',
 'scenario_count':len(scenarios),'variants':list(VARIANTS),'seed':SEED,
 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
 'torch':torch.__version__,'transformers_model':MODEL_ID,'sklearn':sklearn.__version__,
 'phase3_source':meta3['dataset_source'],'phase3_sha256':meta3['dataset_sha256'],
 'phase3_model_file':'knowledge_plus_behavior_proxy.joblib',
 'phase3_heldout_probability_distribution_used':True,
 'phase3_model_loaded':True,
 'phase3_model_used_for_new_scenario_inference':False,
 'phase3_inference_limitation':'No matched scenario-level historical learner features; uses observed Phase III test probabilities as illustrative state conditions',
 'emotion_checkpoint_provenance':'uploaded checkpoint' if checkpoint_loaded else 'retrained on official GoEmotions train with dev epoch selection',
 'emotion_predictions_are_not_ground_truth':True,
 'scenario_dialogues_researcher_authored':True,
 'scenario_dialogue_matched_to_phase3_students':False,
 'pedagogical_policy':'auditable controlled template-based response policy; not a trained generative LLM',
 'evaluations':'mechanical rule conformance and response-action sensitivity, not human-rated pedagogical quality or learning gain',
 'limitation':'Do not claim verified student emotions, multimodal learner alignment, or improvement in learning outcomes.'}
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
(OUT/'README.txt').write_text('AKC-Tutor Phase IV. Controlled researcher-authored dialogue scenarios and rule-based instructional responses. Emotion is RoBERTa-predicted; knowledge states are illustrative probabilities drawn from Phase III held-out model outputs, NOT matched to dialogue. No learning-gain claims. Read run_metadata.json.\\n')
with zipfile.ZipFile(OUT/'AKC_Tutor_Phase4_Results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir():
        if f.is_file() and f.suffix in {'.json','.csv','.png','.txt'}: z.write(f,f.name)
    with zipfile.ZipFile(phase3_zip) as src:
        for name in ['knowledge_plus_behavior_proxy.joblib','knowledge_only.joblib','feature_definitions.json']:
            z.writestr('phase3_artifacts/'+name,src.read(name))
print('Results archive:',OUT/'AKC_Tutor_Phase4_Results.zip')
print('Model checkpoint directory:',EMODIR)
files.download(str(OUT/'AKC_Tutor_Phase4_Results.zip'))
''')
md('''## After execution
Upload `AKC_Tutor_Phase4_Results.zip` to ChatGPT. If you want to preserve your Phase II emotion checkpoint for later reuse, save the **entire** `/content/AKC_Phase4/models/emotion_roberta` directory to Google Drive; do not rely on the temporary Colab runtime. This notebook deliberately does **not** report improvements in learning gains, because no learners were evaluated. For a publishable affect-aware tutor claim, the next phase requires matched learner-dialogue evidence and blinded human pedagogical assessment.''')
b.cells=b.cells
n.write(b,'/mnt/data/AKC_Tutor_Phase4_Automated_Colab.ipynb')
print(len(b.cells))
