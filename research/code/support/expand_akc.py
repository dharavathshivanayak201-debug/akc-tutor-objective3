from pathlib import Path
import re,subprocess,zipfile
p=Path('/mnt/data/AKC_Tutor_IEEE_Paper/AKC_Tutor_IEEE_Full_Paper.tex'); s=p.read_text()
rows='''zhu24|Meta-path structured graph pre-training for improving knowledge tracing in intelligent tutoring|Expert Systems with Applications|10.1016/j.eswa.2024.124451
huang24causal|Learning consistent representations with temporal and causal enhancement for knowledge tracing|Expert Systems with Applications|10.1016/j.eswa.2023.123128
xu23|Improving knowledge tracing via a heterogeneous information network enhanced by student interactions|Expert Systems with Applications|10.1016/j.eswa.2023.120853
sun24|Target hierarchy-guided knowledge tracing: Fine-grained knowledge state modeling|Expert Systems with Applications|10.1016/j.eswa.2024.123898
huang24speed|Response speed enhanced fine-grained knowledge tracing: A multi-task learning perspective|Expert Systems with Applications|10.1016/j.eswa.2023.122107
zhao23|Exploiting multiple question factors for knowledge tracing|Expert Systems with Applications|10.1016/j.eswa.2023.119786
yu24emotion|Bridging computer and education sciences: A systematic review of automated emotion recognition in online learning environments|Computers and Education|10.1016/j.compedu.2024.105111
huang24moe|Pull together: Option-weighting-enhanced mixture-of-experts knowledge tracing|Expert Systems with Applications|10.1016/j.eswa.2024.123419
duan24|Towards more accurate and interpretable model: Fusing multiple knowledge relations into deep knowledge tracing|Expert Systems with Applications|10.1016/j.eswa.2023.122573
liu24hyper|Hyperbolic embedding of discrete evolution graphs for intelligent tutoring systems|Expert Systems with Applications|10.1016/j.eswa.2023.122451
park24concept|Enhancing knowledge tracing with concept map and response disentanglement|Knowledge-Based Systems|10.1016/j.knosys.2024.112346
yu25graph|Exploring long- and short-term knowledge state graph representations with adaptive fusion for knowledge tracing|Information Processing and Management|10.1016/j.ipm.2025.104074
sun24progress|Progressive knowledge tracing: Modeling learning process from abstract to concrete|Expert Systems with Applications|10.1016/j.eswa.2023.122280
yang24hetero|Heterogeneous graph-based knowledge tracing with spatiotemporal evolution|Expert Systems with Applications|10.1016/j.eswa.2023.122249
liu26contrast|Heterogeneous graph knowledge tracing with novel contrastive view and meta-path contexts|Expert Systems with Applications|10.1016/j.eswa.2025.128840
yang25mahkt|MAHKT: Knowledge tracing with multi-association heterogeneous graph embedding based on knowledge transfer|Knowledge-Based Systems|10.1016/j.knosys.2025.112958
xu26sqg|SQGKT: Student-question interaction graph-based knowledge tracing|Expert Systems with Applications|10.1016/j.eswa.2025.129175
luo24|An efficient state-aware coarse-fine-grained model for knowledge tracing|Knowledge-Based Systems|10.1016/j.knosys.2024.112375
wang25states|Learning states enhanced knowledge tracing: Simulating the diversity in real-world learning process|Expert Systems with Applications|10.1016/j.eswa.2025.126838
wu25cognitive|Graph knowledge tracing in cognitive situation: Validation of classic assertions in cognitive psychology|Knowledge-Based Systems|10.1016/j.knosys.2025.113281
wu24program|Programming knowledge tracing based on heterogeneous graph representation|Knowledge-Based Systems|10.1016/j.knosys.2024.112161
xiao26|Information processing dynamic graph for knowledge tracing|Information Processing and Management|10.1016/j.ipm.2025.104331
liebenow25|Self-assessment accuracy in the age of artificial intelligence: Differential effects of LLM-generated feedback|Computers and Education|10.1016/j.compedu.2025.105385
zhang24video|Guiding student learning in video lectures: Effects of instructors' emotional expressions and visual cues|Computers and Education|10.1016/j.compedu.2024.105062
affect26|Affective computing in online higher education: A systematic literature review|Computers and Education: Artificial Intelligence|10.1016/j.caeai.2025.100499'''
refs=[x.split('|') for x in rows.splitlines()]
related=r'''\subsection{Knowledge Tracing and Behavioral History}
Recent knowledge tracing studies represent student--skill relations using heterogeneous interaction graphs \cite{xu23}. Target hierarchies permit finer-grained state estimates \cite{sun24}. Question factors, including difficulty, add contextual information beyond correctness history \cite{zhao23}. Response speed can be modeled as a complementary learning signal \cite{huang24speed}. Temporal and causal enhancements address uneven exposure to concepts \cite{huang24causal}. Graph pretraining seeks more informative representations from sparse interaction histories \cite{zhu24}.

Option-aware mixtures of experts exploit more than binary correctness \cite{huang24moe}. Multiple knowledge relations can be fused into tracing models \cite{duan24}. Hyperbolic embeddings represent relations among tutoring entities \cite{liu24hyper}. Concept maps and unchosen response options may help characterize errors \cite{park24concept}. Long- and short-term knowledge graphs model different time scales \cite{yu25graph}. Progressive models distinguish stages of the learning process \cite{sun24progress}. Spatiotemporal graph models follow changing concept relations \cite{yang24hetero}.

Recent heterogeneous graph tracing combines contrastive views and meta-path contexts \cite{liu26contrast}. Transfer-aware graph representations connect knowledge concepts \cite{yang25mahkt}. Student--question graphs incorporate attempts and hints \cite{xu26sqg}. Coarse--fine-grained models examine the granularity of latent states \cite{luo24}. Learning-state representations model diverse response patterns \cite{wang25states}. Cognitive-situation models link tracing to psychological constructs \cite{wu25cognitive}. Programming-specific graphs show the importance of task context \cite{wu24program}. Information-processing graphs provide an alternative account of sequential performance \cite{xiao26}. These studies motivate our feature ablation; none of their architectures is claimed as an implemented comparator.

\subsection{Educational Affect and Feedback}
A review of automated emotion recognition identifies limited evidence of deployment in authentic learning environments \cite{yu24emotion}. A review of affective computing in online higher education also reports a gap between technical detection and educational application \cite{affect26}. Studies of instructors' emotional expressions distinguish attentional effects from knowledge retention \cite{zhang24video}. Experimental work on LLM-generated feedback reports that its benefits may depend on learners' initial self-assessment \cite{liebenow25}. These findings motivate evaluating response appropriateness separately from measured learning outcomes.
'''
s=s.replace('\\section{AKC-Tutor Framework}',related+'\n\\section{AKC-Tutor Framework}')
methods=r'''\subsection{Research Questions and Leakage Controls}
The experiments address three questions: (RQ1) whether a pretrained encoder improves fine-grained text emotion classification; (RQ2) whether past behavioral features add predictive information to knowledge history on unseen learners; and (RQ3) whether affect-conditioned phrasing improves rated response suitability compared with knowledge-only tutoring. The answers come from distinct datasets, not a single jointly trained learner model.

The emotion experiments retain a held-out test set. The educational split excludes test learners from training, and history features use past interactions only. The response comparison pairs all four policies within each scenario. Pairing limits between-scenario variation but does not remove biases associated with researcher-written templates or rubric interpretation.
'''
s=s.replace('\\section{Results and Discussion}',methods+'\n\\section{Results and Discussion}')
interpret=r'''\subsection{Cross-Phase Interpretation}
The RoBERTa gain in macro-F1 supports transformer-based text representation, but the final macro-F1 below 0.50 indicates remaining class-level errors. Reddit language cannot be assumed to represent classroom discourse. Phase IV therefore uses predicted affect as an illustrative conditioning signal, not a verified measurement of a learner's emotional state.

In Phase III, the improvement in AUC is supported by learner-cluster resampling and disappears under the shuffled-behavior control. The result suggests that temporal activity patterns contain information beyond knowledge-history variables. Nevertheless, attempts, timing, and similar behavioral proxies can also reflect task difficulty and platform usage. They should not be renamed emotional observations.

Phase IV's automated checks establish policy conformance, not pedagogical efficacy. Phase V addresses a different question by collecting human ratings. The combined condition improves affective appropriateness, but the other three criteria show no significant change after Holm correction. This pattern is consistent with affect-conditioned wording rather than a change in instructional decisions. The many zero paired differences also constrain sensitivity to small effects.

\subsection{Reproducibility and Reporting}
Reproduction requires preserving the GoEmotions single-label filter and held-out partition, performing learner-disjoint educational splitting before model fitting, and ensuring that feature windows use past events only. Each tutoring scenario must have one response for each of four policy variants. Evaluator workbooks should be joined on response IDs, and paired confidence intervals should resample scenarios rather than individual rubric scores. Independent provenance of evaluators and effectiveness in real learners remain separate validation requirements.
'''
s=s.replace('\\section{Limitations and Threats to Validity}',interpret+'\n\\section{Limitations and Threats to Validity}')
s=s.replace('A recent systematic review of empirical large-language-model applications in education also reports heterogeneous benefits and evaluation challenges \\cite{llmreview2026}.','Studies of AI feedback show that apparently suitable responses need not improve learning outcomes \\cite{liebenow25}.')
bib='\\begin{thebibliography}{99}\n'
for k,title,journal,doi in refs:
    bib+=f'\\bibitem{{{k}}} ``{title},\'\' \\emph{{{journal}}}, doi: {doi}.\n'
bib+=r'''\bibitem{piech2015} C. Piech \emph{et al.}, ``Deep Knowledge Tracing,'' in \emph{NeurIPS}, 2015.
\bibitem{demszky2020} D. Demszky \emph{et al.}, ``GoEmotions: A dataset of fine-grained emotions,'' in \emph{ACL}, 2020, doi: 10.18653/v1/2020.acl-main.372.
\bibitem{liu2019} Y. Liu \emph{et al.}, ``RoBERTa: A robustly optimized BERT pretraining approach,'' arXiv:1907.11692, 2019.
\bibitem{wang2024} S. Wang \emph{et al.}, ``Artificial intelligence in education: A systematic literature review,'' \emph{Expert Systems with Applications}, vol. 252, Art. 124167, 2024, doi: 10.1016/j.eswa.2024.124167.
\end{thebibliography}'''
s=re.sub(r'\\begin\{thebibliography\}\{00\}.*?\\end\{thebibliography\}',lambda m:bib,s,flags=re.S)
p.write_text(s)
print('Journal entries',len(refs)+1)
