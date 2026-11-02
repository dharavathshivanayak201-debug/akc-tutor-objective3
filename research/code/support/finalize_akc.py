from pathlib import Path
import re,subprocess,zipfile
p=Path('/mnt/data/AKC_Tutor_IEEE_Paper/AKC_Tutor_IEEE_Full_Paper.tex');s=p.read_text()
authors={'zhu24':'M. Zhu, L. Qiu, and J. Zhou','huang24causal':'C. Huang et al.','xu23':'J. Xu et al.','sun24':'X. Sun et al.','huang24speed':'T. Huang et al.','zhao23':'Y. Zhao et al.','yu24emotion':'S. Yu et al.','huang24moe':'T. Huang et al.','duan24':'Z. Duan et al.','liu24hyper':'S. Liu et al.','park24concept':'S. Park et al.','yu25graph':'G. Yu et al.','sun24progress':'J. Sun et al.','yang24hetero':'H. Yang et al.','liu26contrast':'J. Liu et al.','yang25mahkt':'H. Yang et al.','xu26sqg':'T. Xu et al.','luo24':'H. Luo et al.','wang25states':'S. Wang et al.','wu25cognitive':'Q. Wu et al.','wu24program':'Y. Wu et al.','xiao26':'X. Xiao et al.','liebenow25':'L. W. Liebenow et al.','zhang24video':'C. Zhang et al.'}
for key,author in authors.items():s=s.replace('\\bibitem{'+key+'} ``','\\bibitem{'+key+'} '+author+', ``')
# unknown authors: leave title-first rather than fabricate
section=r'''\subsection{Operational Policy Design}
The policy comparison separates \emph{action selection} from \emph{surface realization}. Let $q\in\{L,M,H\}$ be the supplied knowledge band and $e$ the predicted emotion label. The knowledge-conditioned instructional action is $a=g(q)$; in the combined condition, the response is $r=T(a,e)$, where $T$ is a deterministic researcher-written template. For knowledge-only, $r=T(a,\varnothing)$; for emotion-only, $r=T(a_0,e)$ with a fixed generic instructional action $a_0$. This design tests whether affect changes response wording while holding the instructional decision rule explicit.

The experimental design is balanced: $24$ scenarios $\times$ $4$ policies $=96$ responses. Scenarios span fractions, linear equations, and probability. Because the supplied knowledge estimates are illustrative rather than inferred from the scenario speakers, we do not treat the responses as personalized interactions with actual learners. This distinction prevents a circular claim that separate predictive modules were jointly validated.

\subsection{Statistical Estimands}
For a rubric dimension $d$, let $r_{isjd}$ denote the score from evaluator $j$ for scenario $i$ under policy $s$. The scenario-level score is $\bar r_{isd}=\frac{1}{2}\sum_{j=1}^{2}r_{isjd}$. The primary effect is
\begin{equation}
\widehat\Delta_d=\frac{1}{24}\sum_{i=1}^{24}(\bar r_{i,\mathrm{combined},d}-\bar r_{i,\mathrm{knowledge},d}).
\end{equation}
Resampling the $24$ scenario indices preserves the matched comparison and within-scenario dependence. The confidence interval summarizes scenario variability; the sign-flip null assumes exchangeability of paired effect signs. Holm adjustment addresses four prespecified rubric comparisons. Weighted $\kappa$ is computed separately by dimension to account for ordinal distances between evaluator ratings.
'''
s=s.replace('\\section{Results and Discussion}',section+'\n\\section{Results and Discussion}')
# additional figure for phase4 process, actual generated response count
fig=r'''\begin{figure}[t]\centering
\setlength{\fboxsep}{3pt}\fbox{\begin{minipage}{0.92\columnwidth}\centering\footnotesize
24 researcher-authored scenarios\\[3pt]$\Downarrow$\\[3pt]
\begin{tabular}{cc}\fbox{Generic} & \fbox{Knowledge only}\\[4pt]\fbox{Emotion only} & \fbox{Combined}\end{tabular}\\[4pt]$\Downarrow$\\[3pt]
96 deterministic responses $\rightarrow$ two blinded rating workbooks\\[3pt]
4 criteria $\times$ 2 evaluators; scenario-paired inference
\end{minipage}}
\caption{Phase IV--V controlled response generation and evaluation design.}\label{fig:policy}\end{figure}
'''
s=s.replace('\\subsection{Phase V: Human Rubric and Inference}',fig+'\\subsection{Phase V: Human Rubric and Inference}')
p.write_text(s)
subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',p.name],cwd=p.parent,stdout=subprocess.DEVNULL,check=True)
subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',p.name],cwd=p.parent,stdout=subprocess.DEVNULL,check=True)
print(subprocess.check_output(['pdfinfo',str(p.with_suffix('.pdf'))]).decode().split('Pages:')[1].splitlines()[0].strip())
