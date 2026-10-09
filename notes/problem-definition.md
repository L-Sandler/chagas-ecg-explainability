Level 1, the score. "Across many patients, the model's predicted Chagas probability is higher for patients with prevalence of (RBBB, LAFB, combo of LAFB and RBBB, AVB, AF, flutter, or VE) than for patients without it."
- If the model performs well in [metric], but does not align with common ECG abnormalities in predicting chagas, then there is a chance that a new unexplained signal is discovered. Also could be some form of error in training or model takes unintended shortcut?

Level 2 (fill in after some iteration), the explanation. "For patients with [finding], the model's attribution concentrates on ___," where the blank is the part of the heartbeat and the leads where that finding shows up on an ECG.

Audience: Technical reader who is looking to try and implement more trustworthy AI systems. 
Importance: AI systems are a large unknown for most people. If they are going to be implemented in critical scenarios like healthcare, then there must be trust built in order to support their adoption. 