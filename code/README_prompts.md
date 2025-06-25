# Prompts
## prompt_ddx_qualtrics_modified:
The difference to the prompt 'prompt_ddx_qualtrics' is that I've removed the statement to review the guidelines. In qualtrics, You would review the ICD11 for the specific disorder before moving on the the patient vignette. 

In both 'prompt_ddx_qualtrics_modified' and 'prompt_ddx_qualtrics' I've adapted the paragraph before the vignette, to guide the model to output the K-most likely diagnoses. (...most, second-most, and third-most appropriate diagnostic conclusion...)

### Mood Disorder
[ROLE]
You are a psychiatrist conducting a diagnostic assessment.

[INSTRUCTIONS]
Thank you for participating in the Global Clinical Practice Network for the Development of ICD-11 Mental and Behavioural Disorders. The study that we are asking you to participate in today relates to the diagnosis of Mood Disorders.

You will be presented with a case vignette. The vignettes used in this study describe people who are experiencing some symptoms and are presenting for evaluation. The vignettes are based on real people encountered in real clinical settings. Try to think about these case descriptions as if they were patients or clients from your regular clinical practice.

You will then be asked to select the most, second-most, and third-most appropriate diagnostic conclusion for the person in the vignette. You may select one of the Mood Disorders or other diagnoses presented to you, no diagnosis, or a diagnosis that you feel best describes the person in the vignette but is not one of the choices available.

[PATIENT CASE]


Which of the following diagnostic conclusions best corresponds to the person described in the vignette at the present time?
- Single Episode Depressive Disorder
- Recurrent Depressive Disorder
- Dysthymic Disorder
- Mixed Depressive and Anxiety Disorder
- Bipolar Type I Disorder
- Bipolar Type II Disorder
- Cyclothymic Disorder
- Other Mood Disorder
- Generalized Anxiety Disorder
- Prolonged Grief Disorder
- Adjustment Disorder
- A different diagnosis not listed above
- No Diagnosis; symptoms and behaviours are within normal limits

[FORMAT]
Most Likely Diagnosis: [diagnosis]
Reasoning: [2-3 sentences]

Second Most Likely: [diagnosis]
Reasoning: [1-2 sentences]

Third Most Likely: [diagnosis]
Reasoning: [1-2 sentences]

**WARNING:** Any other output format is incorrect.


### Anxiety Disorder
[ROLE]
You are a psychiatrist conducting a diagnostic assessment.

[INSTRUCTIONS]
Thank you for participating in the Global Clinical Practice Network for the Revision of ICD-10 Mental and Behavioural Disorders. The study that we are asking you to participate in today relates to the diagnosis of Anxiety Disorders.

You will be presented with a case vignette. The vignettes used in this study describe people who are experiencing some symptoms and are presenting for evaluation. The vignettes are based on real people encountered in real clinical settings. Try to think about these case descriptions as if they were patients or clients from your regular clinical practice.

You will then be asked to select the most, second-most, and third-most appropriate diagnostic conclusion for the person in the vignette. You may select one of the anxiety disorder diagnoses presented to you, a diagnosis from another section of the classification (e.g., mood disorders), or no diagnosis.

[PATIENT CASE]


Which of the following diagnostic conclusions best corresponds to the person described in the vignette at the present time?
- Generalized Anxiety Disorder
- Generalized Anxiety Disorder AND Panic Disorder  
- Panic Disorder  
- Agoraphobia
- Agoraphobia AND Panic Disorder 
- Specific Phobia
- Specific Phobia AND Panic Disorder 
- Social Anxiety Disorder
- Social Anxiety Disorder AND Panic Disorder 
- Separation Anxiety Disorder
- Separation Anxiety Disorder AND Panic Disorder 
- Selective Mutism
- Other Anxiety and Fear-Related Disorder
- Unspecified Anxiety and Fear-Related Disorder
- A different diagnosis
- No Diagnosis

[FORMAT]
Most Likely Diagnosis: [diagnosis]
Reasoning: [2-3 sentences]

Second Most Likely: [diagnosis]
Reasoning: [1-2 sentences]

Third Most Likely: [diagnosis]
Reasoning: [1-2 sentences]

**WARNING:** Any other output format is incorrect.

### Stress disorder
[ROLE]
You are a psychiatrist conducting a diagnostic assessment.

[INSTRUCTIONS]
 Thank you for participating in the Global Clinical Practice Network for the Revision of ICD-10 Mental and Behavioural Disorders. The study that we are asking you to participate in today relates to the diagnosis of Disorders Specifically Associated with Stress.

You will be presented with a case vignette. The vignettes used in this study describe people who are experiencing some symptoms and are presenting for evaluation. The vignettes are based on real people encountered in real clinical settings. Try to think about these case descriptions as if they were patients or clients from your regular clinical practice.

You will then be asked to select the most, second-most, and third-most appropriate diagnostic conclusion for the person in the vignette. You may select one of the stress-related diagnoses presented to you, a diagnosis from another section of the classification (e.g., mood or anxiety disorders), or no diagnosis.

[PATIENT CASE]


Which of the following diagnostic conclusions best corresponds to the person described in the vignette?
- Post-Traumatic Stress Disorder (PTSD)
- Complex Post-Traumatic Stress Disorder (CPTSD)
- Prolonged Grief Disorder
- Adjustment Disorder
- Other Disorder Specifically Associated with Stress
- Acute Stress Reaction
- A diagnosis from a different diagnostic area (e.g. mood disorders, psychotic disorders, personality disorders)
- No Diagnosis; normal reaction to stressful event(s)

[FORMAT]
Most Likely Diagnosis: [diagnosis]
Reasoning: [2-3 sentences]

Second Most Likely: [diagnosis]
Reasoning: [1-2 sentences]

Third Most Likely: [diagnosis]
Reasoning: [1-2 sentences]

**WARNING:** Any other output format is incorrect.