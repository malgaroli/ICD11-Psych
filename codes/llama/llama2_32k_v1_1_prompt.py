import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, TextStreamer, PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional


tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct",
    trust_remote_code=True, torch_dtype=torch.float16,  device_map='auto')

#1114 tokens
prompt = """
<s> 
[INST] <<SYS>> "You are an AI assistant. User will you give you a task. Your goal is to use given guidelines to infer mental disorder of patient described in vignette. Make your response as concise as possible.

[guidelines]: Q1 Generalized Anxiety Disorder

Marked symptoms of anxiety accompanied by either:
general apprehensiveness that is not restricted to any particular environmental circumstance (i.e., “free-floating anxiety”); or
worry (apprehensive expectation) about untoward events occurring in several different aspects of everyday life (e.g., work, finances, health, family). Anxiety and general apprehensiveness or worry are accompanied by additional symptoms, such as:
Muscle tension or motor restlessness. Sympathetic autonomic overactivity as evidenced by frequent gastrointestinal symptoms such as nausea and/or abdominal distress, heart palpitations, sweating, trembling, shaking, and/or dry mouth. Subjective experience of nervousness, restlessness, or being “on edge”. Difficulties maintaining concentration. Irritability. Sleep disturbances (difficulty falling or staying asleep, or restless, unsatisfying sleep). The symptoms are not transient and persist for at least several months, for more days than not. The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q2  Panic Disorder

Recurrent unexpected panic attacks that are not restricted to particular stimuli or situations. Panic attacks are discrete episodes of intense fear or apprehension also characterized by the rapid and concurrent onset of several characteristic symptoms. These symptoms may include, but are not limited to, the following:
Palpitations or increased heart rate. Sweating. Trembling. Sensations of shortness of breath. Feelings of choking. Chest pain. Nausea or abdominal distress. Feelings of dizziness or lightheadedness. Chills or hot flushes. Tingling or lack of sensation in extremities (i.e., paresthesias). Depersonalization or derealization. Fear of losing control or going mad. Fear of imminent death
Panic attacks are followed by persistent concern or worry (e.g., for several weeks) about their recurrence or their perceived negative significance (e.g., that the physiological symptoms may be those of a myocardial infarction), or behaviours intended to avoid their recurrence (e.g., only leaving the home with a trusted companion). The symptoms are sufficiently severe to result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning. Panic attacks can occur in other Anxiety and Fear-Related Disorders as well as other Mental and Behavioural Disorders and therefore the presence of panic attacks is not in itself sufficient to assign a diagnosis of Panic Disorder.

[vignette]:
JV is a 22-year-old female university student. She presented to her university’s counselling center because of difficulty she is anticipating in her public speaking class.

Presenting Symptoms
In her initial session, JV explains that she is very concerned of having to give a speech in front of a class. The university requires that she take a public speaking course in order to graduate; she says she has postponed taking the class until her final year in school because she dreads the idea of it. However, she knows that she needs to find a way to complete it in order to finish her degree. She came to the counselling center in hopes of finding a way to cope with her anxiety about giving the speech. She explains that she has always been a shy person, and feels uncomfortable if she is the center of attention. 
JV’s teachers consider her to be a quiet but capable student. She says that she rarely raises her hand in class, but she will answer if she were called upon. She has had to give brief presentations in other classes, and has been able to somehow get through it, but it has always made her uncomfortable. She states that she feels very nervous before a presentation, with “butterflies in her stomach,” but calms down quickly after it is finished. However, she has never had to give a presentation to a group as large as the public speaking course with such important consequences, and feels quite anxious that she will not be able to perform satisfactorily and will therefore not be able to graduate.
JV reports that she has a group of friends, but has never been “the talkative one.” JV likes to do things with the group, but rarely speaks up because she finds it hard to break into the stream of conversation. Usually, she prefers to listen. When the clinician asks JV if she is experiencing any anxiety in the current session, JV replies that she feels fine. She elaborates that she usually does well in one-on-one situations; she only starts to feel uncomfortable when the group gets larger, “like more than 5 or 6 people.” She has a boyfriend, and she reports that the relationship is going well. She says he has a similar preference for small groups and quiet activities. When the clinician asks if she would like to change her shyness, JV states that it usually does not cause her any problems and she likes who she is as a person. She emphasizes that her only concern is making it through her required speech course. She says she is ready to tackle the problem, but wants some support and suggestions for how to do so.
JV is an only child whose parents had her relatively late in life and live in a nearby town. She describes her family life as “quiet but normal.” She denies using any drugs, but states that she occasionally drinks one or two alcoholic beverages with her boyfriend on the weekends. A recent physical examination revealed no abnormalities.\n" <</SYS>>\n\n 

\n[/INST]\n\n
[RESPONSE]:
"""
device = torch.device("cuda")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.long)
input_ids = input_ids.to(device)
generation_output = model.generate(
    input_ids=input_ids,
    max_length=1024,
    num_beams=1,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
)

print(tokenizer.decode(generation_output[0], skip_special_tokens=True))

