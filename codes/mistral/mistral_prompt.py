import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda:0" # the device to load the model onto

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", quantization_config=bnb_config, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

prompt = """
[INST] Based on the following disorder guidelines: Q1. Generalized Anxiety Disorder, Q2. Panic Disorder, Q3. Agoraphobia, Q4. Specific Phobia, Q5. Social Anxiety Disorder, Q6. Separation Anxiety Disorder, Q7. Selective Mutism, Q8. Other Anxiety and Fear-Related Disorder, Q9. Unspecified Anxiety and Fear-Related Disorder, please write a diagnosis for the following vignette. [/INST]

Guidelines:
###
Q1 Generalized Anxiety Disorder: Essential (Required) Features: Marked symptoms of anxiety accompanied by either:
general apprehensiveness that is not restricted to any particular environmental circumstance (i.e., “free-floating anxiety”); or
worry (apprehensive expectation) about untoward events occurring in several different aspects of everyday life (e.g., work, finances, health, family). Anxiety and general apprehensiveness or worry are accompanied by additional symptoms, such as:
Muscle tension or motor restlessness. Sympathetic autonomic overactivity as evidenced by frequent gastrointestinal symptoms such as nausea and/or abdominal distress, heart palpitations, sweating, trembling, shaking, and/or dry mouth. Subjective experience of nervousness, restlessness, or being “on edge”. Difficulties maintaining concentration. Irritability. Sleep disturbances (difficulty falling or staying asleep, or restless, unsatisfying sleep). The symptoms are not transient and persist for at least several months, for more days than not. The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q2  Panic Disorder: Essential (Required) Features: Recurrent unexpected panic attacks that are not restricted to particular stimuli or situations. Panic attacks are discrete episodes of intense fear or apprehension also characterized by the rapid and concurrent onset of several characteristic symptoms. These symptoms may include, but are not limited to, the following:
Palpitations or increased heart rate. Sweating. Trembling. Sensations of shortness of breath. Feelings of choking. Chest pain. Nausea or abdominal distress. Feelings of dizziness or lightheadedness. Chills or hot flushes. Tingling or lack of sensation in extremities (i.e., paresthesias). Depersonalization or derealization. Fear of losing control or going mad. Fear of imminent death
Panic attacks are followed by persistent concern or worry (e.g., for several weeks) about their recurrence or their perceived negative significance (e.g., that the physiological symptoms may be those of a myocardial infarction), or behaviours intended to avoid their recurrence (e.g., only leaving the home with a trusted companion). The symptoms are sufficiently severe to result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning. Panic attacks can occur in other Anxiety and Fear-Related Disorders as well as other Mental and Behavioural Disorders and therefore the presence of panic attacks is not in itself sufficient to assign a diagnosis of Panic Disorder.
 
Q3  Agoraphobia: Essential (Required) Features: Marked and excessive fear or anxiety that occurs in, or in anticipation of, multiple situations where escape might be difficult or help might not be available, such as using public transportation, being in crowds, being outside the home alone, in shops, theatres, or standing in line. The individual is consistently fearful or anxious about these situations due to a sense of danger or fear of specific negative outcomes such as panic attacks, symptoms of panic, or other incapacitating (e.g., falling) or embarrassing physical symptoms (e.g., incontinence). The situations are actively avoided, are entered only under specific circumstances (e.g., in the presence of a companion), or else are endured with intense fear or anxiety. The symptoms are not transient, that is, they persist for an extended period of time (e.g., at least several months). The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q4 Specific Phobia: Essential (Required) Features: Marked and excessive fear or anxiety that consistently occurs when exposed to one or more specific objects or situations (e.g., proximity to certain kinds of animals, heights, closed spaces, sight of blood or injury) and that is out of proportion to the actual danger posed by the specific object or situation. The phobic object or situation is actively avoided or else endured with intense fear or anxiety. A pattern of fear, anxiety, or avoidance related to specific objects or situations is not transient, that is, it persists for an extended period of time (e.g., at least several months). The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q5  Social Anxiety Disorder: Essential (Required) Features: Marked and excessive fear or anxiety that occurs consistently in one or more social situations such as social interactions (e.g., having a conversation), being observed (e.g., while eating or drinking), or performing in front of others (e.g., giving a speech). The individual is concerned that he or she will act in a way, or show anxiety symptoms, that will be negatively evaluated by others (i.e., be humiliating, embarrassing, lead to rejection, or be offensive). Relevant social situations are consistently avoided or endured with intense fear or anxiety. The symptoms are not transient; that is, they persist for an extended period of time (e.g., at least several months). The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q6  Separation Anxiety Disorder: Essential (Required) Features: Marked and excessive fear or anxiety about separation from those individuals to whom the person is attached (i.e., has a deep affective bond with). In children and youth, attachment figures are typically parents, caregivers, or other family members, whereas in adults, they are most often a romantic partner or children. Manifestations of fear or anxiety related to separation depend on the individual’s developmental level, but may include: Persistent thoughts that harm or some other untoward event (e.g., being kidnapped) will lead to separation. Reluctance or refusal to go to school or work. Recurrent excessive distress (e.g., tantrums, social withdrawal) related to being separated from the attachment figure. Reluctance or refusal to go to sleep without being near the attachment figure. Recurrent nightmares about separation. Physical symptoms such as nausea, vomiting, stomachache, headache, on occasions that involve separation from the attachment figure, such as leaving home to go to school or work. The symptoms are not transient, that is, they persist for an extended period of time (e.g., at least several months).
The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q7  Selective Mutism: Essential (Required) Features: Consistent selectivity in speaking, such that a child demonstrates adequate language competence in specific social situations, typically at home, but consistently fails to speak in others, typically at school. The duration of the disturbance is at least one month, not limited to the first month of school. The disturbance is not due to a lack of knowledge of, or comfort with, the spoken language demanded in the social situation. Selectivity of speech is sufficiently severe so as to interfere with educational achievement or with social communication or is associated with significant impairment in other important areas of functioning.

Q8  Other Anxiety and Fear-Related Disorder: Essential (Required) Features: The clinical presentation does not satisfy the definitional requirements of any other disorder in the Anxiety and Fear-Related Disorders grouping. The symptoms are not better explained by another Mental and Behavioural Disorder (e.g., a psychotic, mood, or obsessive-compulsive and related disorder). The clinical presentation is judged to be a Mental and Behavioural Disorder that shares primary clinical features with other Anxiety and Fear-Related Disorders (e.g., physiological symptoms of excessive arousal, apprehension, and avoidance behaviour). The symptoms are not developmentally appropriate. The symptoms are in excess of what is normative for the individual’s specific cultural context. The symptoms and behaviours are not explained by another medical disorder that is not classified under Mental and Behavioural Disorders. The symptoms cause significant distress or significant impairment in personal, family, social, educational, occupational or other important areas of functioning.

Q9  Unspecified Anxiety and Fear-Related Disorder: Essential (Required) Features: (No specific diagnostic guidance provided.)
###


Vignette:
###
JV is a 22-year-old female university student. She presented to her university’s counselling center because of difficulty she is anticipating in her public speaking class.

Presenting Symptoms

In her initial session, JV explains that she is very concerned of having to give a speech in front of a class. The university requires that she take a public speaking course in order to graduate; she says she has postponed taking the class until her final year in school because she dreads the idea of it. However, she knows that she needs to find a way to complete it in order to finish her degree. She came to the counselling center in hopes of finding a way to cope with her anxiety about giving the speech. She explains that she has always been a shy person, and feels uncomfortable if she is the center of attention. 

JV’s teachers consider her to be a quiet but capable student. She says that she rarely raises her hand in class, but she will answer if she were called upon. She has had to give brief presentations in other classes, and has been able to somehow get through it, but it has always made her uncomfortable. She states that she feels very nervous before a presentation, with “butterflies in her stomach,” but calms down quickly after it is finished. However, she has never had to give a presentation to a group as large as the public speaking course with such important consequences, and feels quite anxious that she will not be able to perform satisfactorily and will therefore not be able to graduate.

JV reports that she has a group of friends, but has never been “the talkative one.” JV likes to do things with the group, but rarely speaks up because she finds it hard to break into the stream of conversation. Usually, she prefers to listen. When the clinician asks JV if she is experiencing any anxiety in the current session, JV replies that she feels fine. She elaborates that she usually does well in one-on-one situations; she only starts to feel uncomfortable when the group gets larger, “like more than 5 or 6 people.” She has a boyfriend, and she reports that the relationship is going well. She says he has a similar preference for small groups and quiet activities. When the clinician asks if she would like to change her shyness, JV states that it usually does not cause her any problems and she likes who she is as a person. She emphasizes that her only concern is making it through her required speech course. She says she is ready to tackle the problem, but wants some support and suggestions for how to do so.

Additional Background Information

JV is an only child whose parents had her relatively late in life and live in a nearby town. She describes her family life as “quiet but normal.” She denies using any drugs, but states that she occasionally drinks one or two alcoholic beverages with her boyfriend on the weekends. A recent physical examination revealed no abnormalities.
###
"""

messages = [
    {"role": "user", "content": prompt},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
# model.to(device)

try:
    generated_ids = model.generate(model_inputs,
                                max_new_tokens=1000, 
                                do_sample=True,
                                temperature=0.1,
                                pad_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("\n".join(decoded[0].split("[/INST]")[1:]).strip())

except RuntimeError as e:
    print(e)
    print("Try again with a smaller max_new_tokens value")