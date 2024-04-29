from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

access_token = "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"
model = "meta-llama/Llama-2-7b-chat-hf" 

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=access_token, device_map = 'auto', )
print(model)
device_map = {
    "layer.0": 0,
    "layer.1": 1
}

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    tokenizer=tokenizer,
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        repetition_penalty=1.1,
        temperature=0.1,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

prompt = """
<s> 
[INST] <<SYS>> Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. <</SYS>>\n\n

Based on the provided guidelines (Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8), choose the disorder that corresponds to the vignette below. Choose from the following disorders: Generalized Anxiety Disorder, Panic Disorder, Agoraphobia, Specific Phobia, Social Anxiety Disorder, Separation Anxiety Disorder, Selective Mutism, Other Anxiety and Fear-Related Disorder, and none of the above. Explain your choice by referring to the symptoms of the disorder and how they relate to the vignette. Also explain why they do not relate to the other disorders. You must explain in detail why they do not relate to other disorders.

Q1 Generalized Anxiety Disorder

Essential (Required) Features: Marked symptoms of anxiety accompanied by either:
general apprehensiveness that is not restricted to any particular environmental circumstance (i.e., “free-floating anxiety”); or
worry (apprehensive expectation) about untoward events occurring in several different aspects of everyday life (e.g., work, finances, health, family). Anxiety and general apprehensiveness or worry are accompanied by additional symptoms, such as:
Muscle tension or motor restlessness. Sympathetic autonomic overactivity as evidenced by frequent gastrointestinal symptoms such as nausea and/or abdominal distress, heart palpitations, sweating, trembling, shaking, and/or dry mouth. Subjective experience of nervousness, restlessness, or being “on edge”. Difficulties maintaining concentration. Irritability. Sleep disturbances (difficulty falling or staying asleep, or restless, unsatisfying sleep). The symptoms are not transient and persist for at least several months, for more days than not. The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q2  Panic Disorder

Essential (Required) Features: Recurrent unexpected panic attacks that are not restricted to particular stimuli or situations. Panic attacks are discrete episodes of intense fear or apprehension also characterized by the rapid and concurrent onset of several characteristic symptoms. These symptoms may include, but are not limited to, the following:
Palpitations or increased heart rate. Sweating. Trembling. Sensations of shortness of breath. Feelings of choking. Chest pain. Nausea or abdominal distress. Feelings of dizziness or lightheadedness. Chills or hot flushes. Tingling or lack of sensation in extremities (i.e., paresthesias). Depersonalization or derealization. Fear of losing control or going mad. Fear of imminent death
Panic attacks are followed by persistent concern or worry (e.g., for several weeks) about their recurrence or their perceived negative significance (e.g., that the physiological symptoms may be those of a myocardial infarction), or behaviours intended to avoid their recurrence (e.g., only leaving the home with a trusted companion). The symptoms are sufficiently severe to result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning. Panic attacks can occur in other Anxiety and Fear-Related Disorders as well as other Mental and Behavioural Disorders and therefore the presence of panic attacks is not in itself sufficient to assign a diagnosis of Panic Disorder.
 
Q3  Agoraphobia

Essential (Required) Features: Marked and excessive fear or anxiety that occurs in, or in anticipation of, multiple situations where escape might be difficult or help might not be available, such as using public transportation, being in crowds, being outside the home alone, in shops, theatres, or standing in line. The individual is consistently fearful or anxious about these situations due to a sense of danger or fear of specific negative outcomes such as panic attacks, symptoms of panic, or other incapacitating (e.g., falling) or embarrassing physical symptoms (e.g., incontinence). The situations are actively avoided, are entered only under specific circumstances (e.g., in the presence of a companion), or else are endured with intense fear or anxiety. The symptoms are not transient, that is, they persist for an extended period of time (e.g., at least several months). The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Q4 Specific Phobia
 
Essential (Required) Features: Marked and excessive fear or anxiety that consistently occurs when exposed to one or more specific objects or situations (e.g., proximity to certain kinds of animals, heights, closed spaces, sight of blood or injury) and that is out of proportion to the actual danger posed by the specific object or situation. The phobic object or situation is actively avoided or else endured with intense fear or anxiety. A pattern of fear, anxiety, or avoidance related to specific objects or situations is not transient, that is, it persists for an extended period of time (e.g., at least several months). The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

[vignette]:
PK is a 36-year-old woman who lives with her husband and two young daughters, ages 7 and 10, and works in a government office. Last week, she went to see a gastroenterologist with complaints of chronic stomach pain and was diagnosed with an ulcer. In an attempt to identify potential causes of the ulcer, the gastroenterologist asked PK about her stress level. PK mentioned that she had been feeling extremely nervous and worried during the past couple of years, even though nothing unusual or bad has happened. The gastroenterologist suggested that PK speak to a mental health professional to help her manage her anxiety. Presenting symptoms: During the initial mental health evaluation, PK admits that she worries almost all the time, and now she is concerned that her worry is damaging her health. PK says that she has always been “a worrier” and that she thinks that this quality has helped her to be attentive to her family and conscientious in her job.  However, she says that during the past couple of years it has gotten “out of control” and she worries to the point that she feels overwhelmed and exhausted.  When asked what she worries about, she says, “It could be anything—whether my kids might get sick, whether I locked the front door, whether I can finish a report for work on time—whatever I am thinking about can be something to worry about.” She provides the example of when her daughters have a minor cold, she worries that it will turn into something serious to the extent that she has to take them to the doctor, even though she knows that they are in good health overall and that all children get colds. As another example, PK says that her husband is very social and often invites his co-workers over for dinner, which PK says “really stresses her out” because she worries that the meal will turn out badly and that this will somehow damage her husband’s career. On most days, she finds it very difficult to sleep, as she lies in bed thinking about all the things she has to do the next day, and worries that she will not be able to get everything done.  She feels very tense, and often has pain in her shoulders and upper back that she recognizes comes from the tension. What is the most likely diagnosis for PK and why? Your options are the following: Generalized Anxiety Disorder,Panic Disorder, Agoraphobia, Specific Phobia, Social Anxiety Disorder, Separation Anxiety Disorder, Selective Mutism, Other Anxiety and Fear-Related Disorder, Unspecified Anxiety and Fear-Related Disorder. 

\n[/INST]\n\n
[RESPONSE]:
"""
get_llama_response(prompt)