import json

class PromptBuilder:
    """
    Wrapper class for constructing diagnostic prompts using CoT instructions.
    """
    def __init__(self, df_vignettes=None, prompts_path=None):
        self.prompt_anxiety = self.load_txt(prompts_path.joinpath("cot_text_vicky_ddx/cot_anxiety.txt"))
        self.prompt_mood = self.load_txt(prompts_path.joinpath("cot_text_vicky_ddx/cot_mood.txt"))
        self.prompt_stress = self.load_txt(prompts_path.joinpath("cot_text_vicky_ddx/cot_stress.txt"))

        self.df_vignettes = df_vignettes

    def load_txt(self, path):
        with open(path, 'r') as file:
            return file.read()

    def prepare_cot_prompts(self, category):
        """
        Prepare prompts with chain-of-thought instructions for each vignette.
        """
        cot_text = {
            "Anxiety": self.prompt_anxiety,
            "Mood": self.prompt_mood,
            "Stress": self.prompt_stress
        }.get(category, "No specific diagnostic steps for this category.")

        prompts = []
        for _, row in self.df_vignettes.iterrows():
            vignette_id = row['Vignette ID']
            vignette = (f"Referral: {row['Referral']}\n"
                        f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                        f"Additional Background Information: {row['Additional Background Information']}")

            user_prompt = f"""[ROLE]
You are a psychiatrist conducting a diagnostic assessment. Read the patient case and follow the instructions to provide a diagnosis. Note: Not all patients will have a diagnosis.

[PATIENT CASE]
{vignette}

[INSTRUCTIONS]
{cot_text}

[REQUIRED FORMAT]
Provide EXACTLY 3 possible diagnoses in a descending order of likelihood:

1. Most Likely Diagnosis: [diagnosis]
   Reasoning: [2-3 sentences]

2. Second Most Likely: [diagnosis]
   Reasoning: [1-2 sentences]

3. Third Most Likely: [diagnosis]
   Reasoning: [1-2 sentences]"""

            prompts.append({
                "id": f"{category}_{vignette_id}",
                "prompt": [
                    {"role": "user", "content": user_prompt}
                ]
            })

        return prompts
