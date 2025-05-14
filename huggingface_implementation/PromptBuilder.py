# PromptBuilder.py
import json

class PromptBuilder:
    """
    Wrapper class for constructing diagnostic prompts using CoT instructions.
    """
    def __init__(self, df_vignettes=None, prompts_path=None, prompt_id='prompt_ddx', language='en'):
        self.prompt_dict = self.load_json(prompts_path.joinpath("prompts.json"))[language][prompt_id]
        # self.prompt_anxiety = self.load_txt(prompts_path.joinpath("cot_anxiety.txt"))
        # self.prompt_mood = self.load_txt(prompts_path.joinpath("cot_mood.txt"))
        # self.prompt_stress = self.load_txt(prompts_path.joinpath("cot_stress.txt"))
        self.df_vignettes = df_vignettes

    def load_txt(self, path):
        with open(path, 'r') as file:
            return file.read()
    
    def load_json(self, path):
        """
        Loads a JSON file from the given path.
        """
        with open(path, 'r') as file:
            return json.load(file)
        

    def prepare_cot_prompts(self, category):
        """
        Prepare prompts with chain-of-thought instructions for each vignette.
        """
        cot_text = {
            "Anxiety": self.prompt_dict['cot_anxiety'],
            "Mood": self.prompt_dict['cot_mood'],
            "Stress": self.prompt_dict['cot_stress']
        }.get(category, "No specific diagnostic steps for this category.")

        df_cat = self.df_vignettes[self.df_vignettes["Category"] == category]

        prompts = []
        for _, row in df_cat.iterrows():
            vignette_id = row['Vignette ID']
            vignette = self.prompt_dict['vignette_prompt'].replace('{referral}', row['Referral']).replace('{presenting_symptoms}', row['Presenting Symptoms']).replace('{add_background_info}', str(row['Additional Background Information']))

            if f"instruction_{category.lower()}" in self.prompt_dict.keys():
                instruction_prompt = self.prompt_dict[f"instruction_{category.lower()}"]
            else:
                instruction_prompt = self.prompt_dict['instruction']
            user_prompt = instruction_prompt.replace('{vignette}', vignette).replace('{cot_text}',cot_text)

            prompts.append({
                "id": f"{category}_{vignette_id}",
                "prompt": [
                    {"role": "user", "content": user_prompt}
                ]
            })

        return prompts
