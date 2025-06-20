import json
from typing import List, Dict
from datasets import load_dataset
import random
import fastworkflow
import litellm

NUMOF_PERSONAS=fastworkflow.get_env_var('SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS', int)
UTTERANCES_PER_PERSONA=fastworkflow.get_env_var('SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA', int)
PERSONAS_PER_BATCH=fastworkflow.get_env_var('SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH', int)

def generate_diverse_utterances(
    seed_utterances: List[str],
    command_name,
    num_personas: int = NUMOF_PERSONAS,
    utterances_per_persona: int = UTTERANCES_PER_PERSONA,
    personas_per_batch: int = PERSONAS_PER_BATCH
) -> list[str]:
    # Initialize LiteLLM with API key
    api_key = fastworkflow.get_env_var("LITELLM_API_KEY_SYNDATA_GEN")
    model=fastworkflow.get_env_var("LLM_SYNDATA_GEN")
    litellm.api_key = api_key

    # Load PersonaHub dataset
    persona_dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")['train']

    # Randomly sample personas
    selected_indices = random.sample(range(len(persona_dataset)), num_personas)
    selected_personas = [persona_dataset[i]['persona'] for i in selected_indices]

    all_generated_responses = []
    used_personas = []

    # Extract common themes from seed utterances
    keywords = set()
    for utterance in seed_utterances:
        words = utterance.lower().split()
        keywords.update(words)

    utterance_patterns = list(keywords)
    utterance_string = "\n".join(seed_utterances)

    # Process personas in batches
    for batch_start in range(0, len(selected_personas), personas_per_batch):
        batch_end = min(batch_start + personas_per_batch, len(selected_personas))
        batch_personas = selected_personas[batch_start:batch_end]
        
        # Create combined prompt for all personas in batch
        batch_prompt = ""
        for idx, persona in enumerate(batch_personas):
            persona_name = f"Persona_{batch_start + idx + 1}"
            used_personas.append({
                "name": persona_name,
                "description": persona
            })
            batch_prompt += f"\n[{persona_name}]\n{persona}\n"

        messages = [
            {
                "role": "system",
                "content": f"""
                Generate {utterances_per_persona} unique utterances for each of the following personas.

                {batch_prompt}

                Use these seed utterances for style and intent:
                {utterance_string}

                Guidelines:
                - Generate exactly {utterances_per_persona} utterances per persona
                - Keep responses brief and natural
                - Maintain intent consistency with command: {command_name}
                - Avoid repeating the same structure
                - Use varied phrasing based on these themes: {', '.join(utterance_patterns)}
                
                Format your response exactly as follows:
                [Persona_Name]
                utterance
                utterance
                ...

                [Next_Persona_Name]
                utterance
                utterance
                ...
                """
            },
            {
                "role": "user",
                "content": f"Generate {utterances_per_persona} natural utterances for each persona listed above."
            }
        ]

        from fastworkflow.utils.logging import logger

        try:
            response = litellm.completion(
                model=model,  # Corrected model name
                messages=messages,
                max_tokens=1000,
                temperature=1.0,
                top_p=0.9,
                stop=["<|end_of_text|>"]
            )
        except litellm.exceptions.RateLimitError:
            logger.error("LiteLLM Rate limiting error!")
            return []

        # Process responses
        content = response.choices[0].message.content.strip()
        
        # Split by persona sections
        sections = content.split('[')
        for section in sections[1:]:  # Skip first empty section
            try:
                # Extract persona name and utterances
                persona_name = section.split(']')[0].strip()
                utterances = section.split(']')[1].strip().split('\n')
                
                # Clean up utterances
                utterances = [u.strip() for u in utterances if u.strip()]
                utterances = [u for u in utterances if len(u) > 3 and not u.startswith('[')]
                
                all_generated_responses.extend([
                    {"utterance": resp, "persona": persona_name} for resp in utterances
                ])
            except IndexError:
                continue

    # Structure the output
    result = {
        "seed_utterances": seed_utterances,
        "generated_utterances": all_generated_responses,
        "personas": used_personas,
        "metadata": {
            "num_personas": num_personas,
            "utterances_per_persona": utterances_per_persona,
            "personas_per_batch": personas_per_batch,
            "total_utterances": len(all_generated_responses)
        }
    }
    all_utterances = [utt["utterance"] for utt in result["generated_utterances"]]
    return [command_name] + seed_utterances + all_utterances