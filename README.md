# fastWorkflow
A framework for rapidly building large-scale, deterministic, interactive workflows with a fault-tolerant, conversational UX and AI-powered recommendations.

- Built on the principle on "Convention over configuration", ALA Ruby on Rails
- Uses:
  - A custom-built intent detection pipeline for fault-tolerant, self-correcting command routing
  - [Pydantic](https://docs.pydantic.dev/) and [DSPy](https://github.com/stanfordnlp/dspy) for parameter extraction and response generation

# Concepts
- Workflows are defined as a directory hierarchy of workitem types
  - Workitems can be ordered
  - Min/max constraints can be defined for the number of child workitems (one, unlimited, min/max)
  - Workflows can delegate to other workflows
- Commands are exposed for each workitem type
  - Commands may be specific to one workitem type or inheritable by child workitem types (base commands)
- Users are guided through the workflow but have complete control over navigation
  - Workflow navigation and command execution are exposed via a chat interface
  - Special constrained workflows are used to handle routing and parameter extraction errors
- AI-powered recommendations after every command interaction
  - Recommendations are generated AFTER a command has been processed. The user has complete control over the workflow and discretion over whether to follow a recommendation or take a different action.

# Getting started
- Clone the repo
  - Use WSL if you are on Windows
- Create an env folder with a .env file inside and add the following entries
  - DSPY_LM_MODEL: the model to use for the DSPy API
  - SPEEDDICT_FOLDERNAME: the folder where the workflow definitions are stored
 - export the OPENAI_API_KEY as an environment variable
   - Note: If you use a different model, specify the model path in the DSPY_LM_MODEL environment variable (The app uses Litellm as the LLM wrapper)
- Train then run the sample workflow
  - Hint: review the .vscode/launch.json file for training/running the sample workflow
 
# Future Roadmap
- Training pipeline for prompt-tuning/fine-tuning the models - routing, parameter extraction, response generation, and recommendations
- Connectors to email, slack, databases etc.
- A chat assistant for generating workflow application code
- A chat assistant for generating natural language to SQL mappings
- An AI engine to guide users at every step of the workflow with command recommendations
