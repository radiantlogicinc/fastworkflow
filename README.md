# fastWorkflow
A framework for rapidly building large-scale, deterministic, interactive workflows with a fault-tolerant, conversational UX. 

- Built on the principle on "Convention over configuration", ALA Ruby on Rails
- Uses:  
  - [Semantic Router](https://github.com/aurelio-labs/semantic-router) for fast command routing
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
 
# Future Roadmap
- Training pipeline for fine-tuning routing and parameter extraction models
- Connectors to email, slack, databases etc.
- A chat assistant for generating workflow application code
- A chat assistant for generating natural language to SQL mappings
- An AI engine to guide users at every step of the workflow with command recommendations
