# Leveraging Class Inheritance in fastWorkflow: Building Advanced AI Agents

<img src="logo.png" height="96" alt="fastWorkflow Logo and Title">

---

*This is the third article in our four-part series on building AI agents with fastWorkflow. [Read Part 2 here](https://medium.com/p/your-second-article-link-here) if you haven't already.*

In our previous article, we transformed our simple function-based messaging application into a class-based one, introducing state management and context awareness. Now, it's time to take our application to the next level by leveraging one of the most powerful features of object-oriented programming: inheritance.

Inheritance is the foundation of code reuse and polymorphism. In this article, we'll explore how `fastWorkflow` seamlessly supports class hierarchies, allowing you to build sophisticated AI agents that understand the relationships between different types of objects in your application.

## Why Inheritance Matters for AI Agents

Before diving into the code, let's understand why inheritance is so valuable when building AI-powered applications:

1. **Feature Differentiation**: Different user types or entities often share core functionality but have specialized features.
2. **Code Reuse**: Inheritance allows you to define common behavior once and extend it for specific cases.
3. **Polymorphism**: The ability to treat derived classes through their base class interface simplifies command routing.
4. **Privilege Escalation**: Some commands should only be available to certain user types or in specific contexts.
5. **Natural Language Understanding**: Users naturally understand hierarchical relationships ("as a premium user, I can...").

Let's see how `fastWorkflow` makes it easy to model these relationships and create an intuitive command structure that mirrors your application's class hierarchy.

## From Basic Classes to Inheritance Hierarchies

We'll enhance our messaging app from the previous article to support different types of users with different capabilities. Specifically, we'll:

1. Create a `PremiumUser` class that inherits from `User`
2. Add a premium-only feature (`send_priority_message`)
3. Configure the inheritance model so `fastWorkflow` understands the relationship
4. Update the initialization process to support both user types

Let's start by comparing the application code between the two versions.

### Step 1: Enhance the Application Code with Inheritance

In our previous example, we had a simple `User` class:

```python
# messaging_app_2/application/user.py
class User:
    """Simple user class representing the current messaging user."""

    def __init__(self, name: str):
        self.name = name

    def send_message(self, to: str, message: str):
        """Send a message to the target user (prints to stdout)."""
        print(f"{self.name} sends '{message}' to {to}") 
```

Now, we'll add a `PremiumUser` class that inherits from `User`:

```python
# messaging_app_3/application/user.py
class User:
    """Simple user class representing the current messaging user."""

    def __init__(self, name: str):
        self.name = name

    def send_message(self, to: str, message: str):
        """Send a message to the target user (prints to stdout)."""
        print(f"{self.name} sends '{message}' to {to}") 

class PremiumUser(User):
    def send_priority_message(self, to, message):
        print(f"{self.name} sends PRIORITY message '{message}' to {to}")
```

**Key differences:**
- We've added a `PremiumUser` class that inherits from `User`
- `PremiumUser` inherits all methods from `User` (including `send_message`)
- `PremiumUser` adds a new method `send_priority_message` that's only available to premium users
- The new method includes "PRIORITY" in the output to differentiate it

### Step 2: Update the Command Structure for Inheritance

In our previous example, we had a simple command structure:

```
messaging_app_2/
├── _commands/
│   ├── startup.py
│   └── User/
│       └── send_message.py
```

Now, we need to organize our commands to reflect the inheritance hierarchy:

```
messaging_app_3/
├── _commands/
│   ├── initialize_user.py
│   ├── User/
│   │   └── send_message.py
│   ├── PremiumUser/
│   │   └── send_priority_message.py
│   └── context_inheritance_model.json
```

**Key differences:**
- We've renamed `startup.py` to `initialize_user.py` for clarity
- We've added a `PremiumUser/` directory for premium-only commands
- We've added a `send_priority_message.py` command in the `PremiumUser/` directory
- We've created a `context_inheritance_model.json` file to define the inheritance relationship

### Step 3: Define the Inheritance Model

The `context_inheritance_model.json` file is entirely new and critical for inheritance support:

```json
{
    "User": {
        "base": [
          "*"
        ]
      },  
    "PremiumUser": {
        "base": [
          "User"
        ]
      }  
}
```

**Key points:**
- This JSON file defines the inheritance relationships between command contexts
- `"User": { "base": ["*"] }` means that `User` is a base context that can be used anywhere
- `"PremiumUser": { "base": ["User"] }` means that `PremiumUser` inherits from `User`
- This tells `fastWorkflow` that a `PremiumUser` can access all `User` commands, but not vice versa

### Step 4: Update the Initialization Command

We need to modify our initialization command to support creating either a regular `User` or a `PremiumUser`:

```python
# messaging_app_3/_commands/initialize_user.py
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.user import User, PremiumUser


class Signature:
    """Initialize the workflow with a root User context."""

    class Input(BaseModel):
        name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            default='DefaultUser'
        )
        is_premium_user: bool = Field(
            description="Whether the person is a regular user or a premium user"
        )

    plain_utterances = [
        "start messaging session as Jack",
        "initialize user context",
        "login as user Billy who is a premium user",
        "startup with Mary, a regular user",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """Generate training utterances for LLM-based intent matching."""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Create a User instance and attach it as the root command context."""

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str,
        command_parameters: Signature.Input,
    ) -> fastworkflow.CommandOutput:
        # Initialize the current command context
        workflow.current_command_context = (
            PremiumUser(command_parameters.name) if command_parameters.is_premium_user else
            User(command_parameters.name)
        )

        response = (
            f'Context: {workflow.current_command_context_displayname}\n'
            f'Command: {command}\n'
            f'Command parameters: {command_parameters}\n'
            f"Root context set to User('{command_parameters.name}')."
            f"Now you can call commands exposed in this context."
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
```

**Key differences from the previous version:**
- We've added a new parameter `is_premium_user` to determine which class to instantiate
- We've updated the utterances to include examples like "login as user Billy who is a premium user"
- The most important change is in the `__call__` method:
  ```python
  workflow.current_command_context = (
      PremiumUser(command_parameters.name) if command_parameters.is_premium_user else
      User(command_parameters.name)
  )
  ```
- This conditional creates either a `PremiumUser` or a regular `User` based on the input

### Step 5: Add the Premium-Only Command

Finally, we need to add the premium-only command for sending priority messages:

```python
# messaging_app_3/_commands/PremiumUser/send_priority_message.py
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.user import PremiumUser

# the signature class defines our intent
class Signature:
    class Input(BaseModel):
        to: str = Field(
            description="Who are you sending the message to",
            examples=['jsmith@abc.com', 'jane.doe@xyz.edu'],
            pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        )
        message: str = Field(
            description="The message you want to send",
            examples=['Hello, how are you?', 'Hi, reaching out to discuss fastWorkflow'],
            min_length=3,
            max_length=500
        )

    plain_utterances = [
        "Tell john@fastworkflow.ai that the build tool needs improvement",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> None:
        user: PremiumUser = workflow.command_context_for_response_generation
        user.send_priority_message(to=input.to, message=input.message)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        self._process_command(workflow, command_parameters)
        
        response = (
            f'Context: {workflow.current_command_context_displayname}\n'
            f'Command: {command}\n'
            f'Command parameters: {command_parameters}\n'
            f'Response: The message was printed to the screen'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
```

**Key points:**
- This file is similar to `send_message.py` but calls `send_priority_message` instead
- It imports `PremiumUser` specifically, not just `User`
- It's placed in the `PremiumUser/` directory, making it available only to premium users
- The signature and utterances are similar, but the implementation calls the premium-only method

## Running the Inheritance-Aware Workflow

To run our inheritance-aware workflow, we follow the same steps as before:

```sh
# Train the workflow
fastworkflow train ./examples/messaging_app_3 ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env

# Run the workflow
fastworkflow run ./examples/messaging_app_3 ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env
```

When you start the workflow, you'll need to specify whether you're a regular or premium user:

```
User > login as Alice who is a regular user
Context: None
Command: initialize_user
Command parameters: Input(name='Alice', is_premium_user=False)
Root context set to User('Alice').Now you can call commands exposed in this context.

User > tell bob@example.com hello from the regular user
Context: User
Command: User/send_message
Command parameters: Input(to='bob@example.com', message='hello from the regular user')
Response: The message was printed to the screen

User > send priority message to bob@example.com
Error: Command 'send priority message' not found in context 'User'
```

Notice how the regular user can't access the premium-only command. Now let's try with a premium user:

```
User > login as Charlie who is a premium user
Context: None
Command: initialize_user
Command parameters: Input(name='Charlie', is_premium_user=True)
Root context set to User('Charlie').Now you can call commands exposed in this context.

User > tell bob@example.com hello from the premium user
Context: PremiumUser
Command: User/send_message
Command parameters: Input(to='bob@example.com', message='hello from the premium user')
Response: The message was printed to the screen

User > send priority message to bob@example.com with urgent update
Context: PremiumUser
Command: PremiumUser/send_priority_message
Command parameters: Input(to='bob@example.com', message='urgent update')
Response: The message was printed to the screen
```

Behind the scenes, you'll see the console output:
```
Charlie sends 'hello from the premium user' to bob@example.com
Charlie sends PRIORITY message 'urgent update' to bob@example.com
```

Notice how the premium user can access both the regular `send_message` command (inherited from `User`) and the premium-only `send_priority_message` command.

## Understanding the Inheritance Model

The magic that makes this all work is the `context_inheritance_model.json` file. This file tells `fastWorkflow` how your command contexts relate to each other. Let's break it down:

```json
{
    "User": {
        "base": [
          "*"
        ]
      },  
    "PremiumUser": {
        "base": [
          "User"
        ]
      }  
}
```

- `"User": { "base": ["*"] }` defines `User` as a base context that can be used anywhere
- `"PremiumUser": { "base": ["User"] }` defines `PremiumUser` as inheriting from `User`

This means:
1. Commands in the `User/` directory are available to both `User` and `PremiumUser` instances
2. Commands in the `PremiumUser/` directory are only available to `PremiumUser` instances
3. When a user says "send message", `fastWorkflow` knows to look in the `User/` directory for both user types
4. When a user says "send priority message", `fastWorkflow` only finds it for `PremiumUser` instances

This inheritance model mirrors the class inheritance in your application code, creating a consistent and intuitive command structure.

## The Power of Inheritance in AI Agents

With inheritance support, you can now build much more sophisticated applications:

- **User Roles and Permissions**: Create different user types with different capabilities
- **Feature Tiers**: Implement freemium models with premium-only features
- **Specialized Behavior**: Override methods in subclasses to provide specialized behavior
- **Progressive Disclosure**: Show only relevant commands based on the user's context
- **Natural Command Organization**: Group commands logically by the objects they operate on

All of this is possible because `fastWorkflow` understands your application's class hierarchy and can map natural language commands to the appropriate methods on the right objects, respecting inheritance relationships.

## What's Next?

We've now seen how to leverage inheritance to create more sophisticated AI agents. In the final article of this series, we'll explore how to build complex context hierarchies with nested objects, allowing your AI agent to navigate between different parts of your application.

**Next up: [Part 4: Building Complex Context Hierarchies in fastWorkflow](https://medium.com/p/your-next-article-link-here)**

---

## Key Takeaways

- Inheritance allows you to model different types of users or entities with shared and specialized functionality
- `fastWorkflow` understands class hierarchies through the `context_inheritance_model.json` file
- Commands are organized by context, with inherited commands available to derived classes
- The directory structure of your commands mirrors your object inheritance model
- This approach enables building sophisticated applications with role-based permissions and specialized features
- Users can seamlessly access both inherited and specialized commands based on their context 