# Building Stateful AI Agents with fastWorkflow: From Functions to Classes

<img src="logo.png" height="96" alt="fastWorkflow Logo and Title">

---

*This is the second article in our four-part series on building AI agents with fastWorkflow. [Read Part 1 here](https://medium.com/p/your-first-article-link-here) if you haven't already.*

In our previous article, we built a simple messaging workflow using a standalone function. While this approach works well for basic commands, real-world applications are rarely that simple. Most applications maintain state, have complex object models, and require navigating between different contexts.

In this article, we'll explore how `fastWorkflow` seamlessly integrates with object-oriented programming to create stateful AI agents. We'll transform our simple messaging example into a class-based application that maintains user state.

## Why Class Methods Matter for AI Agents

Before diving into the code, let's understand why moving from functions to class methods is so important for building sophisticated AI agents:

1. **State Management**: Classes encapsulate state, allowing your agent to remember information between commands.
2. **Context Awareness**: Different commands make sense in different contexts - classes provide natural boundaries.
3. **Finite State Machines**: Complex workflows often behave like state machines, where actions change the available options.
4. **Object Hierarchies**: Real applications have complex object models with parent-child relationships.
5. **Code Organization**: Classes provide a natural way to group related functionality.

Let's see how `fastWorkflow` makes it easy to work with classes and maintain state across interactions.

## From Functions to Classes: A Step-by-Step Transformation

We'll transform our simple messaging app from the previous article into a class-based version. Here's what we're going to build:

1. A `User` class that maintains the sender's name
2. A startup command that initializes the user context
3. A send_message command that operates on the user instance

Let's start by comparing the application code between the two versions.

### Step 1: Transform the Application Code

In our first example, we had a simple function:

```python
# messaging_app_1/application/send_message.py
def send_message(to: str, message: str) -> str:
    print(f"Sending '{message}' to {to}")
```

Now, we'll transform this into a class with state:

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

**Key differences:**
- We've created a `User` class instead of a standalone function
- The class maintains state (the user's name)
- The `send_message` method includes the sender's name in the output
- We need to initialize the class with a name before we can use it

### Step 2: Update the Command Structure

In our first example, we had a single command file at the root level:

```
messaging_app_1/
├── _commands/
│   └── send_message.py
```

Now, we need to organize our commands by context:

```
messaging_app_2/
├── _commands/
│   ├── startup.py
│   └── User/
│       └── send_message.py
```

**Key differences:**
- We've added a `startup.py` command to initialize the user context
- We've moved `send_message.py` into a `User/` directory to indicate it operates on a User instance
- This directory structure mirrors our object model, making it clear which commands operate on which objects

### Step 3: Create a Startup Command

The startup command is entirely new. It's responsible for initializing our application state:

```python
# messaging_app_2/_commands/startup.py
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.user import User


class Signature:
    """Initialize the workflow with a root User context."""

    class Input(BaseModel):
        name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            default='DefaultUser'
        )

    plain_utterances = [
        "start messaging session",
        "initialize user context",
        "login as user Billy",
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
        # Initialize the root context
        workflow.root_command_context = User(command_parameters.name)

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

**Key points:**
- The startup command takes a `name` parameter to initialize the user
- It creates a new `User` instance with that name
- It sets this instance as the `root_command_context` for the workflow
- This establishes the initial state for our application
- The utterances like "login as user Billy" will be recognized as intent to run this command

### Step 4: Update the Send Message Command

Now, let's modify our `send_message` command to work with the User class:

```python
# messaging_app_2/_commands/User/send_message.py
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.user import User

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
        user: User = workflow.command_context_for_response_generation
        user.send_message(to=input.to, message=input.message)

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

**Key differences from the function-based version:**
- The import path has changed to reference the User class
- The most important change is in the `_process_command` method:
  ```python
  user: User = workflow.command_context_for_response_generation
  user.send_message(to=input.to, message=input.message)
  ```
- Instead of calling a global function, we're accessing the current command context (our User instance)
- We're calling the `send_message` method on that instance
- This allows the method to access the user's state (name)

## Running the Class-Based Workflow

To run our new class-based workflow, we follow the same steps as before:

```sh
# Train the workflow
fastworkflow train ./examples/messaging_app_2 ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env

# Run the workflow
fastworkflow run ./examples/messaging_app_2 ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env
```

When you start the workflow, you'll need to initialize the user context first:

```
User > login as Alice
Context: None
Command: startup
Command parameters: Input(name='Alice')
Root context set to User('Alice').Now you can call commands exposed in this context.

User > tell bob@example.com hello from the stateful workflow
Context: User
Command: User/send_message
Command parameters: Input(to='bob@example.com', message='hello from the stateful workflow')
Response: The message was printed to the screen
```

Behind the scenes, you'll see the console output:
```
Alice sends 'hello from the stateful workflow' to bob@example.com
```

Notice how the message now includes the sender's name, which is maintained as state in our User class.

## Understanding Context Navigation

One of the most powerful features of `fastWorkflow` is its ability to navigate object hierarchies. The framework automatically:

1. **Recognizes command contexts**: Commands in the `User/` directory are automatically associated with User instances
2. **Maintains the current context**: The `workflow.command_context_for_response_generation` property gives you access to the current context
3. **Routes commands to the right objects**: When you say "send message", fastWorkflow knows to look for that command in the current context

This context-awareness is crucial for building complex applications where different commands make sense in different contexts.

## The Power of Stateful Workflows

With this class-based approach, you can now build much more sophisticated applications:

- **Multiple Users**: Create a system with multiple user instances
- **Nested Objects**: Build complex object hierarchies (e.g., User → Conversation → Message)
- **State Transitions**: Change the available commands based on the current state
- **Persistent State**: Save and restore object state between sessions

All of this is possible because `fastWorkflow` understands your application's object model and can map natural language commands to the appropriate methods on the right objects.

## What's Next?

We've taken a significant step forward by moving from standalone functions to stateful class methods. In the next article, we'll explore how to build even more sophisticated applications using inheritance and context hierarchies.

**Next up: [Part 3: Building Context Hierarchies with Inheritance in fastWorkflow](https://medium.com/p/your-next-article-link-here)**

---

## Key Takeaways

- Classes provide a natural way to maintain state in your AI-powered workflows
- `fastWorkflow` seamlessly integrates with object-oriented programming
- The directory structure of your commands mirrors your object model
- The startup command initializes your application state
- Context-aware command routing ensures commands are executed on the right objects
- This approach enables building complex, stateful applications with natural language interfaces 