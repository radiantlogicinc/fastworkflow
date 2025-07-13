# Building Complex Context Hierarchies in fastWorkflow: Advanced AI Agents

<img src="logo.png" height="96" alt="fastWorkflow Logo and Title">

---

*This is the final article in our four-part series on building AI agents with fastWorkflow. [Read Part 3 here](./fastworkflow-article-3.md) if you haven't already.*

In our previous articles, we've progressed from simple function-based commands to stateful classes and inheritance hierarchies. Now, we're ready to tackle the most powerful feature of `fastWorkflow`: complex context hierarchies with bidirectional navigation.

Real-world applications rarely have flat structures. Instead, they consist of nested objects with complex relationships. A user belongs to a chatroom, a document belongs to a project, an order belongs to a customer. In this article, we'll explore how `fastWorkflow` enables your AI agent to navigate these relationships naturally, moving up and down the object hierarchy just as a human would.

## Why Context Hierarchies Matter for AI Agents

Before diving into the code, let's understand why context hierarchies are so important for building sophisticated AI agents:

1. **Natural Navigation**: Users naturally think in terms of hierarchies and want to navigate between contexts ("go back to the chatroom").
2. **Contextual Commands**: Different commands make sense in different contexts, but should be accessible from related contexts.
3. **Shared Resources**: Child objects often need access to resources or state from their parent objects.
4. **Bidirectional Relationships**: Navigation should work both ways - from parent to child and from child to parent.
5. **Complex Workflows**: Real applications involve workflows that span multiple related objects.

Let's see how `fastWorkflow` makes it easy to model these relationships and create an intuitive command structure that mirrors your application's object hierarchy.

## From Inheritance to Context Hierarchies

We'll enhance our messaging app from the previous article to support a full chatroom with multiple users. Specifically, we'll:

1. Create a `ChatRoom` class that can contain multiple `User` objects
2. Define parent-child relationships between `ChatRoom` and `User`
3. Enable navigation between contexts (from user to chatroom and back)
4. Add commands for managing users and broadcasting messages
5. Implement advanced features like database validation for parameter extraction

> [!tip]
> You can fetch messaging_app_4 code using `fastworkflow examples fetch messaging_app_4` if you want to skip writing the code 

Let's start by comparing the application code between the two versions.

### Step 1: Enhance the Application Code with Parent-Child Relationships

In our previous example, we had a simple `User` and `PremiumUser` hierarchy:

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

Now, we'll modify the `User` class to have a reference to its parent `ChatRoom` and add a new `ChatRoom` class:

```python
# messaging_app_4/application/user.py
from __future__ import annotations

# Avoid a runtime circular import: only import ChatRoom when running type checks
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover – only needed for static type checkers
    from .chatroom import ChatRoom

class User:
    """Simple user class representing the current messaging user."""

    def __init__(self, chatroom: "ChatRoom", name: str):
        self.chatroom = chatroom
        self.name = name

    def send_message(self, to: str, message: str):
        """Send a message to the target user (prints to stdout)."""
        print(f"{self.name} sends '{message}' to {to}") 

class PremiumUser(User):
    def send_priority_message(self, to, message):
        print(f"{self.name} sends PRIORITY message '{message}' to {to}")
```

```python
# messaging_app_4/application/chatroom.py
from .user import User, PremiumUser


class ChatRoom:
    def __init__(self):
        self.current_user = None
        self.users = []

    def add_user(self, user: User):
        self.users.append(user)

    def list_users(self) -> list[str]:
        return [user.name for user in self.users]

    @property
    def current_user(self) -> User:
        return self._current_user

    @current_user.setter
    def current_user(self, value: User):
        self._current_user = value

    def broadcast(self, message) -> None:
        sender_name = self._current_user.name if self._current_user else 'Anonymous'
        msg_priority = 'PRIORITY' if isinstance(self._current_user, PremiumUser) else ''

        if self.users:
            for user in self.users:
                if user.name == sender_name:
                    continue
                print(f"user {sender_name} is broadcasting {msg_priority} '{message}' to {user.name}")
        else:
            print("No users found in this chat room. Add some users first")
```

**Key differences:**
- `User` now has a reference to its parent `ChatRoom`
- We've added a new `ChatRoom` class that manages a collection of users
- `ChatRoom` has methods for adding users, listing users, and broadcasting messages
- `ChatRoom` tracks the current active user
- We've used TYPE_CHECKING to avoid circular imports, a common pattern in complex object hierarchies

### Step 2: Define the Context Hierarchy Model

In our previous example, we only had an inheritance model. Now, we need to define the parent-child relationships between contexts:

```json
# messaging_app_4/context_hierarchy_model.json
{
    "User": {
      "parent": [
        "ChatRoom"
      ]
    },
    "PremiumUser": {
      "parent": [
        "ChatRoom"
      ]
    }
}
```

**Key points:**
- This JSON file defines the parent-child relationships between command contexts
- `"User": { "parent": ["ChatRoom"] }` means that a `User` has a `ChatRoom` as its parent
- `"PremiumUser": { "parent": ["ChatRoom"] }` means that a `PremiumUser` also has a `ChatRoom` as its parent
- This tells `fastWorkflow` how to navigate up the context hierarchy

### Step 3: Implement the Context Navigation Classes

To enable context navigation, we need to add special `_Context.py` files in each command directory:

```python
# messaging_app_4/_commands/User/_User.py
from ...application.chatroom import ChatRoom
from ...application.user import User

class Context:
    @classmethod
    def get_parent(cls, command_context_object: User) -> ChatRoom:
        return command_context_object.chatroom
```

```python
# messaging_app_4/_commands/PremiumUser/_PremiumUser.py
from ...application.chatroom import ChatRoom
from ...application.user import PremiumUser

class Context:
    @classmethod
    def get_parent(cls, command_context_object: PremiumUser) -> ChatRoom:
        return command_context_object.chatroom 
```

```python
# messaging_app_4/_commands/ChatRoom/_ChatRoom.py
from ...application.chatroom import ChatRoom


class Context:
    @classmethod
    def get_parent(cls, command_context_object: ChatRoom) -> None:
        return None
```

**Key points:**
- Each context class has a special file named `_<ClassName>.py`
- These files define how to navigate from one context to another
- The `get_parent` method returns the parent object for a given context object
- For `User` and `PremiumUser`, the parent is the `chatroom` attribute
- For `ChatRoom`, there is no parent (it's the root of our hierarchy)

### Step 4: Update the Root Context Initialization

Instead of initializing a `User` as the root context, we now initialize a `ChatRoom`:

```python
# messaging_app_4/_commands/set_root_context.py
import fastworkflow

from ..application.chatroom import ChatRoom


class ResponseGenerator:
    """Create a User instance and attach it as the root command context."""

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str
    ) -> fastworkflow.CommandOutput:
        # Initialize the root command context
        workflow.root_command_context = ChatRoom()

        response = (
            f'Context: {workflow.current_command_context_displayname}\n'
            f'Command: {command}\n'
            f"Now you can call commands exposed in this context."
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
```

**Key differences:**
- We've renamed the file from `initialize_user.py` to `set_root_context.py`
- We're creating a `ChatRoom` instance as the root context
- We've simplified the command to take no parameters
- This sets up the initial context hierarchy

### Step 5: Add ChatRoom Commands

Now we need to add commands that operate on the `ChatRoom` context:

#### Add User Command

```python
# messaging_app_4/_commands/ChatRoom/add_user.py
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.chatroom import ChatRoom
from ...application.user import User, PremiumUser

class Signature:
    class Input(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            min_length=3,
            max_length=20
        )
        is_premium_user: bool = Field(
            description="Whether this is a premium user",
        )

    class Output(BaseModel):
        user_added: bool = Field(
            description="Whether the user was added",
        )

    plain_utterances = [
        "Add Fred to our list of users",
        "Update the chatroom with new premium user John Smith",
        "We have a new regular user Mary Jane",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Signature.Output:
        chatroom: ChatRoom = workflow.command_context_for_response_generation

        if input.user_name in chatroom.list_users():
            return Signature.Output(user_added=False)

        chatroom.add_user(
            PremiumUser(chatroom, input.user_name) 
            if input.is_premium_user else
            User(chatroom, input.user_name)
        )

        return Signature.Output(user_added=True)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        output = self._process_command(workflow, command_parameters)
        
        response = (
            f'Context: {workflow.current_command_context_displayname}\n'
            f'Command: {command}\n'
            f'Command parameters: {command_parameters}\n'
            f'Response: {output.model_dump_json()}'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
```

#### Set Current User Command with Database Validation

```python
# messaging_app_4/_commands/ChatRoom/set_current_user.py
from typing import Any
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.signatures import DatabaseValidator

from ...application.chatroom import ChatRoom

class Signature:
    class Input(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            json_schema_extra={'db_lookup': True}
        )

    class Output(BaseModel):
        user_found: bool = Field(
            description="Whether we found the user",
        )

    plain_utterances = [
        "Set Fred as the current user",
    ]

    @staticmethod
    def db_lookup(workflow: fastworkflow.Workflow, 
                  field_name: str, 
                  field_value: str
                  ) -> tuple[bool, str | None, list[str]]:
        if field_name == 'user_name':
            chatroom: ChatRoom = workflow.command_context_for_response_generation
            key_values = chatroom.list_users()
            matched, corrected_value, field_value_suggestions = DatabaseValidator.fuzzy_match(field_value, key_values)
            return (matched, corrected_value, field_value_suggestions)
        return (False, '', [])

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Signature.Output:
        chatroom: ChatRoom = workflow.command_context_for_response_generation

        if input.user_name not in chatroom.list_users():
            return Signature.Output(user_found=False)

        for user in chatroom.users:
            if user.name == input.user_name:
                break

        chatroom.current_user = user

        # lets change the current context to this user
        workflow.current_command_context = user

        return Signature.Output(user_found=True)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        output = self._process_command(workflow, command_parameters)
        
        response = (
            f'Context: {workflow.current_command_context_displayname}\n'
            f'Command: {command}\n'
            f'Command parameters: {command_parameters}\n'
            f'Response: {output.model_dump_json()}'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
```

**Key features:**
1. **Database Validation**: The `db_lookup` method provides real-time validation of user input against the current list of users
2. **Context Switching**: The `set_current_user` command changes the current context from `ChatRoom` to `User`
3. **Structured Output**: Commands return structured data using Pydantic models
4. **Fuzzy Matching**: The `DatabaseValidator.fuzzy_match` method helps correct typos in user names

### Step 6: Add a Startup Action

To make our workflow more user-friendly, we'll add a startup action that initializes the root context automatically:

```json
# messaging_app_4/startup_action.json
{
    "command_name": "set_root_context",
    "command": "",
    "parameters": {}
}
```

**Key points:**
- This JSON file defines an action to run when the workflow starts
- It calls the `set_root_context` command with no parameters
- This ensures the ChatRoom is initialized before the user starts interacting

## Running the Context Hierarchy Workflow

To run our context hierarchy workflow, we follow the same steps as before, but now we can use the `--startup_action` flag:

```sh
# Train the workflow
fastworkflow train ./examples/messaging_app_4 ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env

# Run the workflow with the startup action
fastworkflow run ./examples/messaging_app_4 ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env --startup_action ./examples/messaging_app_4/startup_action.json
```

Let's walk through a typical interaction with our new workflow:

```
# The workflow starts with the ChatRoom context due to our startup action
Context: ChatRoom
Command: set_root_context
Now you can call commands exposed in this context.

User > add user Alice who is a regular user
Context: ChatRoom
Command: ChatRoom/add_user
Command parameters: Input(user_name='Alice', is_premium_user=False)
Response: {"user_added":true}

User > add user Bob who is a premium user
Context: ChatRoom
Command: ChatRoom/add_user
Command parameters: Input(user_name='Bob', is_premium_user=True)
Response: {"user_added":true}

User > list users
Context: ChatRoom
Command: ChatRoom/list_users
Response: {"users":["Alice","Bob"]}

User > set current user to Alice
Context: ChatRoom
Command: ChatRoom/set_current_user
Command parameters: Input(user_name='Alice')
Response: {"user_found":true}

# Notice how the context has changed from ChatRoom to User
User > tell bob@example.com hello from Alice
Context: User
Command: User/send_message
Command parameters: Input(to='bob@example.com', message='hello from Alice')
Response: The message was printed to the screen

# Let's navigate back to the parent context
User > go back to chatroom
Context: User
Command: navigate_to_parent
Response: Successfully navigated to parent context ChatRoom

User > set current user to Bob
Context: ChatRoom
Command: ChatRoom/set_current_user
Command parameters: Input(user_name='Bob')
Response: {"user_found":true}

User > send priority message to alice@example.com with urgent update
Context: PremiumUser
Command: PremiumUser/send_priority_message
Command parameters: Input(to='alice@example.com', message='urgent update')
Response: The message was printed to the screen

User > broadcast message to all users that lunch is ready
Context: PremiumUser
Command: navigate_to_parent
Response: Successfully navigated to parent context ChatRoom

Context: ChatRoom
Command: ChatRoom/broadcast_message
Command parameters: Input(message='lunch is ready')
Response: Message has been broadcast
```

Behind the scenes, you'll see the console output:
```
Alice sends 'hello from Alice' to bob@example.com
Bob sends PRIORITY message 'urgent update' to alice@example.com
user Bob is broadcasting PRIORITY 'lunch is ready' to Alice
```

Notice how we can navigate between contexts:
1. We start in the `ChatRoom` context
2. We set the current user to Alice, which changes the context to `User`
3. We navigate back to the parent context (`ChatRoom`)
4. We set the current user to Bob, which changes the context to `PremiumUser`
5. We navigate back to the parent context again to broadcast a message

This natural navigation between contexts mirrors how users think about the application structure.

## Advanced Features in Context Hierarchies

Our example demonstrates several advanced features of `fastWorkflow` that are essential for building sophisticated AI agents:

### 1. Database Validation and Fuzzy Matching

The `set_current_user` command uses database validation to ensure the user exists:

```python
@staticmethod
def db_lookup(workflow: fastworkflow.Workflow, 
              field_name: str, 
              field_value: str
              ) -> tuple[bool, str | None, list[str]]:
    if field_name == 'user_name':
        chatroom: ChatRoom = workflow.command_context_for_response_generation
        key_values = chatroom.list_users()
        matched, corrected_value, field_value_suggestions = DatabaseValidator.fuzzy_match(field_value, key_values)
        return (matched, corrected_value, field_value_suggestions)
    return (False, '', [])
```

This enables:
- **Real-time validation** of user input against the current state
- **Fuzzy matching** to handle typos or partial matches
- **Suggestions** when the exact match isn't found

### 2. Bidirectional Context Navigation

Our workflow supports navigation in both directions:
- **Downward**: From `ChatRoom` to `User` or `PremiumUser` using `set_current_user`
- **Upward**: From `User` or `PremiumUser` back to `ChatRoom` using `navigate_to_parent`

This bidirectional navigation is defined through:
- The `context_hierarchy_model.json` file
- The `_<ClassName>.py` files with `get_parent` methods

### 3. Structured Inputs and Outputs

Commands use Pydantic models for both inputs and outputs:

```python
class Input(BaseModel):
    user_name: str = Field(
        description="Name of a person",
        examples=['John', 'Jane Doe'],
        json_schema_extra={'db_lookup': True}
    )

class Output(BaseModel):
    user_found: bool = Field(
        description="Whether we found the user",
    )
```

This provides:
- **Type safety** for all parameters
- **Validation** of inputs before command execution
- **Structured responses** that can be easily processed
- **Self-documenting** API with descriptions and examples

### 4. Context-Aware Command Routing

Commands are automatically routed to the appropriate context:
- Commands in `ChatRoom/` are available in the `ChatRoom` context
- Commands in `User/` are available in the `User` context
- Commands in `PremiumUser/` are available in the `PremiumUser` context

The framework automatically handles:
- Finding commands in the current context
- Finding commands in parent contexts when needed
- Respecting inheritance relationships

## Building Real-World Applications with Context Hierarchies

With context hierarchies, you can now build truly sophisticated AI agents that mirror the structure of your application:

- **Document Management**: Navigate between projects, folders, documents, and sections
- **E-commerce**: Move between customers, orders, products, and reviews
- **CRM Systems**: Navigate between accounts, contacts, opportunities, and activities
- **Development Tools**: Switch between projects, files, functions, and tests
- **Healthcare Systems**: Navigate between patients, visits, treatments, and prescriptions

The possibilities are endless. By modeling your application's natural structure as a context hierarchy, you create an AI agent that users can interact with intuitively, using the same mental model they already have for your application.

## What's Next? Building Your Own Workflows

We've now completed our journey from simple function-based commands to sophisticated context hierarchies. You have all the tools you need to start building your own AI-powered workflows with `fastWorkflow`.

Remember the key principles we've covered:
1. Start simple with function-based commands
2. Add state with classes and methods
3. Use inheritance for specialized behavior
4. Build context hierarchies for complex applications
5. Enable bidirectional navigation between contexts

The `fastWorkflow` framework handles all the complex parts - intent detection, parameter extraction, command routing, and context navigation - so you can focus on building your application logic.

---

## Conclusion: Join the fastWorkflow Community

We hope this series has shown you the power and flexibility of `fastWorkflow` for building AI agents that understand your application's structure and can interact with it naturally.

`fastWorkflow` is an open-source project released under the Apache 2.0 license, which means you can use it freely in your own projects, both commercial and non-commercial.

We invite you to:
- **Star the repository** at [github.com/fastworkflow/fastworkflow](https://github.com/fastworkflow/fastworkflow)
- **Try the examples** included in the repository
- **Build your own workflows** using the techniques we've covered
- **Contribute** to the project with bug reports, feature requests, or pull requests
- **Join the community** to share your experiences and learn from others

Whether you're building a simple chatbot or a complex enterprise application, `fastWorkflow` provides the foundation you need to create AI agents that truly understand your application's structure and can interact with it naturally.

Happy building!

---

## Key Takeaways

- Context hierarchies enable natural navigation between related objects in your application
- Bidirectional navigation (parent-to-child and child-to-parent) creates intuitive workflows
- Database validation ensures commands operate on valid data
- Structured inputs and outputs provide type safety and clear documentation
- Context-aware command routing automatically finds the right command in the right context
- The combination of inheritance and context hierarchies enables modeling complex real-world applications 