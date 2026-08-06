from typing import Any

import fastworkflow

from ...application.user import PremiumUser


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, command_context_object: PremiumUser) -> None:
        return None

    @classmethod
    def get_state(cls, command_context_object: PremiumUser) -> dict[str, Any]:
        """Snapshot the premium user so a session holding it can be evicted.

        initialize_user picks the class from its is_premium_user parameter, so
        a session can be anchored on either User or PremiumUser and each needs
        its own hook: the framework rebuilds through the Context class named
        after the recorded class, and refuses a from_state that hands back a
        different one.
        """
        return {'name': command_context_object.name}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> PremiumUser:
        """Rebuild the premium user from a snapshot written at *state_version*.

        Only version 1 has ever shipped, so there is nothing to migrate yet.
        Once a second version exists, translate the older shape here rather
        than refusing it — this hook is the only code that understands it.
        """
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)

        return PremiumUser(state['name'])

    # No get_locator/find_by_locator: the current context is the only context
    # this workflow sets, so every slot is the anchor itself.
