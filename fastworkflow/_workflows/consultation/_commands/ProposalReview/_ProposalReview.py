# Context callback for ProposalReview. No command enters it directly.
# fastWorkflow requires this class to be named exactly `Context`.


class Context:
    occupiable = False


class ProposalReview:
    """Occupiable context marker; ``current_command_context_name`` is ``ProposalReview``."""
