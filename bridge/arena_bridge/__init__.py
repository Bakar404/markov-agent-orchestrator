"""Drive arena episodes with Microsoft Agent Framework instead of a human in the loop.

The arena's contract is unchanged: ``live/open`` asks the policy who acts, and ``live/report``
folds back what those agents produced. What changes is who fills the middle. A person driving
that loop can quietly fabricate a report; a framework cannot, because every number it sends is
read off the response it just received.
"""

__all__ = ["arena", "executor", "driver", "judge"]
