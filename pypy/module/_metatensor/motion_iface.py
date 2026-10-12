"""The vocabulary a change rule may use, and nothing else.

Two abstract classes naming membership, policy and uncommitted contributions,
but no kernel, fused region, guard or resume data. benchmark/eval/audit_rules.py
checks the rule file against that.
"""

from pypy.interpreter.baseobjspace import W_Root

RETAIN = 'retain'
REVOKE = 'revoke'


class Membership(W_Root):
    """Who takes part, and the declared action for each leave reason."""

    def drop_member(self, p):
        """p no longer takes part; anything that assumed it did is void."""
        raise NotImplementedError

    def action(self, reason):
        """The declared action for this leave reason: RETAIN or REVOKE."""
        raise NotImplementedError


class Uncommitted(W_Root):
    """Contributions received but not committed yet."""

    def exclude(self, p):
        """Leave p's contributions out of what will be committed."""
        raise NotImplementedError
