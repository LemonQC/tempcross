"""Finite-evidence specialization of the TOCAT draft, not a trained policy.

The independent path-search comparator has exactly the same witnessed edges.
An absent edge is unknown, never an impossibility label. No continuous TAMP
solver or counterfactual robot trajectory is implemented here.
"""
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Edge:
    option: str
    destination: str
    duration: int
    witness_id: str


@dataclass
class EvidenceGraph:
    edges: dict
    goals: set

    def validate(self):
        for source,edges in self.edges.items():
            for edge in edges:
                if edge.duration<1 or not edge.witness_id:
                    raise ValueError('Every edge needs positive duration and a factual witness')


def continuation_contract(graph,root,budget):
    """Evidence-backed predecessor recursion for the proposed contract.

    Returns supported first options, not a probability or a physical success
    prediction. Unknown transitions do not enter the lower support set.
    """
    @lru_cache(None)
    def reachable(node,b):
        if node in graph.goals:return True
        return any(e.duration<=b and reachable(e.destination,b-e.duration)
                   for e in graph.edges.get(node,()))
    if root in graph.goals:return frozenset({'ALREADY_GOAL'})
    return frozenset(e.option for e in graph.edges.get(root,())
                     if e.duration<=budget and reachable(e.destination,budget-e.duration))


def witnessed_task_planner(graph,root,budget):
    """Independent forward exhaustive path search, same information and budget."""
    if root in graph.goals:return frozenset({'ALREADY_GOAL'})
    found=set()
    frontier=[(e.destination,budget-e.duration,e.option)
              for e in graph.edges.get(root,()) if e.duration<=budget]
    while frontier:
        node,left,first=frontier.pop()
        if node in graph.goals:found.add(first);continue
        for edge in graph.edges.get(node,()):
            if edge.duration<=left:
                frontier.append((edge.destination,left-edge.duration,first))
    return frozenset(found)


def physical_feasibility_interval(supported_options,certified_impossible=False):
    if supported_options and certified_impossible:
        raise ValueError('Contradictory existence and impossibility certificates')
    if supported_options:return (1,1)
    if certified_impossible:return (0,0)
    return (0,1)


def symbolic_nonidentifiability_witness():
    """Logical possible-world witness only. NEVER a robot training label."""
    observations={('s','logged-a'):'end-without-goal'}
    world_a=dict(observations,**{})
    world_a[('s','unobserved-b')]='goal'
    world_b=dict(observations,**{})
    world_b[('s','unobserved-b')]='end-without-goal'
    return dict(kind='ABSTRACT_LOGICAL_WITNESS_NOT_PHYSICAL_LABEL',observations=observations,
                world_a=world_a,world_b=world_b,
                feasible_a=True,feasible_b=False)
