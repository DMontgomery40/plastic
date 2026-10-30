# Plastic

Plastic studies adaptation during a session, lasting changes across sessions, and
the decisions that retain or refuse those changes. This glossary names those
concepts without treating a proposed method or an accepted update as a measured
learning result.

## Language

### Learning and state

**Fast weights**:
The session-specific learned memory that changes while the model processes an
experience stream.
_Avoid_: slow weights, conversation history

**Activation state**:
The temporary recurrent and convolution context carried while processing inputs,
distinct from the fast weights.
_Avoid_: learned memory, lasting learning

**Session state**:
The carried state needed to continue a session, including its fast weights,
activation state, and any pending updates or momentum.
_Avoid_: fast weights alone

**Slow weights**:
The model parameters shared by fresh sessions, including learned initial fast
weights where the model has them.
_Avoid_: saved session state

**Adaptation**:
A change in behavior within an episode as the model processes its experience.
_Avoid_: lasting learning

**Lasting learning**:
An experience-induced improvement on held-out situations that survives clearing
temporary state and removing the experience stream, and disappears when the
pre-experience slow weights are restored.
_Avoid_: lower training loss, saved memory, fact recall alone

### Transactions and sessions

**Transaction**:
An attempted session-state update over an input chunk, together with the harness
decision about the change to retain.
_Avoid_: chat turn, Sleep run

**Proposed update**:
The memory change attempted before the harness applies its decision.
_Avoid_: accepted update, budget spent

**Accepted update**:
The actual retained memory change after the harness decision and final checks,
including any retained decay.
_Avoid_: proposed update, proof of useful learning

**Committed state**:
The session state retained after a transaction, from which continuation and forks
proceed.
_Avoid_: proposed state, slow weights

**Freeze**:
A control that prevents memory writes and decay while activation state continues
to advance.
_Avoid_: pause, undo all input influence

**Writes-disabled**:
A control that suppresses new writes without necessarily suppressing memory
decay.
_Avoid_: freeze

**Rollback**:
Refusal of a proposed memory update with frozen replay of the input; emitted
outputs and all activation influence are not undone.
_Avoid_: erase the turn, full reversal

**Chat turn**:
A user prompt and its generated reply, potentially spanning several transactions
with different learning eligibility and decisions.
_Avoid_: transaction

### Consolidation

**Sleep**:
Consolidation of session experience into slow weights, producing a candidate
child checkpoint.
_Avoid_: session save, demonstrated retention

**Selected turn**:
A chat turn admitted as Sleep material by provenance and flag-selection rules;
the experimental all-turn control deliberately admits refused turns too.
_Avoid_: verified truth, sanitized session state

**Damage gate**:
The available checks for specified deterioration after Sleep, distinct from
evidence that the child learned or avoided contamination.
_Avoid_: learning gate, retention guarantee

### Text learning contract

**Learning contract**:
The evaluation of transfer, adaptation speed, forgetting, correction and revert,
with acceptance and refusal counts reported separately.
_Avoid_: safety guarantee, automatic proof of rule knowledge

**Composition**:
An ordered application of named rule operators; a new ordered pair can still
produce the same answers as a training composition.
_Avoid_: necessarily novel behavior

**Situation**:
One application of a composition to an input word list, with its target answer.
_Avoid_: episode

**Episode**:
Several situations of one composition in one sequence, where later situations
follow earlier worked examples.
_Avoid_: independent examples

**Experience stream**:
The sequence of lesson episodes available to a learner for an update or an
in-context comparison.
_Avoid_: held-out evaluation material

**First situation**:
The first application in an episode, with no earlier worked example from that
episode; stated definitions or an explicit in-context prefix may still be present.
_Avoid_: automatically context-free

**Format-only control**:
Lessons with answers shuffled among situations of each episode, disrupting the
input-answer pairing while preserving the composition's name and decoration.
_Avoid_: content null

**Content null**:
The name-permuted lesson stream, where each composition name is consistently
paired with another composition's answers of the same arity.
_Avoid_: format-only control

**Clean-again control**:
A second clean update from the post-clean state, paired with a poisoned update
from that same state to distinguish poison effects from further training.
_Avoid_: another independent baseline, refusal count

**Revert check**:
A comparison after restoring pre-experience slow weights against the original
measurements on matched material.
_Avoid_: reset of fast state alone, transaction rollback
