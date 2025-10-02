# Agent Instructions

## Code Philosophy
- **Simplest solution that works**: Prefer boring, battle-tested patterns over clever abstractions (Worse is Better)
- **Fail fast and loud**: No silent failures, no fallback chains that mask errors (crash-only design)
- **Traceable execution**: Rich context at decision points—include IDs, input values, state transitions to make failures debuggable
- **Explicit over implicit**: Clear data transformations, explicit error types
- **Composition over inheritance**: Single responsibility per module/class
- **Determinism first**: Idempotent operations, predictable outputs

## Technical Preferences

**Type Safety & Data Flow:**
- Explicit typing throughout
- **Function signatures are contracts**: Never deviate from declared return types, input constraints, or documented behavior (Liskov Substitution)
- **Parse, don't validate**: Transform invalid input into typed errors at system boundaries, not scattered checks
- Clear input → transformation → output pipelines
- Return types that encode success/failure (Result<T, E>, Option<T>)

**Error Handling:**
- Explicit error handling via return types, not exceptions
- Propagate errors up, don't catch and ignore
- No generic "catch-all" handlers

**Code Organization:**
- Minimal public interfaces—expose only what's necessary
- Standard library first, external dependencies only when justified
- Flat module structures—avoid deep nesting

**Control Flow:**
- Minimize branching and nesting depth
- Early returns over nested conditionals
- Guard clauses at function entry

**Observability:**
- Concise status output showing what's happening
- Structured logs over print debugging
- Rich tracing: include relevant context (IDs, input values, state transitions) to make failures debuggable
- Log at decision points, not just outcomes

## Process Management
- **Never block the terminal**: Avoid commands that don't naturally exit (uvicorn, flask run, tail -f, watch, etc.)
- Use background processes with explicit job control: `uvicorn main:app &` or run in separate terminals
- Prefer one-shot commands for testing: `curl localhost:8000/health` over running the server interactively
- If a long-running process is necessary, acknowledge it explicitly and provide stop instructions

## Explicit Dislikes
- Vague requirements → Always clarify scope first
- Unnecessary comments → Code should be self-documenting
- Enterprise patterns for MVPs → YAGNI applies
- Feature creep → Implement exactly what's requested
- Redundant functionality → Remove dead code
- Mutable state scattered across modules → Centralize or eliminate
- Legacy support without justification → Don't preemptively support unused features

## Communication & Output Format
- Professional and concise; no verbose explanations
- Advanced CS/math vocabulary is welcome
- Code changes only—no "Here's what I did" preamble
- Brief summary line if multiple files changed: "Updated auth.py, added logging to db.py"
- Ask clarifying questions before implementation when scope is ambiguous
- Demonstrate understanding through code, not theory
- No emojis
