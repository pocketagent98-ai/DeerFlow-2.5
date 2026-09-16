---
name: game-factory
description: Use this skill when the user wants to plan, research, or architect a game or app before building it. Trigger on requests like "plan a game", "make a game bible", "break this game into tasks", "research this game idea", "create a development plan for my game", or whenever a bible.md / game brief is provided. Produces a Game Blueprint, Game Bible, and an Atomic Task DAG ready for user approval before any coding begins.
---

# Game Factory Skill — Research → Blueprint → Bible → Atomic Task DAG

## Overview

This skill turns a simple game idea (one sentence or a `bible.md` file) into a
complete, verified development plan BEFORE any code is written. It is the
"Research Director + Game Architect + Planner" stage of the Game Factory
pipeline. It never writes the actual game code — its output is research,
decisions, architecture, a Game Bible, and an Atomic Task DAG, which are then
handed to the user for approval, and after approval to the coding stages.

## Pipeline Position

```
Game Brief / bible.md
        |
   [THIS SKILL]
        |
 Understand -> Deep Research -> Game Bible -> Atomic Task DAG
        |
  USER APPROVAL GATE  (mandatory stop; present the plan, wait for approval)
        |
  (downstream: coding / build / test / fix — NOT part of this skill)
```

## Phase 1 — Understand the Game

Extract and record: genre, target platforms, core loop, target audience,
monetization assumptions, art direction, technical constraints,
multiplayer/single-player, backend needs, expected scope, and unknowns.

Then ask: "What do I need to know to plan this game completely?"
That question produces the Research Mission.

## Phase 2 — Research Mission (objective-driven, time-budgeted)

Never run research as "search the internet and make a report". Build:

1. Research objectives (what must be known before architecture can be fixed)
2. Research questions per objective
3. A research plan with parallel branches:
   gameplay / world / combat / progression / economy / AI / art / audio /
   technical (engine-specific) / performance / networking / backend /
   security / platforms / monetization / legal

Research budget:
- MINIMUM: 20 minutes
- STANDARD: 30-45 minutes
- DEEP: 45-60 minutes (hard maximum)

Completion rule: time alone never ends research. Research is complete only
when: objectives are complete AND evidence is sufficient AND major
contradictions are checked AND a synthesis pass is done. If the budget
expires, stop and synthesize what was gathered (never discard findings).

Dynamic research: after each wave, ask "what else do I need to know?" and open
new branches when findings reveal gaps (e.g. a chosen architecture is too
expensive for mobile → new branch: mobile-friendly alternatives).

Source verification: for every load-bearing claim, compare across multiple
sources, note agreement/contradiction, and attach a confidence level. Never
treat a single source's claim as fact.

## Phase 3 — Game Architecture

Decide the full system tree and dependencies:

```
Game
├── Core systems ├── Gameplay ├── Player ├── Combat ├── Inventory
├── Mission      ├── World    ├── AI    ├── UI    ├── Audio
├── Assets       ├── Backend  ├── Save  ├── Localization
├── Security     └── Platform
```

For every system, list what it depends on and what depends on it.

## Phase 4 — Game Bible

Write the long-term source of truth:
- What the game is
- How the project is structured
- What decisions were made and why
- What has already been built (starts empty)
- Project state: completed_tasks, failed_tasks, current_task, current_build,
  known_errors, attempted_fixes, successful_fixes, files_modified

## Phase 5 — Atomic Task DAG

The most important deliverable. Rules:

- NEVER emit a task like "Make combat system".
- Break every system into atomic, executable tasks. Example for combat:
  create combat state -> attack input -> hit detection -> damage interface ->
  enemy damage receiver -> cooldown logic -> animation hooks -> audio hooks ->
  VFX hooks -> write tests -> integration test.
- Every atomic task MUST carry:
  - task_id, goal, description
  - dependencies (task_ids)
  - required_files, allowed_files, do_not_touch_files
  - acceptance_criteria, test_criteria
  - expected_output, risk, research_references
- The "do_not_touch" list is as important as the task itself: the coder must
  know both what to do and what NOT to modify.

## Phase 6 — Approval Gate

Present the Blueprint + Game Bible + Task DAG summary to the user and STOP.
No coding begins until explicit approval. On rejection, revise the affected
phases only.

## Boundaries (what this skill does NOT do)

- No actual large-scale coding (that is the coding harness's job downstream)
- No agent execution routing
- No unsafe host execution
- No asset generation (only decide which assets are needed, style, format,
  and whether each is procedural / AI-generated / 3D-tool-made)
- No device testing

## Output Format

Deliverables are markdown documents written under `game_bible/`:
- `game_bible/blueprint.md` — the synthesized design from research
- `game_bible/bible.md` — the long-term source of truth
- `game_bible/task_dag.md` — the atomic task DAG with full task metadata
