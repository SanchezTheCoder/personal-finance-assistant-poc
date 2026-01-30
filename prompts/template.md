# [FEATURE NAME] - Ralph Wiggum Loop Prompt

Read specs/README.md for system context.
Read specs/[FEATURE]-plan.md for the task list.

Pick the most important thing to do from the plan (first incomplete task).
Do exactly ONE task, then EXIT immediately.

## Build & Verify

```bash
# Verify your changes work
uv run uvicorn backend.main:app --reload --port 8000 &
sleep 3
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"utterance": "test query here"}'

# Run eval to check for regressions
curl -X POST http://localhost:8000/eval
```

## Workflow

1. Read specs/README.md (system index)
2. Read specs/[FEATURE].md (feature spec)
3. Read specs/[FEATURE]-plan.md (task list)
4. Pick first incomplete task
5. Implement the task
6. Run verification commands above
7. Commit with descriptive message
8. Mark task as ~~DONE~~ in specs/[FEATURE]-plan.md
9. EXIT

## FINAL STEP: Update specs/README.md
If you add new files or patterns, update the PIN index.

## FINAL STEP: If behavior changes
Update specs/[FEATURE].md to reflect the new behavior.

## Exit Conditions

When all tasks in specs/[FEATURE]-plan.md are ~~DONE~~, output:

```
RALPH_LOOP_COMPLETE_X9K7
```

Then EXIT immediately.
