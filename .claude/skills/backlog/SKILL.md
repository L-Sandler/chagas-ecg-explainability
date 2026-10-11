---
name: backlog
description: >
  Use this skill when the user wants to log a backlog item, improvement, or follow-up
  without derailing current work. Trigger on phrases like: "note this for later",
  "add to backlog", "we should improve this", "don't forget to", "log this idea",
  "technical debt", "future improvement", "/backlog", or "let's come back to this".
  Also trigger in triage mode when the user says "review backlog", "what's in the backlog",
  "let's triage", or "clear the backlog".
---

# Backlog Skill

This skill has two modes: **capture** (fast, non-distracting) and **triage** (review and act).
The goal is to honor good ideas without letting them pull the user off their current task.

---

## Backlog File

Items are stored at `notes/backlog.md` in the project root. If it doesn't exist, create it with this header:

```markdown
# Project Backlog

Items here are captured quickly during focused work. Triage regularly to keep this actionable.

---
```

Each item follows this format (one item = one block):
```markdown
- [ ] **[CATEGORY]** Description of the item
  - _Added: YYYY-MM-DD | Context: brief note on what triggered this_
```

Categories: `IMPROVEMENT`, `TECH-DEBT`, `FEATURE`, `BUG`, `QUESTION`, `RESEARCH`

---

## Mode 1: Capture

When the user wants to log something quickly:

### Step 1: Extract the item
Parse the user's message to identify:
- What the item is (the actionable thing to do or investigate)
- The category (infer from context if not stated)
- The context (what file, function, or situation triggered this)

If ambiguous, ask ONE clarifying question max. Don't ask about priority — that's for triage.

### Step 2: Write the item
Append to `backlog.md` using the Read + Edit tools. Do NOT rewrite the whole file.

### Step 3: Confirm and redirect
Reply with exactly:
```
Logged: [one-line summary of the item]
Back to [what you were doing].
```

Then immediately continue whatever work was in progress. Do not ask follow-up questions.
Do not discuss the item further unless the user brings it up.

**Anti-patterns to avoid during capture:**
- Do not start implementing the improvement
- Do not rate or prioritize it
- Do not expand on why it's a good idea
- Do not suggest related improvements
- Total time cost to the user: under 30 seconds

---

## Mode 2: Triage

When the user asks to review the backlog (not during active implementation work):

### Step 1: Read the current backlog
Read `backlog.md` and present the open items (unchecked `- [ ]`) grouped by category.
Show the count per category.

### Step 2: Walk through items together
For each item (or a batch the user wants to focus on), ask:
- **Act now?** → help scope the work and start it
- **Keep for later?** → leave as-is, optionally add a note
- **Drop?** → mark as dropped with a brief reason: `~~item~~ — dropped: reason`
- **Defer to spec?** → if it's a real architectural decision, help the user add it to `spec/spec.md` and remove from backlog

### Step 3: After triage
- Mark completed items with `[x]`
- Update the file
- Summarize: "X items cleared, Y kept, Z remain open."

---

## Graduated priority

Don't assign numeric priorities. Instead, when an item has been in the backlog for >2 weeks
and the user is about to start a new phase of work, surface it: "You have 3 older backlog
items that might be relevant to what you're starting — want to triage them first?"

---

## Key principles

- **Speed is the point.** Capture should never take longer than naming the thing.
- **Backlog is not a graveyard.** If it's been there >3 weeks without triage, it's probably noise. Say so during triage.
- **One file, always readable.** Keep `backlog.md` human-scannable without Claude — plain markdown, no special syntax.
- **Return to flow.** After capture, always redirect back to whatever the user was doing.
