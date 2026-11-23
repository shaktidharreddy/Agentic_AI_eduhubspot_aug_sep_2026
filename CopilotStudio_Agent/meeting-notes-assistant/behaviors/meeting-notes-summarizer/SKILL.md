---
name: meeting-notes-summarizer
description: Turns raw meeting notes, transcripts or chat logs into a structured summary with key points, decisions, action items (owner and due date) and open questions. Use when the user shares meeting content and asks to summarize it, write a recap or minutes, or pull out decisions, action items, next steps or follow-ups.
---

# Meeting notes summarizer

Turn unstructured meeting content into a concise, structured summary that someone who missed the meeting can act on.

## When to use

- The user pastes or uploads meeting notes, a transcript, or a chat log and asks for a summary, recap, minutes, or "what did we decide".
- The user asks for only part of the summary, such as the action items or the decisions. Produce just that section.

Do not use this skill for content that is not a meeting or conversation, such as a report or an article. Summarize those normally.

## Inputs

- **Required:** the meeting content (notes, transcript, or chat log).
- **Optional:** meeting title, date, attendees, and purpose. Use them when given; never invent them.

If the user asks for a summary but has not shared any content, ask them to paste the notes or transcript. Ask nothing else up front.

## Procedure

1. Read all of the content before writing anything. Note the meeting's purpose, the participants, and any dates mentioned.
2. **Key points:** capture the main topics discussed and the important facts or updates shared. Merge repeated points and drop small talk.
3. **Decisions:** list only what the group actually agreed or approved. A suggestion that nobody confirmed is not a decision; list it under open questions instead.
4. **Action items:** for each task someone committed to or was asked to do, record:
   - the task, starting with a verb;
   - the owner, exactly as named in the content, or **Unassigned**;
   - the due date as stated, converted to a calendar date only when the meeting date makes it unambiguous, otherwise **Not specified**.
5. **Open questions:** list unresolved issues, risks, blockers, and anything parked for later, with who raised it when known.
6. Check the draft against the source. Every item must be traceable to the content; remove anything that is not.

## Output format

Use exactly this structure, in Markdown:

**Meeting summary:** <title or main topic> (<date, if known>)

**Attendees:** <names, if known; otherwise omit this line>

### Key points
- ...

### Decisions
- ... (write "No decisions recorded." when there are none)

### Action items
| # | Action | Owner | Due date |
|---|--------|-------|----------|
| 1 | ...    | ...   | ...      |

(write "No action items recorded." instead of the table when there are none)

### Open questions
- ... (write "None." when there are none)

Keep the summary under about 300 words unless the user asks for more detail. Write in the language of the content unless the user asks for another.

## Rules

- Never invent owners, dates, decisions, or numbers. Missing information stays **Unassigned** or **Not specified**.
- Keep names, figures, and product terms exactly as they appear in the source.
- Stay neutral: report what was said without judging the participants.
- For very long content, summarize each agenda item or section first, then merge the results.
- If parts of a transcript are garbled or ambiguous, say so in one line at the end rather than guessing.

## Follow-ups

After the summary, offer one short follow-up, such as drafting a recap email to the attendees or turning the action items into a checklist. Do only what the user asks for next.
