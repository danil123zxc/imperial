# PRD: Chess Study Motivation App (Multi-Tutor Chess School)

## 1. Introduction / Overview

Students at a small chess school (one owner/admin, several tutors, each with their own students) practice inconsistently between lessons. This app gives them a short, rewarding daily practice habit — daily puzzle sets, streaks, XP, and spaced-repetition chess cards — while giving tutors visibility into who is practicing and tools to assign targeted homework, including positions taken from students' own imported games.

The app is a **responsive web application** (desktop and mobile browser). This PRD covers the **MVP in detail** plus an outline of Phase 2.

**Key roles:**
- **Admin** (school owner): manages tutors, shared content, sees school-wide stats.
- **Tutor**: manages their own groups/students, assigns homework and card decks, views their dashboard.
- **Student**: does daily puzzles, reviews due cards, completes homework, earns XP/streaks/badges.

## 2. Goals

- Get students practicing between lessons: target ≥60% of active students practicing 4+ days/week within 2 months of launch.
- Make practice sessions short and completable: a full daily session (puzzle set + due cards) takes under 15 minutes.
- Give every tutor a weekly-glance dashboard that answers "who practiced, who is slipping, what are they failing" in under 1 minute.
- Turn student mistakes into study material: failed puzzles automatically become spaced-repetition cards; tutors can create cards from students' own games.
- Enforce strict data isolation between tutors: a tutor can only ever see their own students.

## 3. User Stories

Stories are ordered so each builds on the previous ones. Each story is sized for one focused implementation session.

---

### Epic A: Foundation — accounts, roles, tenancy

### US-001: Core database schema with row-level security
**Description:** As a developer, I need the multi-tenant data model in place so all later features attach to the right school, tutor, and student.

**Acceptance Criteria:**
- [ ] Migration creates tables: `schools`, `profiles` (linked to `auth.users`; fields: `role` ('admin' | 'tutor' | 'student'), `display_name`, `school_id`, `timezone`), `groups` (`school_id`, `tutor_id`, `name`, `join_code`), `group_members` (`group_id`, `student_id`)
- [ ] Every domain table carries `school_id`
- [ ] RLS policies: students read/write only their own rows; tutors read rows of students in their groups; admins read school-wide
- [ ] Seed script creates one school, one admin, one tutor, one group, two students for local dev
- [ ] Typecheck passes

### US-002: Authentication and role-based app shell
**Description:** As a user, I want to sign in and land on the right home screen for my role.

**Acceptance Criteria:**
- [ ] Email/password and magic-link sign-in via Supabase Auth
- [ ] After sign-in, students land on Student Home, tutors on Tutor Dashboard, admins on Admin Panel
- [ ] Unauthenticated users are redirected to sign-in; users can never reach another role's routes (server-side check, not just UI hiding)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-003: Admin invites tutors
**Description:** As an admin, I want to invite tutors by email so signup stays closed to outsiders.

**Acceptance Criteria:**
- [ ] Admin panel has "Invite tutor" form (email input); creates a single-use invite link
- [ ] Opening the link lets the tutor set a password and creates a `profiles` row with role 'tutor' in the admin's school
- [ ] Used or expired (7-day) links show a clear error
- [ ] Admin sees a list of pending and accepted invites
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-004: Tutor creates groups and invites students by join code
**Description:** As a tutor, I want to create groups (e.g. "Monday beginners") and give students a join code so my roster builds itself.

**Acceptance Criteria:**
- [ ] Tutor can create/rename/archive groups; each group gets a 6-character join code
- [ ] Student signup page accepts a join code; on signup the student is added to that group and linked to its tutor
- [ ] A student can belong to multiple groups (many-to-many), including groups of different tutors
- [ ] Tutor sees group roster with display names; can remove a student from a group
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### Epic B: Puzzle library and daily practice

### US-005: Puzzle library with seed import
**Description:** As a developer, I need a puzzle bank so daily sets and homework have content from day one.

**Acceptance Criteria:**
- [ ] `puzzles` table: `fen`, `moves` (UCI solution line), `themes` (text[]), `rating`, `source`
- [ ] Import script loads a filtered subset (~20k puzzles, rating 600–2200, common tactical themes) from the Lichess puzzle database (CC0)
- [ ] Puzzles are queryable by rating range and theme with acceptable latency (indexed)
- [ ] Typecheck passes

### US-006: Interactive chess board component
**Description:** As a student, I want to make moves on a real board so solving feels like playing, not multiple choice.

**Acceptance Criteria:**
- [ ] Board renders any FEN; supports drag-and-drop and tap-tap moves (mobile friendly)
- [ ] Illegal moves are rejected (chess.js validation); last move is highlighted
- [ ] In "solve" mode the component accepts a solution line, auto-plays opponent replies, and emits solved/failed events
- [ ] Board is fully usable at 375px viewport width
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-007: Daily puzzle set
**Description:** As a student, I want a short daily set of puzzles at my level so I always know what to do when I open the app.

**Acceptance Criteria:**
- [ ] Each student gets a generated set of 6 puzzles per day, selected near their puzzle rating (±150) and weighted toward their weakest themes
- [ ] Student puzzle rating updates after each attempt (simple Elo vs. puzzle rating)
- [ ] Set persists for the day: leaving and returning resumes progress
- [ ] Completing the set shows a celebration screen with XP earned
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-008: Puzzle solving flow with attempt history
**Description:** As a student, I want clear feedback when I solve or fail a puzzle, and the app should remember my results.

**Acceptance Criteria:**
- [ ] Wrong move: gentle shake/red flash, one retry allowed, then "Show solution" becomes available
- [ ] Solution playback steps through the correct line move by move
- [ ] Every attempt is recorded in `puzzle_attempts` (`student_id`, `puzzle_id`, `first_try`, `solved`, `solved_at`)
- [ ] Failed puzzles are flagged for card creation (consumed by US-015)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### Epic C: Gamification

### US-009: XP and levels
**Description:** As a student, I want to earn points for everything I do so effort is always visible.

**Acceptance Criteria:**
- [ ] `xp_events` ledger table (`student_id`, `amount`, `reason`, `ref_id`); balances are computed, never stored as a mutable counter
- [ ] Awards (tunable constants in one config file): puzzle solved 10 XP (+5 first-try bonus), card review 2 XP, daily set completed +20 XP, homework item completed 50 XP
- [ ] Level = f(total XP) with a published threshold table; level and progress bar shown on Student Home
- [ ] Level-up moment shows a celebration animation
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-010: Daily streaks with weekly freeze
**Description:** As a student, I want a streak counter so I have a reason to come back every day.

**Acceptance Criteria:**
- [ ] A "streak day" is earned by completing the daily puzzle set OR clearing all due cards (whichever exists that day)
- [ ] Day boundaries use the student's own timezone
- [ ] One automatic "streak freeze" per calendar week: the first missed day consumes the freeze instead of resetting the streak, and the UI shows it was used
- [ ] Student Home shows current streak, longest streak, and freeze status
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-011: Weekly quests
**Description:** As a student, I want small weekly goals so practice doesn't feel like the same thing every day.

**Acceptance Criteria:**
- [ ] Every Monday (school timezone) each student gets 3 auto-generated quests, e.g. "Solve 20 puzzles", "Clear due cards 4 days", "Solve 5 pin-theme puzzles"; quest templates live in config
- [ ] Quest progress updates automatically from existing events (no manual claiming)
- [ ] Completing a quest awards bonus XP; completing all 3 awards an extra bonus
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-012: Badges
**Description:** As a student, I want achievements for milestones so bigger accomplishments are permanently commemorated.

**Acceptance Criteria:**
- [ ] Badge definitions in config: first solved puzzle, 7/30-day streak, 100/500 puzzles solved, first cleared review queue, first completed homework, 5 first-try solves in one day
- [ ] Badges are awarded automatically and exactly once; awarding is idempotent
- [ ] Badge case on the student profile shows earned (colored) and unearned (grayed) badges
- [ ] New badge triggers a toast/modal at the moment it is earned
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-013: Group leaderboard (effort-based, weekly)
**Description:** As a student, I want to see how my weekly effort ranks in my group so there's friendly competition anyone can win.

**Acceptance Criteria:**
- [ ] Leaderboard ranks by XP earned this week only (resets Monday, school timezone) — never by chess rating
- [ ] Scoped to the group; a student in multiple groups can switch between their group leaderboards
- [ ] Always shows the viewing student's own row even if outside the top 10
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### Epic D: Spaced-repetition cards (FSRS)

### US-014: Card and review data model with FSRS scheduling
**Description:** As a developer, I need cards and an FSRS scheduler so positions resurface at optimal intervals.

**Acceptance Criteria:**
- [ ] Tables: `decks` (`school_id`, `owner_tutor_id` nullable — null means shared school deck, `name`), `cards` (`deck_id`, `fen`, `prompt`, `solution_moves`, `source` ('manual' | 'failed_puzzle' | 'game')), `deck_assignments` (deck → student or group), `reviews` (per student+card FSRS state: stability, difficulty, due, reps, lapses, last_review)
- [ ] Scheduling uses the `ts-fsrs` library with default parameters; grade in → next due date out
- [ ] Unit tests cover: new card, correct review, failed review, and relearning transitions
- [ ] Typecheck passes

### US-015: Review session — answer on the board, auto-graded
**Description:** As a student, I want to review due cards by playing the move on the board, without having to grade myself.

**Acceptance Criteria:**
- [ ] "Reviews due: N" entry point on Student Home; session presents due cards one at a time on the interactive board
- [ ] Auto-grading maps to FSRS: correct first try → Good; correct after one wrong try → Hard; wrong/gave up → Again (solution then shown)
- [ ] Failed daily puzzles (from US-008) automatically create cards in a personal "My mistakes" deck, deduplicated by FEN
- [ ] Session end screen: cards reviewed, accuracy, XP earned
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-016: Tutor creates decks and cards
**Description:** As a tutor, I want to build card decks (opening lines, endgame positions) and assign them so my lesson material gets rehearsed all week.

**Acceptance Criteria:**
- [ ] Tutor can create a deck and add cards by: pasting a FEN, setting up a position on an editor board, or picking a library puzzle
- [ ] Each card gets a prompt (e.g. "White to move — what's the plan?") and a solution line entered on the board
- [ ] Deck can be assigned to a group or an individual student; assignment creates review state for those students
- [ ] Admin can mark a deck as shared (school-wide, visible to all tutors)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### Epic E: Game import and homework

### US-017: Link Lichess/Chess.com accounts and import games
**Description:** As a student, I want my online games to appear in the app so my tutor can use them for homework.

**Acceptance Criteria:**
- [ ] Student settings page accepts a Lichess username and/or Chess.com username
- [ ] "Sync games" fetches the latest 50 games per platform (Lichess export API, Chess.com published-data API), storing PGN, result, time control, played_at, and external ID (deduplicated)
- [ ] Sync runs on demand plus nightly per linked student
- [ ] Import failures (bad username, API down) surface a readable error and never lose previously imported games
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-018: Tutor game viewer with "make a card from here"
**Description:** As a tutor, I want to step through a student's game and turn any position into a card or homework item in one click.

**Acceptance Criteria:**
- [ ] Tutor sees a student's imported games list; opening a game shows a board with move-by-move navigation (keyboard arrows + buttons)
- [ ] "Create card from this position" pre-fills a card with the current FEN; tutor enters the solution line and target deck
- [ ] Created card is tagged `source: 'game'` and linked back to the game and move number
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-019: Homework assignment and tracking
**Description:** As a tutor, I want to assign homework with a due date and see who finished it.

**Acceptance Criteria:**
- [ ] Homework = a named bundle of items (specific puzzles, a deck to clear, or "N puzzles of theme X"), assigned to a group or student, with a due date
- [ ] Students see pending homework on Student Home with due date and progress; completion awards homework XP
- [ ] Tutor view shows per-student completion status (not started / in progress / done / overdue)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

### Epic F: Dashboards

### US-020: Tutor dashboard
**Description:** As a tutor, I want a weekly-glance view of all my students so I know who to praise and who to nudge at the next lesson.

**Acceptance Criteria:**
- [ ] One row per student: last active date, current streak, puzzles solved this week, due-card backlog, homework status, top 2 weakest themes (lowest solve rate, min 5 attempts)
- [ ] Students inactive 3+ days are visually flagged
- [ ] Row click opens student detail: activity graph, badge case, recent games, decks
- [ ] Dashboard only ever contains the tutor's own students (verified by RLS test)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-021: Admin panel
**Description:** As the school owner, I want to see the whole school and manage tutors so the program stays healthy.

**Acceptance Criteria:**
- [ ] Tutor list with per-tutor aggregates: student count, % of students active this week, homework issued
- [ ] School-wide stats: weekly active students, total puzzles solved, average streak
- [ ] Admin can move a student between groups (including across tutors) and deactivate accounts
- [ ] Admin manages the shared deck library (promote/demote shared status)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

## 4. Functional Requirements

**Tenancy & access**
- FR-1: The system must support three roles — admin, tutor, student — within a single school.
- FR-2: Every data access must be constrained by role via Postgres RLS: students see only their own data; tutors see only students in their groups; admins see the whole school.
- FR-3: Signup must be closed: tutors join via admin email invite; students join via a group join code.
- FR-4: Student–group membership must be many-to-many; a student may have multiple tutors via multiple groups.

**Practice content**
- FR-5: The system must maintain a puzzle library imported from the Lichess puzzle database (CC0), filtered to rating 600–2200.
- FR-6: The system must generate a 6-puzzle daily set per student, calibrated to their puzzle rating (±150) and weighted toward weak themes.
- FR-7: The system must track a per-student puzzle rating updated by an Elo-style formula after each attempt.
- FR-8: All puzzle solving must happen on an interactive board with legal-move validation; no multiple-choice answers.

**Spaced repetition**
- FR-9: Card scheduling must use FSRS (via `ts-fsrs`), with per-student-per-card review state.
- FR-10: Review grading must be automatic from board input: first-try correct → Good; correct after retry → Hard; failed → Again.
- FR-11: Puzzles failed in the daily set must automatically create cards in the student's personal "My mistakes" deck (deduplicated by FEN).
- FR-12: Tutors must be able to create decks/cards from FEN, a board editor, or library puzzles, and assign them to students or groups.

**Gamification**
- FR-13: XP must be recorded as an append-only event ledger; all balances derived from it.
- FR-14: Streak days must be computed in the student's timezone, with one automatic streak freeze per calendar week.
- FR-15: Weekly quests must auto-generate every Monday and self-track from existing activity events.
- FR-16: Badges must be awarded idempotently from config-defined rules.
- FR-17: Leaderboards must rank by weekly XP (effort), never chess rating, and default to group scope.

**Games & homework**
- FR-18: Students must be able to link Lichess and Chess.com usernames; the system must import their recent games on demand and nightly.
- FR-19: Tutors must be able to navigate an imported game and create a card from any position in ≤2 clicks.
- FR-20: Homework must support due dates, group or individual assignment, and per-student completion status visible to the tutor.

**General**
- FR-21: All student-facing screens must be fully functional at 375px viewport width.
- FR-22: A complete daily session (daily set + typical due cards) must be finishable in under 15 minutes.

## 5. Non-Goals (Out of Scope for MVP)

- No playing chess against other users or an engine (Lichess does this better).
- No chess-rating-based rankings anywhere in the product.
- No automated blunder detection / engine analysis of imported games (Phase 2 — MVP relies on the tutor picking positions).
- No Anki `.apkg` export (Phase 2).
- No push/email notifications or nudges (Phase 2).
- No video lessons, chat, or messaging — lessons stay with the tutor.
- No payments, billing, or subscription management.
- No native mobile apps; responsive web only.
- No multi-school support in the UI (schema carries `school_id` so it stays possible later).
- No parent accounts.

## 6. Phase 2 Outline (planned, not in this build)

1. **Anki export** — "Export my mistakes" generates an `.apkg` (position image front / solution back) for students who use Anki.
2. **Automated blunder mining** — server-side Stockfish pass over imported games; candidate mistake positions suggested to the tutor as one-click cards.
3. **Nudges** — email notification to student (or tutor) after 3 inactive days; weekly tutor digest.
4. **Team challenges** — group vs. group weekly effort competitions.
5. **Cosmetic unlocks** — board themes and piece sets purchasable with XP.
6. **Opening repertoire trees** — multi-move variation trees per repertoire, drilled as card sequences.
7. **PWA install + push notifications** — home-screen install and streak reminders.

## 7. Design Considerations

- Mobile-first: most students will practice on phones; the board is the hero element on every practice screen.
- Kid-friendly but not childish: celebratory moments (confetti on set completion, badge toasts) with restrained visual style; suitable for ages ~8–18.
- Effort framing everywhere: copy praises practice ("12 puzzles today!") not talent or rating.
- Color-blind-safe palette for success/failure states (not red/green alone — add icons/shapes).
- Empty states matter: a student with no homework and a finished daily set should see "You're done for today 🎉", never a dead end.
- Board rendering: `react-chessboard` (MIT) + `chess.js` for rules; avoid GPL-licensed chessground unless the project accepts GPL implications.

## 8. Technical Considerations

- **Stack:** Next.js (App Router) on Vercel; Supabase for Postgres, Auth, and RLS. Nightly game sync via Vercel Cron hitting a server route.
- **RLS is the security boundary**, not UI checks. Write RLS tests early (e.g. tutor A queries tutor B's student → zero rows). Server components should use the user-scoped Supabase client, never the service key, for user-facing reads.
- **FSRS:** `ts-fsrs` npm package; store full FSRS state per review row so parameter tuning later doesn't require migration.
- **Puzzle source:** Lichess puzzle DB is a CSV (CC0) — import offline via script, don't call the Lichess puzzle API at runtime.
- **Game APIs:** Lichess game export API (NDJSON, no key required, rate-limited — throttle politely); Chess.com published-data API (monthly PGN archives, no key). Both fetched server-side.
- **Timezones:** store IANA timezone per profile; compute streak days and quest weeks server-side with explicit timezone math. This is a classic bug source — unit-test the day-boundary logic.
- **XP as ledger:** append-only `xp_events` prevents double-award bugs and makes weekly leaderboards a simple `WHERE created_at >= week_start` aggregate.
- **Minors' data:** store first name / display name only; no birthdate, no photos. Games imported from public APIs are already public data.

## 9. Success Metrics

- ≥60% of active students practice on 4+ distinct days per week (by week 8).
- Median current streak across active students ≥5 days.
- Homework completion rate ≥70% by due date.
- Every tutor opens their dashboard at least weekly.
- Median daily session length between 8 and 15 minutes (long enough to matter, short enough to sustain).
- 30-day student retention ≥70% (still active in week 5 after signup).

## 10. Open Questions

1. **Localization:** English only at launch, or is a second language (e.g. Russian) needed for the student audience? Affects copy architecture from day one.
2. **Parental consent:** does the school's enrollment paperwork already cover online-tool consent for minors (GDPR/COPPA-equivalent), or does the app need a consent step at signup?
3. **Daily set size:** 6 puzzles is the default — should tutors be able to override per student (e.g. 4 for young beginners, 10 for tournament players)?
4. **Leaderboard opt-out:** should a tutor be able to hide a specific student from leaderboards (anxiety, parental request)?
5. **Puzzle rating visibility:** show the student their internal puzzle rating (motivating for some, discouraging for others), or keep it hidden and tutor-only?
6. **Streak repair:** beyond the weekly freeze, should there be a paid-in-XP "streak repair" (Duolingo-style), or is that too gamey for the school's culture?
