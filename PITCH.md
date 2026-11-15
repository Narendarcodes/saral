# Saral: a voice agent that tells you why your loan was rejected

*Solution pitch for the Great Indian AI Internship Challenge by Gnani AI. Built solo by Golla Narendar, CSE (AI & ML), CBIT Hyderabad.*

## The problem
In India, a rejected loan applicant usually hears one word: **declined**. No reason, no next step. People with good credit scores are turned down for things they never learn about, such as frequent job changes or a high existing EMI load. So they apply again somewhere else and get rejected for the same reason. The people hit hardest are the ones least likely to read a policy PDF or a chat window, but every one of them can answer a phone call.

## The solution
Saral calls the borrower back in their own language and explains the decision.

1. **Recognises the caller** by their registered mobile number and greets them in their language. They can press 9 to switch to any of the 10 supported languages.
2. **Verifies them** with the last 4 digits of their application number, entered on the keypad. The digits never reach the AI and are never logged. Until the caller is verified, Saral plays only pre-recorded prompts.
3. **Explains the real reason** in plain words, read from the lender's actual decision, so it never guesses.
4. **Gives next steps**: ask for a lender review, attach the right documents, add a co-applicant, or apply for a smaller amount.

Saral **explains, never decides**. It cannot approve, reverse or promise a loan.

## How it uses Gnani AI
- **Prisma** (speech to text) understands the caller in their language.
- **Timbre** (text to speech) answers in the same language, with amounts spoken as words.
- Built for all 10 Gnani languages, and tested in Telugu and Hindi.

## How it is built
LiveKit voice pipeline → keypad (DTMF) verification → an agent grounded in the lender's decision and policy rules → Timbre. A web phone page lets anyone try the full call from a browser. All demo data is synthetic.

## Why it matters
A clear reason turns a dead end into a plan. Lenders get fewer repeat rejections and support calls, and borrowers stop guessing.

**Demo video:** see the LinkedIn post linked in the submission. **Code:** this repository ([README](README.md)).
