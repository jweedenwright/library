# Level 2: AI Explorer

AI models don't understand text like humans do - they predict it.

Decoder-only models (like Claude or GPT) generate text one token at a time based on your prompt and context window. Other types - such as encoder-only models like BERT - analyze or classify text rather than generate it.

In Level 1, you learned what Generative AI and NLP are. Now you'll see how your model actually reasons, remembers, and decides what to say next - the foundation for everything else in this course.

---
## Exercise: Tune the Personality of the Model

### Goal
Experiment with core sampling parameters to see how they change a model’s reasoning style and the feel of its responses. This lesson builds a foundation for understanding how models decide what to say next.
We’re intentionally focusing on model behavior knobs in isolation here; future lessons will touch on context structure, memory, and inference strategies.

We’re intentionally focusing on model behavior knobs in isolation here; future lessons will touch on context structure, memory, and inference strategies.
Setup
1. Go to [Google AI Studio](https://aistudio.google.com/prompts/new_chat) and sign in.
2. In the settings panel on the right, open Advanced Settings.
3. Find Temperature and Top-p. (Top-k isn't shown.)
You can find the chat input at the bottom of the page. Keep the same test prompt for each run but refresh page between runs:
You are a productivity coach. A developer says: "I have 47 open browser tabs and 12 half-finished tasks." Give them advice to help them focus.

* **Temperature**: lower temp (0.2) results in less randomness and more specific/robotic results. Higher temps (2.0) increase creativity and unpredicability (jokes/metaphors)
    * Temp of 2.0 - `You do not need to sort 47 tabs. Sorting is just productive procrastination.`
* **Top-p**: Lower the value, the smaller the pool of tokens that the model has access to

---
## Exercise: Rule Files

In Lesson 1, you learned how inference parameters like temperature and top-p influence how a model writes — its creativity, stability, and randomness.

In this lesson, you’ll shift from shaping output style to shaping the model’s actual behavior using rule files. These live in the developer layer of the context hierarchy (system > developer > user > chat history) and act as persistent, repo-level instructions that govern how your coding assistant works.

A strong rule file defines how the assistant should think and act, step-by-step:
• What to check first (linting, tests, documentation)
• What actions to take next (suggest fixes, summarize diffs)
• What to avoid (deleting files, rewriting untested code)

The result is a lightweight, repeatable workflow that keeps AI contributions safe, predictable, and aligned with your team’s standards.

Learning Resources:
- [Cline Docs - Rules File Reference](https://docs.cline.bot/features/cline-rules)
- [Zed Docs - AI Rules](https://zed.dev/docs/ai/rules)
- [Claude Code - Memory and Project Files](https://docs.claude.com/en/docs/claude-code/memory)
- [Cursor Rules - Examples](https://github.com/PatrickJS/awesome-cursorrules)

---
## MCP Fundamentals

By now, you’ve learned two core ideas:
* Lesson 1: Models reason through layered context — system → developer → user → chat history.
* Lesson 2: Rule files live in the developer layer, giving you persistent instructions inside your repo.

Now we’ll shift from instructions to capabilities.

Your coding assistant can already read files and run commands through its IDE integration — but those abilities are fixed. You can’t extend them or connect the assistant to new systems on your own. That’s where the Model Context Protocol (MCP) comes in.

MCP is a tooling layer: a standard way to add new, permissioned tools your assistant can call when needed. With MCP, you can securely plug in capabilities like API access, database queries, org-specific scripts, or custom workflows.
In short:
* **Rule** files tell the model **what it should do**.
* **MCP** gives the model **new things it can do**.

Learning Resources:
* [Model Context Protocol - Getting Started](https://modelcontextprotocol.io/docs/getting-started/intro)
* [Model Context Protocol - Arch, Servers, Clients, Versioning](https://modelcontextprotocol.io/docs/learn/architecture)
* [MCP Clearly Explained - Video](https://www.youtube.com/watch?v=7j_NE6Pjv-E)

## Exercise: Using Your MCPs

### Goal:
See how MCPs extend your coding assistant’s capabilities and how rule files in the developer layer guide when and how those capabilities are used

### Setup:
Within the ToDo App repo you will find an MCP-Setup-Guide.md file with instructions on how to set up the Context7 & Playwright MCP for Claude Code, Zed, and Cline. For any other tool you will need to look up the documentation for its setup.

## MCP Security

Model Context Protocol (MCP) gives AI tools a standardized way to interact with files, APIs, and local processes. This makes development workflows more powerful and automated — but like any integration layer, it also introduces responsibilities around configuration and access.

Because MCP is still evolving, some of the security patterns we see in mature RPC or plugin systems aren’t fully built out yet. Community and security research highlight areas where thoughtful setup makes a meaningful difference: how servers are hosted, what they can access, and how broadly they’re exposed.

This lesson focuses on the practical engineering side of MCP — what types of access it provides and what simple guardrails help ensure you're using it safely and predictably. You'll learn how to evaluate MCP tools in your environment using standard software-engineering principles like least privilege, scoping, and dependency awareness.

Learning Resources:
* [Microsoft Tech Community - MCP Security](https://techcommunity.microsoft.com/blog/microsoft-security-blog/secure-model-context-protocol-mcp-implementation-with-azure-and-local-servers/4449660)
* [MCP Vulnerabilities Every Dev Should Know](https://composio.dev/blog/mcp-vulnerabilities-every-developer-should-know)
* [MCP: An Accidentally Universal Plugin System](https://worksonmymachine.ai/p/mcp-an-accidentally-universal-plugin)

**NOTE**: using Playwright MCP, the --allowed-urls, --allowed-origins, and --allowed-hosts all allowed access to other domains, so there may be an actual security vulnerability there (integrated through cline for VS Code using Claude 3.7 sonnet)

## Automated PR Review

Setup claude code to run and do code review on merge request of gitlab.

Needed these 3 vars in gitlab: ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, GITLAB_TOKEN

* .gitlab-ci.yml
```
stages:
  - review

# Automatic MR review job using Claude Code CLI
# Runs on every merge request to provide AI-powered code reviews
claude_review:
  stage: review
  image: node:24-alpine3.21
  before_script:
    # Validate required environment variables
    - |
      if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$ANTHROPIC_BASE_URL" ] || [ -z "$GITLAB_TOKEN" ]; then
        echo "ERROR: Required environment variables not set"
        echo "Please configure: ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, GITLAB_TOKEN"
        exit 1
      fi
    # Install required dependencies
    - apk add --no-cache git bash curl jq
    # Install Claude Code CLI
    - npm install -g @anthropic-ai/claude-code
  script:

    # Set up error handling to capture and print errors from claude
    - |
      error_handler() { echo "Error occurred on line $1"; cat /tmp/review.md; exit 40;}

    # Trap errors and call the handler function
    - |
      trap 'error_handler $LINENO' ERR

    # Run Claude Code in headless mode to review the MR and save output
    - |
      claude -p "Review ONLY the changes in this merge request. Do not review the entire codebase - focus specifically on the files and lines that were modified in this MR.

      $(cat rules/code-review-assistant.md)" \
      --model claude-3-7-sonnet-latest \
      --permission-mode acceptEdits \
      --allowedTools "Read(*) Grep(*) Glob(*)" \
      --output-format text > /tmp/review.md

    # Post the review as a comment on the MR using GitLab API
    - |
      # Verify review file was created and is not empty
      if [ ! -f /tmp/review.md ] || [ ! -s /tmp/review.md ]; then
        echo "ERROR: Review file is missing or empty"
        exit 2
      fi

      REVIEW_CONTENT=$(cat /tmp/review.md)
      FULL_COMMENT="## 🤖 Claude Code Review

      ${REVIEW_CONTENT}

      ---
      *Automated review powered by Claude Code*"

      # Disable command echoing to prevent token exposure in logs
      set +x

    - |
      cd

      # Post the review and check for errors
      if jq -n --arg body "$FULL_COMMENT" '{body: $body}' | \
        curl --fail --silent --show-error --request POST \
          --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
          --header "Content-Type: application/json" \
          --data @- \
          "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}/notes"; then
        echo "Review posted successfully!"
        cat /tmp/review.md
      else
        echo "ERROR: Failed to post review comment"
        exit 3
      fi
  rules:
    # Only run on merge request events
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  variables:
    # Clone full git history for better context
    GIT_DEPTH: 0
    # ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY should be set in GitLab CI/CD Variables
    # GITLAB_TOKEN should also be set in GitLab CI/CD Variables
```

## Putting it all together

This final exercise brings everything from Level 2 together. You’ll take a real feature request from a PRD, plan it using a task-specific rule file, implement it with Act Mode, test it with the Playwright MCP, and push it through a full CI/CD review cycle.
Think of it as the complete AI-assisted workflow:
Plan → Build → Test → Review → Iterate → Ship.
Each step reinforces the core ideas from this level — rule files for process, Ask/Act Modes for intentional behavior, and MCPs as capability layers.
Step 1 — Read Through the PRD
In your ToDo App repo, open exercises/PRD-Final.md.
This PRD outlines the Todo Categories feature you’ll be building.
Step 2 — Planning Phase (Ask Mode): Create a Task-Based Rule File
Before starting this exercises create a new branch to work off of. You will create an MR later on.
Creating a Planning Rule File:
1. Locate the template: rules/planning-assistant-template.md
2. Copy the entire contents of that template.
3. Create a new file (e.g., rules/planning-assistant.md).
4. Paste the template contents into your new file.
5. Fill out the template with your own structured planning workflow.
6. Keep the original template file unchanged so it can be reused.
Run in Ask Mode:
Use the `planning-assistant-template.md` rule file to plan the Todo Categories feature.
Step 3 — Implementation Phase (Act Mode)
Switch to Act Mode and ask your coding assistant to implement the feature based on the PRD and your planning rule file.
Step 4 — Testing Phase (Act Mode with Playwright MCP)
Use Act Mode with the Playwright MCP to test the feature.
Ask your assistant to:
Run end-to-end tests for the Todo Categories feature
Verify:
  1. Creating a new category
  2. Assigning one or more categories to a todo
  3. Displaying category labels in the todo list
  4. Filtering todos by category

If anything fails:
• Capture screenshots
• Explain what went wrong
• Generate missing Playwright tests
• Run the updated tests again
Step 5 — Code Review Phase (CI/CD)
Commit your changes:
feature: categories
Push your branch and open a Merge Request in GitLab.
Review the automated feedback from your code review assistant.
Step 6 — Iteration Phase
Address the issues surfaced during code review. Your goal is to refine your implementation until the major concerns are resolved.
Commit your changes:
CI/CD Review Feedback
Once the changes are solid: merge your code. If you can't merge check that you are comparing against your fork and not main
