---
chapter_id: prompt-and-constrain
---

# Prompt a small model and constrain its answers to the teams

By the end of this chapter, you can write a router prompt for a small open model, constrain its answer so that it can only be a team name, and measure how many of those answers are right. You will also see why a valid answer is not a correct one, which is why the next chapter trains a model instead of asking it better questions.

A bank's support inbox, which this book calls the inbox, sends each message to one of 77 teams. [Chapter 1](route-a-message) asked a small model for one team in free text, and the model did not return teams. A router can send a message on only when the answer is a team, so this chapter first makes the model's answer a team by construction. It then runs the two open models from [chapter 3](choose-a-base-model), SmolLM2-135M-Instruct and SmolLM2-360M-Instruct, with one prompt and one constraint, and measures what they return. The 135M model is the one [chapter 5](fine-tune-with-lora) trains, so the counts here are the ones that training has to beat.

Two kinds of number appear, and the chapter keeps them apart. The live examples run on the first six messages of `test_40`, the 40-message fixed sample of the test split that [chapter 1](route-a-message) uses, and the same six for both models. They show the mechanism. Six messages cannot separate two models, so the comparison uses a recording: each model and prompt answered all 200 messages of `test_200`, a fixed sample of 200 test messages that includes those six, once, and the answers were saved. The example that reads the recording loads no model. Its counts come from the saved answers and nothing else.

## The prompt, settled before any recording {#the-prompt-settled-before-any-recording}

Every example in this chapter uses one router prompt. It tells the model that the message comes from an online bank, asks for one support team and the team name only, lists the 77 teams, and ends with `Team:`, so the model's next words are the answer. A raw model that has not been trained for the task gets nothing else, so the wording matters. This chapter does not tune it. The prompt was fixed before any recording, and the recording's manifest holds the prompt's hash. A changed prompt is a different recording, and its counts cannot be set beside the ones here.

The list of teams in the prompt is the same list the constraint uses. The trie in the next section is built from the same label names in the same order, so the prompt and the constraint cannot disagree about which teams exist. The same dependency works the other way. A reader who adds or renames a team changes the prompt, and so changes the recording's hash. The counts in this chapter then no longer describe that prompt, and the reader must record again before comparing.

The raw form passes that text to the model as it stands. The chat form wraps the same text in the model's chat template, which adds a default system prompt. The chat form runs in its own example on the same six messages:

[Example: constrained-decode-135m-chat](../examples/constrained-decode-135m-chat.py)

The book uses the raw form as its default. The choice was made before any recording was read: both models in the comparison are measured on it, so the comparison is between models, not between prompts. The chat form was recorded afterwards, as an observation. On the 200 messages it got {{out:prompting-on-test-200:135m-chat correct}} right, and the raw form got {{out:prompting-on-test-200:135m-raw correct}}. A gap of two messages in 200 is not a difference this sample can show in either direction, so the chat form is not shown to help, and the raw form is not shown to hurt, and the default is not changed because of this count. The trie's names begin with a space, which follows `Team:` in the raw form but not the new line that opens the answer in the chat template. The chat count is therefore for this trie as well as for the template. The result is not evidence that raw prompts beat chat templates for small models in general, and one sample of 200 messages does not settle that.

**Working rule:** Use one prompt for every model you compare, in the form the comparison needs. Before you adopt a chat template, measure it on the same sample as the raw form, and report a small gap as a small gap.

## Why free text fails {#why-free-text-fails}

Chapter 1 asked the raw model for one team with no constraint, and the model decoded freely: greedy, taking the highest-scoring token at each step, as in this chapter, but with no mask on the tokens it could pick. Its example counted the answers. {{out:unconstrained-first-try:answers that are a team name}} were team names, so the router had nothing to send on.

Nothing in the model's own decoding stops an answer from being a sentence, a team that is not on the list, or a near-miss spelling of one. At each step the decoder may pick any token in the vocabulary, so the answer can go anywhere. Chapter 1's stand-in did not show this failure, because its recorded answers came from a grammar that allowed only the team names. A model that runs in the router needs the same guarantee from the decoder itself.

The tempting repair is to take whatever the model wrote and match it to the nearest team name. That repair is a guess. A near-miss spelling may match the right team, or a wrong team that is close in letters, and the router would then send the message to a team the model never named. Chapter 1 said what to do instead: an answer that is not a team counts as an error, is counted on the sample, and the message goes to a person. A constraint removes most of those answers before they are written. The count is still kept, so that a constraint which fails to hold shows up as a number and not as a quiet change in routing.

**Working rule:** Treat an answer that is not a team as a failed route. Count such answers on a fixed sample before a model routes anything, and do not match a near-miss to the closest team. Constrain the answer at decode time, and keep the count, so that a broken constraint shows up as a number.

## A token trie allows only team names {#a-token-trie-allows-only-team-names}

A model writes tokens, not words, and a team name is a short run of them. `build_trie` in `router_lib`, the book's shared module, stores the 77 names as a trie: a tree whose paths spell out each team's tokens, with the team's index at the node where its path ends. The function asserts that no two teams end at the same node, so each path names one team.

The decoder walks this tree as it writes. At the start, the allowed tokens are the first token of each team name. After each token, the allowed set narrows to the children of the node the decoder has reached. Where a path has ended on a team, the end-of-answer token joins the allowed set, so the answer can stop there. No one of the 77 names is the start of another, so every path ends at a leaf. The code would still handle a name that was: at that node the decoder could stop, or continue along the longer name. A walk of this kind cannot leave the list, because every token it may choose leads toward a team.

Greedy decoding then takes the highest-scoring of the allowed tokens at each step. Transformers applies the rule through a callback, `prefix_allowed_tokens_fn`, which returns the allowed tokens for each position. The processor starts from a mask of minus infinity for every token, sets the allowed tokens' entries to zero, and adds the mask to the scores with `scores_processed = scores + mask`. Every token that is not allowed keeps minus infinity, so greedy decoding cannot pick it ([transformers {{version:transformers}}, `src/transformers/generation/logits_process.py`](https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/generation/logits_process.py)).

Chapter 1's stand-in got the same guarantee from its library. `recordings/common.py` builds a grammar, `root ::= "card_arrival" | ...`, that lists the 77 names as literal strings, and the recordings applied it through the chat template. llama-cpp-python adds a grammar to its sampler before the greedy step. In `llama_cpp/llama.py` the line `sampler.add_grammar(self._model, grammar)` runs under `if grammar is not None:`, and `sampler.add_greedy()` runs after it, under `elif temp == 0.0:` ([llama-cpp-python v0.3.36, `llama_cpp/llama.py`](https://raw.githubusercontent.com/abetlen/llama-cpp-python/v0.3.36/llama_cpp/llama.py)). So the stand-in's answers were limited to the names by the library, where this chapter's callback does the same job in the decoder.

Two details of that code shape the callback. The processor calls it as `self._prefix_allowed_tokens_fn(batch_id, sent)`, so the batch index comes first. And an empty list is an error: the processor raises `ValueError` when the callback returns no tokens. The book's callback therefore never returns an empty list. When no path can continue, it returns the end-of-answer token, and the answer stops there. That token is the model's `<|im_end|>`, which is also its pad token (id 2). The decoder stops after 32 new tokens (`max_new_tokens=32`). Every one of the 600 recorded answers ended on a team. A node with no children always holds a team, so the empty-list case is a guard and not a case the 77 names produce.

[Example: constrained-decode-135m](../examples/constrained-decode-135m.py)

The example prints {{out:constrained-decode-135m:answers that are a team}} answers that are team names, for {{out:constrained-decode-135m:messages}} messages. Every answer is a team, which is the guarantee the trie gives. The accuracy on the same messages is {{out:constrained-decode-135m:accuracy}}. The [measurement section](#measure-the-constrained-answers) takes that second number apart.

The router does not read the team out of the text. `constrained_team` returns the index stored at the node where the walk ended, and the text is decoded only so that a reader can see it. The index is fixed by the trie and not by a parser that runs after the model. That is the reason the constraint is worth the code: a parser that matches text to teams can fail in ways the trie cannot. These examples return the team only so that it can be counted. Nothing in this chapter routes a message. A route returns a team only through `handoff`, which sends an uncertain answer to a person, and the examples here do not call it.

**Working rule:** Constrain the answer to the labels with a token trie built from the label names, and count the answers that are labels on a fixed sample. The constraint makes every answer a team. It does not make any answer right, so count the right answers as a separate measure.

## Decode greedily so every count repeats {#ties-and-determinism}

Every example here uses greedy decoding: the highest score wins, and nothing is sampled. That matters for the counts. A sampled answer would differ from one run to the next, and then no printed count could be checked against a recording. The examples also fix their seeds in `router_lib.setup_runtime`, use float32 and set two threads, so on one machine a run repeats its answers. Each example calls `torch.set_num_threads(2)` itself, at the top, so that it does not depend on how its runner was started. Each example's printed output is compared with its recorded output on every run, and a changed line fails the check.

Repeatable on one machine is not the same as repeatable everywhere. The PyTorch notes on reproducibility say: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms" ([PyTorch {{version:torch}} notes on randomness](https://docs.pytorch.org/docs/2.15/notes/randomness.html)). So the book claims repeatability for the versions pinned in `examples/requirements.txt`, on the kind of machine the book ran on. A reader on another platform should expect to check the recorded lines again. If a recorded line fails there, read the changed line before touching the example. A different count of right answers can come from a different machine, a different library version or a changed prompt, and each of those needs a different fix.

Ties need a fixed rule as well. Label scoring in [the last section](#why-the-prior-wins) breaks a tie in score by the lower team index, through its sort key `(score, -index)`, and the most-given-team count breaks a tie in count by team name. Constrained decoding has no team scores. Its greedy step takes the first maximum among the allowed tokens, so a tie there goes to the lower token id.

**Working rule:** Use greedy decoding, fix the seeds and the thread count, and compare each printed line with a recorded copy on every run. Do not describe a result as reproducible beyond the versions and the kind of machine it was run on.

## Measure the constrained answers {#measure-the-constrained-answers}

The [live examples](#a-token-trie-allows-only-team-names) show the mechanism on six messages. The 135M model with the raw prompt got {{out:constrained-decode-135m:accuracy}} right, and the 360M model got {{out:constrained-decode-360m:accuracy}}. The 135M model's most common answer was {{out:constrained-decode-135m:most given team}}. The chat form on the same six messages got {{out:constrained-decode-135m-chat:accuracy}}. Six messages cannot rank these models: one right answer is the whole difference between 1/6 and 0/6.

[Example: constrained-decode-360m](../examples/constrained-decode-360m.py)

For the comparison, each variant answered 200 messages. The recording's counts are in the table, with the same messages in every row:

| Variant on test_200 | Answers outside the team list | Correct | Distinct teams given |
| --- | --- | --- | --- |
| 135M, raw prompt | {{out:prompting-on-test-200:135m-raw answers outside the team list}} | {{out:prompting-on-test-200:135m-raw correct}} | {{out:prompting-on-test-200:135m-raw distinct teams given}} |
| 135M, chat template | {{out:prompting-on-test-200:135m-chat answers outside the team list}} | {{out:prompting-on-test-200:135m-chat correct}} | {{out:prompting-on-test-200:135m-chat distinct teams given}} |
| 360M, raw prompt | {{out:prompting-on-test-200:360m-raw answers outside the team list}} | {{out:prompting-on-test-200:360m-raw correct}} | {{out:prompting-on-test-200:360m-raw distinct teams given}} |

[Example: prompting-on-test-200](../examples/prompting-on-test-200.py)

The constraint does what it promises: no answer in any row is outside the team list. The 360M raw row has the most right answers, at {{out:prompting-on-test-200:360m-raw correct}}, and it gives {{out:prompting-on-test-200:360m-raw distinct teams given}} distinct teams where the 135M raw row gives {{out:prompting-on-test-200:135m-raw distinct teams given}}. Distinct teams alone mislead. The 135M raw row gives most of its answers to one team, {{out:prompting-on-test-200:135m-raw most given team}}, while the 360M raw row's most given team gets {{out:prompting-on-test-200:360m-raw most given team}}. No row is close to a router, and no row beats another by more than a few messages.

The outside-the-list line prints a count with no denominator, so the reader should take the 200 from the sample it describes. Every row answered the same 200 messages.

To read a row, start with the first column. If it is not zero, the constraint has not held, and the other two columns mean nothing until that is fixed. The second column is the one that matters for routing: it counts the messages whose team is the gold label, out of the whole sample, with the wrong answers counted as wrong. The third column says how many teams the variant uses at all. A variant that gives only a few teams is either guessing a few popular teams or has learned something narrow, and the count does not say which. A large count does not rule out concentration either: the 135M raw row uses {{out:prompting-on-test-200:135m-raw distinct teams given}} teams and gives one of them most of its answers. A reader who runs the same three counts on a sample of their own messages gets a comparison on their own data, with the same meaning in each column.

The 360M model is the larger of the two, and [chapter 3](choose-a-base-model) counted how much larger it is. The table shows what that cost buys on this prompt and this constraint: {{out:prompting-on-test-200:360m-raw correct}} right, against {{out:prompting-on-test-200:135m-raw correct}} for the 135M model on the same raw form. That is a few more messages for a larger model, and the chapter does not count it as a reason to pay for the extra size.

The book does not compute an interval for these counts. A gap of a few messages out of 200 could come from the sample, so the table supports one statement: the three variants are close, and none of them is a router. A reader who wants to choose between two prompts should run both on a sample of their own messages and count the same three things.

The trie checks the shape of the answer and nothing else. It does not read the message beyond the prompt, and it does not know which team is right. Any team that is on the list is allowed at every step where its name could still be written, so the decoder's choice among them is the model's choice, and the mask only decides which teams can be written at all.

Here is one valid wrong answer, described by its teams only, since messages are never quoted in this book. One test message whose gold team is `card_not_working` got `card_arrival` from the 135M raw row, a team on the list, so the constraint let it through and the count of answers outside the list stays at zero.

The lesson the outputs support is that constrained decoding makes every answer a team, but a valid answer is not a right one. The constraint only removes answers that are not teams. It does not choose a better team, and the prompt change did not either. Chapter 5 is the next step: it trains a head on the labelled messages and measures whether the trained model routes better on the same sample.

In production the failure does not show up as an error. The count of answers that are not teams stays at zero, so a check that watches only that count stays green. Only a count of right answers on a fixed, labelled sample shows the problem.

The cost falls on the people who receive the message. A valid wrong team sends the message to the wrong destination, where support staff see a request that is not theirs, and they must find the right team and move it, which is the same work a hand-off creates, with no sign on the message that it was misrouted. A router that cannot say when it is unsure gives those people nothing to act on. Constrained decoding also returns no score for the other 76 teams, so a hand-off rule built on it would have no per-team scores to compare.

**Working rule:** Report the count of right answers beside the count of valid answers, on one fixed sample, for every variant you compare. A router whose answers are all valid can still be wrong on most messages, and only the second count shows it.

## Why a prior is not a router {#why-the-prior-wins}

Constrained decoding returns one answer and no scores. Label scoring gives every team a score. For each team, it sums the log-probability of the team's tokens after the prompt, where a log-probability is the logarithm of the probability the model gives a token. The highest sum is the model's top team for the message.

[Example: label-scoring-prior](../examples/label-scoring-prior.py)

The example scores all 77 teams for {{out:label-scoring-prior:messages}} messages. The top team is the same for every one of them: the example prints {{out:label-scoring-prior:distinct top teams}} distinct top team, {{out:label-scoring-prior:most frequent top team name}}, which is top for {{out:label-scoring-prior:most frequent top team}}. That is the model's prior, the team it favours before any training. Three messages are too few to say more about it.

If the top team is the same for every message, the score is the model's prior and says little about any one message. The 200-message recording shows the same prior in constrained decoding: the 135M raw row gives most of its answers to one team, {{out:prompting-on-test-200:135m-raw most given team}}, and that team is the gold team of {{out:prompting-on-test-200:135m-raw gold for most given team}} messages in the sample.

The sum favours short names. Each token of a team's name adds a log-probability, which is never above zero, so a team with more tokens has more terms to pull its total down. The example does not divide by length, and the chapter does not claim to correct for it. A reader who uses label scoring should check the top teams against a sample with gold labels before trusting them.

Label scoring does give a score for each team, which is what a hand-off rule needs. Each message is scored against all 77 team names, which the code scores in one batch after the prompt runs once. Chapter 5 gets a score for each team from a single forward pass through a head that is trained on labelled messages. Label scoring is the bridge from this chapter to that one: it shows that the model already has per-team scores, and that before training they say little about the message.

**Working rule:** Use label scoring to see which team a model favours before training, and do not route with it. Do not route messages with a prompted model of this size on this task; the 200-message recording is the evidence. Train a head on the labelled messages instead, as [chapter 5](fine-tune-with-lora) does.

## What this chapter decided {#what-this-chapter-decided}

The rules in this chapter are scoped to these two models, this prompt and this constraint, on the 200 messages of the recording. A larger model, a different prompt or a longer sample could move the counts, and the book has not tested any of them on that sample. What the chapter does establish is narrower: with the raw prompt and the trie, a model of this size returns valid teams and few right ones, and the count of right ones is the number to watch.

Each decision is tested in its own section: the prompt in the first, the trie in [its section](#a-token-trie-allows-only-team-names) and greedy decoding in [its section](#ties-and-determinism), and the two counts in the measurement section. A reader who changes any of them must record again before comparing, as the first section says.

This chapter made these decisions:

- The router prompt is the raw form, and both models in the comparison use it.
- Constrained decoding uses a token trie over the 77 team names, with greedy decoding and fixed seeds.
- Every variant is reported with its count of valid answers and its count of right answers, on one fixed sample.
- The live examples show the mechanism on six messages (three for label scoring), and the comparison uses the recorded 200.
