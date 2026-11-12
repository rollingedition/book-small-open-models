---
chapter_id: measure-the-routing-task
---

# Measure the routing task before you choose a model

By the end of this chapter, you can say what each of the book's samples is for, compute the three measures the router is scored on with their denominators, and read a baseline that was recorded once and is reused for every comparison.

A bank's support inbox, which this book calls the inbox, sends each message to one of 77 teams. A hosted model picks the team. The book's router is a small open model that the bank would run itself. The question is whether it can match the hosted model on the messages it keeps, and say so when it is not sure of a message. Before any model is compared, the comparison needs fixed data, fixed samples and fixed measures. Otherwise two numbers that look alike may come from different messages, or be divided by different counts.

This book doesn't call a hosted model, so it uses a stand-in. The stand-in is a larger open model, Qwen3-4B-Instruct-2507, in the 4-bit Q4_K_M GGUF conversion that a third party published, at its pinned revision, with its answers limited to the 77 team names. It answers every test message once, in a recording, and the examples read those answers offline, so anyone can reproduce them. [Chapter 1](route-a-message#ask-a-model-for-one-team) introduced the stand-in. A reader's own hosted model will score differently, and [chapter 1](route-a-message#ask-a-model-for-one-team) shows how to record that model's answers the same way.

## The data and its licence {#the-data-and-its-licence}

The router is trained and measured on BANKING77, a public dataset of customer service queries in the banking domain (Casanueva et al., 2020). Each message carries one of 77 labels. The dataset calls its labels intents. This book calls them teams, because that is what the inbox routes to, and the word "team" is used in every chapter from here on.

The paper that introduces the dataset is Casanueva et al. (2020), "Efficient Intent Detection with Dual Sentence Encoders", in the Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pages 38–45 (Anthology ID 2020.nlp4convai-1.5; arXiv 2003.04807). Its abstract describes such a dataset as "a new challenging single-domain intent detection dataset comprising 13,083 annotated examples over 77 intents." The paper's text names the dataset: "a new and challenging intent detection dataset in the banking domain, dubbed BANKING 77". Its section 3 gives the count: "The new BANKING 77 dataset comprises 13,083 customer service queries labeled with 77 intents." The data card that ships beside the files gives the commit the files come from. The two files together hold the same 13,083 messages, which the next section counts.

The data card also states the facts the book relies on. It says that the files are a copy of the source's own train and test split, and that the split is not drawn again. It says that every team appears in both files: "Every label appears in both `train.csv` and `test.csv`." These two statements matter for the rest of the chapter. A router trained on the training file and scored on the test file is scored on the source's own held-out messages, not on a split that the book chose.

The licence is CC BY 4.0. The LICENSE file that sits beside the data begins "Attribution 4.0 International". Under that licence the data may be shared and adapted, provided the source is credited. The book keeps the licence file beside the data and carries the citation in the data card, so a reader who reuses the files has what the licence asks for.

The messages stay as published. No example prints message text, and no chapter quotes a message, so the content of the messages does not reach a page. The examples print counts and rates, and nothing else from the data.

**Working rule:** Keep the licence file beside the data, cite the paper in the data card, and print counts, never message text. If you reuse a dataset of real customer messages, read its card before you print anything from it, and cite the source where the data is documented.

## Dev, test and fixed samples {#dev-test-and-fixed-samples}

The [two files of the source split](#the-data-and-its-licence) are not enough by themselves. A router needs messages to tune on, messages to report on, and a small set that examples can run in seconds. Each one has to be fixed in advance, or a later result can be explained by a different sample rather than by a different model.

The dev split is carved from the training file, never from the test file. For each team, in the order of the categories file, the book draws about a tenth of that team's training messages with a fixed seed, and it commits the row numbers to `examples/data/banking77/samples.json`. The training messages that are not in dev are the ones the models train on. Because the draw is made team by team, each team's share of dev follows its share of the training file, and a team with few training messages keeps a small dev share instead of none.

The test split is the source's own held-out file. The book draws one stratified sample from it, {{out:count-the-splits:test_1001 per label min}} messages for each team, which gives {{out:confusions-in-the-baseline:messages}} messages in all. The sample is shuffled once, with the same generator as the dev draw, continuing after it, and the shuffled order is committed. Three prefixes of that order are named. The first 40 messages are `test_40`, the first 200 are `test_200`, and the whole sample is `test_1001`. A prefix of a shuffled sample is a random draw, so a short example does not read the teams in order, and `test_40` is not a sample of the first few teams in the file.

The examples print their own counts from the committed samples, and the book quotes only those. The example also checks that every dev row number is unique and falls inside the training file, so every dev message is a training row. It checks that `test_40` and `test_200` are the first 40 and first 200 of `test_1001`.

The next example runs no model. It reads the training file, the test file and the dev sample, and it counts each split.

[Example: count-the-splits](../examples/count-the-splits.py)

The output has one labelled line per count. The training file holds {{out:count-the-splits:train rows}} messages, and the dev sample holds {{out:count-the-splits:dev rows}} of them. The test file holds {{out:count-the-splits:test rows}} messages across {{out:count-the-splits:labels}} teams. The test file is even: the fewest messages for a team is {{out:count-the-splits:test per label min}} and the most is {{out:count-the-splits:test per label max}}. The training file is not even: its fewest is {{out:count-the-splits:train per label min}} and its most is {{out:count-the-splits:train per label max}}, which is why the dev draw is made team by team rather than as one random tenth of the file. The last two lines give the fewest and the most for the test sample, which are {{out:count-the-splits:test_1001 per label min}} and {{out:count-the-splits:test_1001 per label max}}, so the sample also holds the same number for every team.

The even test file is a property of the source, and the book relies on it. Equal counts per team mean that the test split does not reward a router for being good at the largest team. A router that does well on a sample with uneven counts can look better than it is, because most of its score comes from the teams with the most messages. The stratified test sample keeps that effect out of the measures.

[Chapter 5](fine-tune-with-lora#choose-the-threshold-on-dev) chooses the hand-off threshold on dev, and it records the choice in the adapter's manifest. The threshold is chosen on dev. The prompt, the model and the recipe are fixed before any test result is read, and none is changed because of one. The test split is never used to tune, and the test results are read once, to report them, on the fixed samples above.

In production the same rule holds for the inbox. Keep a dev set drawn from messages the inbox already has, and keep a test set that nobody reads while the router is being tuned. A test set that is read during tuning still looks like a test set, but it stops measuring anything new, and a router that is later measured on it will look better than it will be on the traffic that follows.

**Working rule:** Draw dev from training rows only, team by team, with a fixed seed, and commit its row numbers. Draw each test sample once, commit its order, and read it only to report. Choose every threshold, prompt and setting on dev. When dev is too small to decide, draw more dev messages from the training file, never from the test file.

## Three measures, each with its denominator {#three-measures-each-with-its-denominator}

The [samples](#dev-test-and-fixed-samples) say which messages to score. The measures say what to count. A router that returns a team, a confidence and a hand-off flag can be scored three ways, and each way answers a different question for the bank that runs it.

Accuracy asks how often the router's team is the right one. The hand-off rate asks how often a person has to read a message. Accuracy on the messages not handed off asks how often the router is right on the messages it keeps. Each question has its own denominator, and the book keeps the three apart in every output line. Confidence is the probability the router gives its team, and the threshold is the value below which a message is handed off, as [chapter 1](route-a-message#hand-off-the-unsure-messages) defines them.

| Measure | Counts | Divided by |
| --- | --- | --- |
| Accuracy | Messages whose team equals the gold team | All messages in the sample |
| Hand-off rate | Messages whose confidence is below the threshold | All messages in the sample |
| Accuracy on messages not handed off | Correct messages among those not handed off | Messages not handed off |

As formulas, with N messages in the sample: accuracy = correct / N; hand-off rate = handed off / N; accuracy on messages not handed off = correct among kept / kept, where kept = N minus handed off.

Two rules make these measures comparable across candidates. The first is that a handed-off message counts in the accuracy denominator, as wrong when its team is wrong. Accuracy alone does not reward a hand-off, because a handed-off message scores as correct only when its team is right. Accuracy on the messages not handed off, taken alone, hides the cost of the hand-off, because it leaves out exactly the messages that went to people. The hand-off rate shows that cost. The three numbers have to be read together, and a router that is better on one of them alone has not yet shown it is better.

The second rule is about printing. Every rate is rounded to one decimal place as a percentage, and it is printed beside its count, so a reader can check a rate by dividing by hand. A rate with no count beside it cannot be checked, and a rate that is printed without its denominator can be quoted in a way that hides its sample.

When the denominator of the third measure is zero, the measure is not defined. That happens when every message is handed off, so no message is kept. The book prints "not defined" in that case, as the example in [the baseline section](#the-baseline-from-recorded-answers) shows, and does not print zero, because zero would say that every kept message was wrong when there are no kept messages at all.

These measures score labels. The examples in this chapter, and the measurements in [chapter 1](route-a-message#ask-a-model-for-one-team) and [chapter 4](prompt-and-constrain), return teams only to be scored, and they return nothing to a caller. A router that returns a team to a caller does so through the hand-off rule that [chapter 1](route-a-message) introduces, so that a message below the threshold goes to a person rather than to a team.

In production, the hand-off rate is the measure that changes first. The mix of messages changes with the seasons, with a product launch, or with a fault that floods the inbox with one kind of question. A router whose accuracy holds can still hand off far more messages than the dev split predicted, and the only way to see it is to print the rate with its count on the traffic the router actually receives.

**Working rule:** Print each of the three measures with its count and its denominator, and round every rate to one decimal place. Report accuracy on the messages not handed off beside the hand-off rate, never alone. When no message is kept, print "not defined" rather than zero.

## The baseline from recorded answers {#the-baseline-from-recorded-answers}

A measure means little until something has been scored against it. The baseline is the stand-in's score on `test_1001`, the sample that the final comparison in [chapter 10](when-a-small-model-is-the-wrong-choice) uses. Later chapters score candidates on its prefixes, `test_40` and `test_200`, which are the first 40 and first 200 messages of the same committed order. Each prefix is scored from the same recording, so the baseline on it is the same answers for the same messages. Each candidate is compared with the same messages, counted in the same way, and reported beside the baseline.

The stand-in answered each of the messages in `test_1001` once. The answers are stored in a recording, `examples/data/recorded/standin-baseline/answers.jsonl`, and each answer names a team and gives its log-probability. The recording has a manifest beside it. The manifest names the script that made the recording, the model and its pinned revision, the hash of the prompt, the decoding settings and the seed, and the hash of the answers file. The examples check the answers file against that hash before they read a line, so a changed file fails the run rather than changing a score.

The recording is read, not run. Every example that uses the baseline reads the same answers, so the baseline cannot change from one run to the next, even if the runtime that made the recording is later updated. The reason for recording, rather than calling a model in each example, is the same reason the pins exist. A recording fixes the answers to one model at one revision, so a reader who runs the same example later sees the same numbers. The book never calls a hosted API, because a hosted model's answers are not fixed in the same way. The stand-in is an open model that anyone can run, and its recording is the only route to its answers that the book takes.

The baseline is also not a ceiling. It is the score a candidate has to beat on the same messages, and it says nothing about the hand-off trade-off by itself. A candidate that beats it on accuracy may still hand off more messages than the support staff can read, so the baseline is always read beside the three measures and never as a single number.

The example below scores the stand-in on `test_1001` with the three measures. For the stand-in, confidence is the probability that llama-cpp-python returns for its constrained answer, which is exp of the recorded log-probability. It uses chapter 1's placeholder threshold of {{out:route-recorded-answers:placeholder threshold}}, so its hand-off rate is not the book's final one, and [chapter 5](fine-tune-with-lora#choose-the-threshold-on-dev) chooses the real threshold on dev. It also applies a threshold above every confidence, which hands off every message.

[Example: baseline-measures](../examples/baseline-measures.py)

The example prints {{out:baseline-measures:messages}} messages. At the placeholder threshold the accuracy is {{out:baseline-measures:accuracy}}, the hand-off rate is {{out:baseline-measures:handed off}}, and the accuracy on messages not handed off is {{out:baseline-measures:accuracy on messages not handed off}}. At a threshold above every confidence, the hand-off rate is {{out:baseline-measures:handed off at that threshold}}, so the accuracy on messages not handed off is {{out:baseline-measures:accuracy on messages not handed off at that threshold}}.

A reader's own hosted model is a different baseline. It will score differently on the same messages, and the book does not claim that the stand-in's numbers are the inbox's numbers. The reader's own baseline is their model's recorded answers on `test_1001`, scored with the three measures in this chapter. The comparison that matters is between the reader's candidate and that baseline, on the same messages, with the same counts.

The [next section](#where-the-labels-collide) counts the stand-in's wrong answers and shows which teams they fall between. The wrong-answer count is the baseline's first number, and it is the one every later comparison starts from.

**Working rule:** Record the baseline once, on `test_1001`, the committed sample that every candidate's prefixes come from. Store the model's pinned revision, the prompt's hash and the decoding settings in a manifest, and check the answers file against its hash before you read a line. Compare each candidate only with the baseline scored on the same messages, whether that is `test_40`, `test_200` or `test_1001`. When your own hosted model is the baseline, record its answers the same way, and say in the write-up that its scores differ from the stand-in's.

## Where the labels collide {#where-the-labels-collide}

The [baseline's counts](#the-baseline-from-recorded-answers) say how often the stand-in is wrong. They do not say which teams it mixes up, and the mix-ups are what the support staff can act on. A pair of teams that the router confuses might need a clearer team description, a rule for that pair, or more messages for one of the two teams in training. None of those is known from the total count, so the book reads the pairs from the recorded answers rather than assuming which teams are close.

The next example reads the recording, checks it against the manifest, and checks that the recorded messages are exactly the `test_1001` sample. It then counts the wrong answers and the three gold-to-answer pairs that occur most often. Ties break on each team's position in the categories file, so the same pairs are printed in the same order on every run.

[Example: confusions-in-the-baseline](../examples/confusions-in-the-baseline.py)

The example prints {{out:confusions-in-the-baseline:messages}} messages, of which {{out:confusions-in-the-baseline:wrong answers}} were answered with the wrong team. The wrong rate is {{out:confusions-in-the-baseline:wrong rate}}. With no threshold applied, the error is one minus accuracy, so the wrong rate here and the accuracy in [the baseline section](#the-baseline-from-recorded-answers) add up to 100%.

The three most frequent pairs follow. The first pair is {{out:confusions-in-the-baseline:pair 1 count}} messages whose gold team is {{out:confusions-in-the-baseline:pair 1 gold}}, which the stand-in answered with {{out:confusions-in-the-baseline:pair 1 answer}}: {{out:confusions-in-the-baseline:pair 1 of gold team}} of that team's sample messages. The second pair is {{out:confusions-in-the-baseline:pair 2 count}} messages whose gold team is {{out:confusions-in-the-baseline:pair 2 gold}}, answered with {{out:confusions-in-the-baseline:pair 2 answer}}: {{out:confusions-in-the-baseline:pair 2 of gold team}}. The third pair is {{out:confusions-in-the-baseline:pair 3 count}} messages whose gold team is {{out:confusions-in-the-baseline:pair 3 gold}}, answered with {{out:confusions-in-the-baseline:pair 3 answer}}: {{out:confusions-in-the-baseline:pair 3 of gold team}}.

The first two pairs take all of their gold team's messages in the sample, and the third takes all but one. The team names suggest a reason for each pair: the first concerns a refund, the second a transfer, and the third a card's arrival and its delivery estimate. That is a reading of the names. The chapter does not read the messages, and no example prints them, so the pairs are a reason to look, not a finding about what the messages say.

The example also lists the teams the stand-in never answers. Across the 1,001 messages, its answers never name {{out:confusions-in-the-baseline:teams never answered}} of the 77 teams, so every message whose gold team is one of those is wrong. The example prints each team's sample count, and for each the team its wrong answers go to most often. For {{out:confusions-in-the-baseline:never answered 2 team}} that is {{out:confusions-in-the-baseline:never answered 2 most often as}}, which takes {{out:confusions-in-the-baseline:never answered 2 most often count}} of its messages, and for {{out:confusions-in-the-baseline:never answered 4 team}} it is {{out:confusions-in-the-baseline:never answered 4 most often as}}, which takes {{out:confusions-in-the-baseline:never answered 4 most often count}} of its messages. The recording does not show why these teams are never given, so the list is a reason to look here too.

The pairs describe the stand-in on this sample. They do not describe the teams, and they do not describe the inbox. A reader's own model will have its own pairs, and a reader should count them from their own recorded answers. When the question is how to fix a pair, the fix has to be chosen and checked on dev. A change that is chosen by reading the pairs on `test_1001` has been tuned on the test split, and its result on that split cannot be reported as a measure of the router.

In production, the same count is the first thing to repeat when hand-offs or complaints rise for one team. Count the pairs again on the inbox's own recorded answers, print each pair with its count, and compare the list with the one from the baseline. A change in the list says where the router's behaviour has moved, even when the overall rate has not.

The chapter's decisions, in short:

- Dev is drawn team by team from the training file, with a fixed seed, and its row numbers are committed.
- The test samples are fixed in order, and they are read only to report.
- Each measure is printed with its count and denominator, and "not defined" replaces a zero denominator.
- The stand-in's recorded answers on `test_1001` are the baseline, read from a manifest-checked file.

[Chapter 3](choose-a-base-model#measure-what-one-message-costs) counts the router prompt's tokens on `test_40` before it chooses a base model. [Chapter 5](fine-tune-with-lora) chooses the hand-off threshold on dev, with the rule stated there. [Chapter 10](when-a-small-model-is-the-wrong-choice) compares the router with the baseline on the same `test_1001` sample.

**Working rule:** Count the wrong answers and the most frequent pairs for every candidate, on the same sample, and print each pair with its count. Read the pairs as a list of where to look. Print the three most frequent pairs rather than only the first, because one pair invites over-reading, and a list shows whether the errors cluster or scatter. Check any fix on the dev split before the test split is read again.
