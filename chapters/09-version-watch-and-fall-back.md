---
chapter_id: version-watch-and-fall-back
---

# Version the router, watch its hand-offs and fall back to people

By the end of this chapter, you can name a router by the parts that make its answers, check those parts before the route loads them, count the hand-off rate for each window of messages against a stated baseline, and say where each handed-off message goes.

[Chapter 8](serve-the-router-on-a-cpu) put the router behind a route, and each answer carries a team or a flag that sends the message to a person. A route that runs for months raises three questions that the route cannot answer by itself. Which model is answering? Are its files the ones that were checked? Are the messages it hands off becoming more or fewer? This chapter gives each question a control and a printed count. The controls are the book's own: the rule for drift and the rule for retraining are choices, and the chapter labels them as choices.

This chapter covers what running your own model adds: versions, a fallback to people and retraining. Watching a service in general, with its dashboards, logs and alerts, is ground for a book on production monitoring. This chapter does not repeat it.

## Name the router by its parts {#name-the-router-by-its-parts}

A router is more than its model. It is the base model, the adapter trained on that base, the threshold that sends a message to a person, and the data and recipe that produced the adapter. Two routers that differ in any one of these can answer the same message differently. A log line that says only "the router" cannot tell them apart.

The adapter's files do not include the base model. The [PEFT checkpoint guide](https://raw.githubusercontent.com/huggingface/peft/v0.21.2/docs/source/developer_guides/checkpoint.md) puts it this way: "The `state_dict` only contains the parameters of the adapter module, not the base model." The adapter therefore has to name the base it was trained on, and the name has to be exact. This book names the base by its commit, the pin in `examples/models.toml`. The [Hugging Face download guide, at v2.4.0](https://huggingface.co/docs/huggingface_hub/v2.4.0/en/guides/download), says that, when a commit hash is used, "it must be the full-length hash instead of a 7-character commit hash." The same guide says that "By default, the latest version from the `main` branch is downloaded." A branch name therefore moves with the branch, and it is not a pin. Chapter 5 recorded the base revision in the adapter's manifest, next to the threshold, the data commit and the recipe hash.

The adapter's name carries its version. `banking77-lora-v1` is the recorded LoRA adapter, trained for one epoch on the training rows outside the dev split, version one. A new recording of the same recipe is `v2`, and it gets its own name and its own manifest. The manifest has no separate version field. The name is the version, and [chapter 8](serve-the-router-on-a-cpu#one-route-that-returns-the-hand-off)'s route reports the version as the suffix of the adapter's name. For `banking77-lora-v1` that suffix is `v1`. The int8 copy from [chapter 7](quantize-and-measure) is a different model, `banking77-lora-v1-int8`, and this route does not serve it. Its suffix would read `int8`, so a copy served by name needs a version field of its own.

The first check in this chapter reads the manifest and compares the base revision with the pin. Its output begins with the adapter's name, then whether the base revision in the manifest is the pinned one:

[Example: adapter-manifest-check](../examples/adapter-manifest-check.py)

The example prints {{out:adapter-manifest-check:adapter}}, and then whether the base revision matches the pin: {{out:adapter-manifest-check:base revision matches pin}}. The version and the base revision are the two parts a reader can check against the pins before any file is read.

**Working rule:** Name a router by its base revision, its adapter's name with the version in that name, its data commit, its threshold and its recipe hash, and write all five into the adapter's manifest before the route serves it. Compare the base revision in the manifest with the pin in `examples/models.toml` each time you load the adapter. A pin that has moved means a different base model, and the adapter needs a new check before it serves.

## Check the adapter's files against its manifest {#check-the-adapters-files-against-its-manifest}

A load is not a version check. The [PEFT checkpoint guide](https://raw.githubusercontent.com/huggingface/peft/v0.21.2/docs/source/developer_guides/checkpoint.md) says that "most of the entries are not necessary to load the model" in a config file, so a load can succeed with a config that someone has changed by hand. A successful load does not show that the adapter is the pinned one. The extra-key copy below loads, and the example's output records one warning from that load. A mismatched base does fail at load, with a state-dict error, in the research probe; this example does not print that case.

So the check compares the adapter with its manifest, in three parts. It computes the SHA-256 of `manifest.json` itself, and compares it with the digest pinned for the adapter's name in `examples/adapter-pins.json`. The pin covers every field the manifest holds, including the threshold, the base revision and the output digests. It computes the SHA-256 of each output file the manifest lists, and compares it with the digest the manifest records. It also reads the three LoRA settings in the config, the rank, the alpha that sets the scaling and the target layers, and compares them with the manifest. The example prints how many of each agree, and whether the route may load the folder. For the recorded adapter, the manifest and every setting and file agree:

the manifest matches its pin: {{out:adapter-manifest-check:manifest matches pin}}, {{out:adapter-manifest-check:settings matching}} settings and {{out:adapter-manifest-check:files matching}} files agree, and the route may load: {{out:adapter-manifest-check:route may load}}.

The example also checks two edited copies of the adapter, and each line reports what the check finds. The first copy has its `handoff_threshold` changed to 0 in the manifest. Its line reads: {{out:adapter-manifest-check:threshold copy}}. Its settings and files still agree, because the threshold is not a setting and no output file changed. Only the manifest's own digest sees the change, and the copy is refused. The second copy has one extra key added to its config. Its line reads: {{out:adapter-manifest-check:extra-key copy}}. Its settings still agree, and the file check refuses it, because the config file's hash no longer matches. The example then loads that copy with PEFT and keeps only the warnings that name the extra key: {{out:adapter-manifest-check:warnings naming the extra key while PEFT loads the copy}}. It reads: {{out:adapter-manifest-check:that warning}} The load returns a model with the key ignored, and that warning is all PEFT says about it. A warning is not a refusal. The check is what refuses the copy.

The route in [chapter 8](serve-the-router-on-a-cpu#load-the-model-once-at-startup) runs this check. Its `make_app` calls `check_adapter` before it loads the model, and it also compares the manifest's base revision with the pin. It raises an error on any mismatch, so a copy that fails is refused before the model is in memory. The example shows this on both copies: {{out:adapter-manifest-check:threshold copy refused by make_app}} for the threshold copy and {{out:adapter-manifest-check:extra-key copy refused by make_app}} for the extra-key copy. The check trusts the repository. Its pinned digest sits in `examples/adapter-pins.json`, next to the code, so a reader who edits the folder and not the pins file is refused, and a reader who edits both is not. The pins file is therefore reviewed like any other code.

**Working rule:** Run the check before any route loads an adapter. Pin the SHA-256 of the manifest with the code, count the matching settings and files, print each count with its total, and refuse the load when any count falls short or the manifest digest does not match. Take the hashes from the files on disk, and check and load from the same files, with nothing writing to them in between.

## Watch the hand-off rate {#watch-the-hand-off-rate}

The hand-off rate is the number of messages below the threshold divided by all the messages in a window. [Chapter 2](measure-the-routing-task#three-measures-each-with-its-denominator) defined it, and this chapter watches it over time. A watch needs three things: a window, a baseline and a rule for when a window has moved too far from the baseline.

The baseline is the hand-off rate on the dev split at the adapter's threshold. The example computes it with `dev_handed_off`, which routes all the dev messages once and counts the hand-offs. It prints {{out:handoff-windows:dev hand-off rate (baseline)}}, and the same count appears in the output of the threshold example in [chapter 5](fine-tune-with-lora#choose-the-threshold-on-dev): {{out:choose-threshold-on-dev:threshold 0.63 handed off}}. The two agree, so the baseline is the one that chapter chose the threshold on.

A baseline from dev is not a forecast for the inbox. The dev split is drawn from training rows, team by team. The test sample is drawn from the test split. Neither is a sample of the inbox's traffic, and the mix of teams in live messages is not known. A gap between a window and the baseline is therefore a question to ask, and it does not answer the question itself.

The drift rule is this book's choice, not a figure from a source. A window drifts when its hand-off rate is more than a stated number of points from the baseline, in either direction. The example prints the rule as {{out:handoff-windows:drift rule}}. No source gives this number, and whoever runs the watch should choose its own and write it down before reading the windows.

`test_1001`, the committed test sample, has no dates, so the example makes its windows from its first 100 messages. The first two windows, of {{out:handoff-windows:week 1 messages}} messages each, are the first messages of the sample, in sample order. Weeks 3 and 4 are a replay, not new traffic: the example takes the next messages of the sample, sorts them by the router's confidence from lowest to highest, and splits the sorted list into two windows. The example labels the replay in its output: {{out:handoff-windows:weeks 3 and 4}}. The replay is built to produce a spread of confidences, so it shows what the rule does with a skewed window. It is not evidence of what the inbox did.

[Example: handoff-windows](../examples/handoff-windows.py)

The windows are small because the sample is small: four windows of {{out:handoff-windows:week 1 messages}} messages each. In a window this size, one message is 4 points of the rate, so one message can move a window across the rule's line. A watch on a real inbox needs windows large enough that one message does not decide the flag. The window size is a choice like the drift rule, and it belongs in the report, stated before the windows are read.

The output gives each window's count, its rate and its drift flag:

- Week 1 hands off {{out:handoff-windows:week 1 handed off}}, and its drift flag is {{out:handoff-windows:week 1 drift}}.
- Week 2 hands off {{out:handoff-windows:week 2 handed off}}, and its drift flag is {{out:handoff-windows:week 2 drift}}.
- Week 3 hands off {{out:handoff-windows:week 3 handed off}}, and its drift flag is {{out:handoff-windows:week 3 drift}}.
- Week 4 hands off {{out:handoff-windows:week 4 handed off}}, and its drift flag is {{out:handoff-windows:week 4 drift}}.

Week 4 is the case a reader should look at closely. A rate that falls to zero drifts under this rule, because the rule treats a fall as a change in the same way it treats a rise. The output explains why the rate is zero. The lowest confidence in week 4 is {{out:handoff-windows:week 4 lowest confidence}}, which is above the threshold, so no message in the window is handed off. The labels show that the router was right on every message it kept: {{out:handoff-windows:week 4 right among messages not handed off}}. The window is one of confident, correct answers, and the drift flag reports that its hand-off rate has moved, not that the router has made mistakes.

Week 3 is the other kind of window. Its lowest confidence is {{out:handoff-windows:week 3 lowest confidence}}, its hand-offs rise to {{out:handoff-windows:week 3 handed off}}, and the router is right on {{out:handoff-windows:week 3 right among messages not handed off}} of the messages it keeps. A rise in hand-offs means more messages reach a person. It does not, by itself, show which cause produced the rise: a change in the messages, a change in the teams that appear, or a change in the model. The route cannot say which, because its answers carry no record of the messages, and [chapter 8](serve-the-router-on-a-cpu#what-goes-wrong-in-production) says the same about the rate it returns.

**Working rule:** Count the hand-offs in each window of the same size, print each count with its denominator, and compare each rate with the dev baseline at the adapter's threshold. Set the drift rule, and its number of points, before you read the windows, and write it into your report. Treat a drifting window as a reason to read its messages and their teams, not as a verdict on the router.

## Sample handed-off messages for people {#sample-handed-off-messages-for-people}

The [hand-off watch](#watch-the-hand-off-rate) counts how many messages go to a person. It leaves two questions open: was the router right on the messages it handed off, and was it right on the messages it kept? This section describes how the first would be answered. It is a next step. This chapter draws no sample, reports no labels and gives no number for it.

A person would label a sample of the handed-off messages, drawn with a fixed seed, and the labels would say which team the person chooses for each. Set against the team the router's model would have chosen, they would show how often the two differ on the messages the router was unsure of. A router that is unsure about a message is the one whose choices people should check most, so this sample is the one to draw first. A reader who builds it can use the gold labels of a public dataset as a stand-in for a person's labels, and should say so in the report.

A sample of handed-off messages does not say how the router does on the messages it keeps. The keeps are the messages whose answers the route returned, and the hand-off never sends them to a person. Their accuracy can be estimated only when people check some of them too, after the route has already answered them. The check on kept messages is a second sample, drawn in the same way, and it is not a by-product of the sample of handed-off messages. The measure is accuracy on the messages not handed off, with its count and its denominator, as [chapter 5](fine-tune-with-lora#choose-the-threshold-on-dev) set it.

**Working rule:** When you build the sample, draw it with a fixed seed, print how many messages were labelled, and report how often the router's team and the person's team agree, with its count. Label a sample of kept messages on the same schedule, and read the router's accuracy on kept messages only from that sample. Never read the kept rate off the handed-off sample.

## Fall back to a person {#fall-back-to-a-person}

The route returns a flag and no team for a handed-off message, as [chapter 8](serve-the-router-on-a-cpu#one-route-that-returns-the-hand-off) built it. Something must still happen to that message. The rule in this chapter is that a handed-off message always goes to a person. The recorded answer of the stand-in, Qwen3-4B-Instruct-2507, may travel with it as a suggestion, labelled as the stand-in's, and the person decides whether to use it. The stand-in's answers on `test_1001` are recorded in `standin-baseline` ([chapter 2](measure-the-routing-task#the-baseline-from-recorded-answers)). The router's team is never returned for a handed-off message.

The route's JSON has `team`, `confidence`, `handed_off`, `adapter` and `version`, and it has no field that says who answered. A fallback answer must therefore never go out in that shape. A reader who wants a suggestion in the response adds a field of their own, on purpose, and says in the response what the field is.

The book does not teach a call to a hosted model as the fallback, and it sets no threshold for the stand-in. Its answers come from a recording, and the example reads them from it. The example measures how often the suggestion is right, and it does not decide anything from that measure.

[Example: fallback-to-larger-model](../examples/fallback-to-larger-model.py)

The example routes the first two windows, which is {{out:fallback-to-larger-model:messages routed}} messages. The router hands off {{out:fallback-to-larger-model:handed off by the router}} of them. Each handed-off message gets one record, with no team. Where the stand-in has a recorded answer, the record carries it as a suggestion. The output reads {{out:fallback-to-larger-model:sent to a person}} to a person, {{out:fallback-to-larger-model:suggestions attached}} with a suggestion attached, and the count of teams returned for a handed-off message is {{out:fallback-to-larger-model:team returned for a handed-off message}}. The list of records stands in for the person's queue. The book does not build that queue, and a reader who builds one should keep each record with its message.

The suggestions are right on {{out:fallback-to-larger-model:suggestions right}} of the handed-off messages that carry one. That is fewer than half, and it is the reason the suggestion is shown to a person and never used as the answer. The example does not measure how the stand-in does on messages the router keeps. In this sample, every handed-off message has a suggestion, and the chapter does not extend that to the inbox.

**Working rule:** Send each handed-off message to a person. A recorded answer from the stand-in may come with it, labelled as a suggestion. Never return a team for it, and never put a suggestion in the route's response shape without a field that says who answered. Count the suggestions that are right, with the count and its denominator, before a reader decides how much weight to give them.

## Retrain on a trigger {#retrain-on-a-trigger}

A drifting window is a signal to look, and it does not on its own start a retraining. The book sets three triggers, and this section states each one. The first is drift: the rule in this chapter's watch, applied to consecutive windows. The book's choice is to retrain when two consecutive windows drift. No source gives that number, and you may choose another, provided you state the choice in advance. The second trigger is a team added to the 77 the router returns. The router has no training rows for it, and people's labels would show it as a team outside the 77. The third is a rise in wrong teams on messages the router keeps, confirmed in the kept-message sample from the previous section. It is not the sample of handed-off messages, whose teams are never returned, so wrong teams there are expected and do no harm. The example in this chapter exercises only the first. The recorded data holds no added team and no person's labels, so the second and third are rules without a test here.

[Example: retrain-trigger](../examples/retrain-trigger.py)

The example applies the drift rule to the same four windows as the watch, and it reads the windows in order. The output prints each window's flag, the count of drifting windows and the longest run of them, then the decision: {{out:retrain-trigger:weeks drifting}} windows drift, the longest run is {{out:retrain-trigger:longest run of drifting weeks}}, and the trigger is {{out:retrain-trigger:retrain trigger}}. Weeks 3 and 4 drive the result, and they are the replay from the watch. The trigger fires because of how the replay was built, so the example shows that the rule works as stated, and it says nothing about what the inbox did.

When a retraining does start, it follows the same path as the first adapter. It is a recording in `recordings/`, as `banking77-lora-v1` was, written by a script there, and it writes a manifest with the same fields. It is named as the next version of the same recipe, so the retrained adapter is `v2`. Its threshold is chosen again on the dev split by the rule in [chapter 5](fine-tune-with-lora#choose-the-threshold-on-dev), never on the test sample. The retrained folder then needs its manifest's digest added to `examples/adapter-pins.json`, as a change to the repository that someone reviews. Until then, `make_app` refuses to build the app from it. The old adapter stays in place until `make_app` builds an app from the new folder.

**Working rule:** Retrain when the trigger you stated is met, and not on one drifting window. Write the trigger, its number and the rule beside the windows that met it. A retrained adapter is a recording named as the next version, with its manifest digest pinned and its threshold chosen again on dev. Keep the previous adapter until the new one passes the check in `make_app`.

## What goes wrong in production {#failures-in-production}

- **A manifest is edited by hand.** The manifest's digest is pinned, so the check refuses it. A check that trusts the repository can be defeated only by editing the pins file too, which is a change someone reviews.
- **A pin moves.** The base pin in `examples/models.toml` can change under an adapter whose manifest still names the old revision. `make_app` compares the two and refuses the build.
- **A fallback routes on the wrong scale.** A confidence from a model that was never calibrated to the threshold is not a gate. The book's fallback therefore never routes, and the suggestion travels only with a person's message.
- **The window is too small.** In a window of 25 messages, one message is 4 points. A flag then reflects one or two messages, and a reader who acts on it has acted on noise.
- **The rate moves and nothing explains it.** The route's answers carry no record of the messages, so the rate's movement is a question to answer from the messages and their teams, with the samples this chapter describes as next steps.

You notice the first kind of failure when the app is built. A refused folder raises an error at startup, so a deployment that builds the app when it starts sees a bad folder before any message fails. The other failures show in the counts. A moved hand-off rate shows in a window's count, and the drift rule reports it. A fallback that answers the wrong messages shows in the suggestion count, and a sample of people's labels shows the rest.

**Working rule:** Print each control's count where a reader can see it at startup and in each window, and keep the counts that would reveal a failure in the same report as the rate they explain.

## The decisions this chapter made {#the-decisions-this-chapter-made}

- A handed-off message always goes to a person. The stand-in's answer, when there is one, is a labelled suggestion, and the chapter counts how often it is right.
- The version check pins the SHA-256 of each adapter's `manifest.json` in `examples/adapter-pins.json`, and `make_app` refuses to build on any mismatch or a moved base pin.
- The drift rule is 10 points from the dev baseline, in either direction, on windows of 25 messages. Both are the book's choices.
- A retrain is due when two consecutive windows drift. The second and third triggers are stated rules, not tested here.

[Chapter 10](when-a-small-model-is-the-wrong-choice) uses the same measures as this chapter.

Each of these is a choice, not a finding. The fallback rule follows the rule from [chapter 1](route-a-message#what-the-router-returns) that a handed-off message never gets a guessed team. The pin rule trades a review of the repository for a check that a load alone cannot make. The drift rule, the window size and the two-window trigger are not sourced, and a reader who copies them should state them again in the report with the reason for each.

**Working rule:** Write each decision above into the report with its number and whether it is the book's choice or a sourced figure. Change a decision only by changing the rule it names, and rerun the examples that print its counts.

