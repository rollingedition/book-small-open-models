"""The two monthly bills against messages a month, from zero to three times the break-even: the hosted bill grows with volume and the machine's stays flat, so they cross at the break-even. The break-even is asserted equal to break-even-from-recorded-ratio's line; no model runs and the prices are the example values."""

import math
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402

torch.set_num_threads(2)

import router_lib  # noqa: E402

router_lib.setup_runtime()
inputs = router_lib.bill_inputs_recorded()
messages = [router_lib.read_rows("test")[i][0] for i in router_lib.read_sample("test_200")]

# The same computation as break-even-from-recorded-ratio.
input_tokens = router_lib.router_prompt_tokens(messages)
per_message = (input_tokens * inputs["price_per_million_input_tokens"]
               + inputs["output_tokens_per_message"] * inputs["price_per_million_output_tokens"]) / 1_000_000
machine_month = inputs["machine_cost_per_hour"] * inputs["hours_per_month"]
capacity = inputs["router_messages_per_second"] * 3600 * inputs["hours_per_month"] * inputs["utilisation"]
breakeven = math.ceil(machine_month / per_message)
assert breakeven == 477906, "not break-even-from-recorded-ratio"

# The span runs from zero to three times the break-even.
span_end = 3 * breakeven
volumes = [span_end * k / 200 for k in range(201)]

print(f"example machine bill per month (example cost per hour), USD: {machine_month:.2f}")
print(f"example break-even messages per month (example prices and cost per hour): {breakeven}")
print(f"example hosted bill at the break-even (example prices), USD: {breakeven * per_message:.2f}")
print(f"span end, messages a month: {span_end}")
print(f"example hosted bill at the span end (example prices), USD: {span_end * per_message:.2f}")
print(f"span within machine capacity: {'yes' if span_end <= capacity else 'no'}")

from lib.figures import ACCENT, INK, MUTED, new_figure, save  # noqa: E402

fig, ax = new_figure()
ax.plot([v / 1000 for v in volumes], [v * per_message for v in volumes], color=ACCENT, label="hosted model bill (example prices), USD")
ax.plot([0, span_end / 1000], [machine_month, machine_month], color=INK, linestyle="--",
        label="machine bill (example cost per hour), USD")
ax.plot([breakeven / 1000], [machine_month], "o", color=MUTED)
ax.annotate(f"break-even: {breakeven:,} messages a month", (breakeven / 1000, machine_month),
            xytext=(12, -36), textcoords="offset points", color=INK)
ax.set_xlabel("messages a month, thousands")
ax.set_ylabel("monthly bill, USD")
ax.set_xlim(0, span_end / 1000)
ax.set_ylim(0, span_end * per_message * 1.1)
ax.legend(loc="upper left")
save(fig, "break-even-bills")
