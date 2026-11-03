"""Lab 04 - build a synthetic SMS-style dataset: "phish" (phishing-style) vs "legit".

Run from the lab folder:  python data/make_data.py       ->  data/messages.csv

Why synthetic?  The classic UCI SMS Spam Collection needs a download and is SPAM, not phishing
(never call SMS spam labels "phishing"). Here the instructor generates a clearly-labelled,
fictional dataset (made-up brands, 20 % phishing).

Messages are COMPOSED from fragments (opener + topic + requested action + urgency + link).
Every fragment can appear in BOTH classes, only with different probabilities - like real life:
  * legit messages also talk about accounts, codes, packages, links and sometimes sound urgent
  * phishing messages are sometimes polite, short, or link-free
so a perfect score is impossible and the errors are worth analysing.
If you have the real UCI file, adapt load_data() in src/pipeline.py (the lab README explains how).
"""
import csv
import os
import random
import string

SEED = 42
N_PHISH, N_LEGIT = 400, 1600

BANKS = ["Northbank", "Delta Credit", "Atlas Pay", "Civic Savings"]
CARRIERS = ["SwiftPost", "ParcelGo", "BlueFreight"]
SERVICES = ["StreamBox", "CloudDrive", "MusicNest"]
TIMES = ["6pm", "7:30", "noon", "8am", "tomorrow at 9", "Friday at 5"]
MEALS = ["dinner", "lunch", "coffee", "the match"]

TOPICS = ["account", "package", "code", "payment", "subscription", "prize", "appointment", "order", "chat"]
WEIGHTS = {
    "phish": [0.28, 0.20, 0.08, 0.14, 0.10, 0.12, 0.01, 0.02, 0.05],
    "legit": [0.12, 0.14, 0.10, 0.10, 0.06, 0.01, 0.15, 0.14, 0.18],
}
ACTION_KINDS = ["credential", "payment", "app", "none"]
ACTION_W = {"phish": [0.50, 0.30, 0.05, 0.15], "legit": [0.08, 0.10, 0.40, 0.42]}
ACTIONS = {
    "credential": ["Verify your identity.", "Confirm your details.", "Reply with your PIN.", "Log in to confirm."],
    "payment": ["Pay the fee now.", "Update your payment info.", "Claim it today."],
    "app": ["Open the official app to view it.", "You can manage this in the app settings.",
            "Details are in your online account."],
    "none": [""],
}
URGENCY = ["Act now.", "Within 24 hours or your access will be suspended.", "This is your final notice.",
           "Immediately.", "Please reply as soon as you can."]
OPENERS = {"phish": ["Dear customer,", "ALERT:", "Notice:", "Hello,", "Hi,"],
           "legit": ["Hi,", "Hey,", "Hello,", "Reminder:", "FYI:"]}
P_OPENER = {"phish": 0.35, "legit": 0.25}
P_URGENT = {"phish": 0.65, "legit": 0.12}
P_LINK = {"phish": 0.70, "legit": 0.30}
P_BAD_URL = {"phish": 0.75, "legit": 0.30}


def rid(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def bad_url(rng):
    return rng.choice([f"http://bit.ly/{rid(rng)}", f"http://secure-{rid(rng, 4)}-verify.com/{rid(rng)}",
                       f"https://{rid(rng, 5)}-login.info/a{rid(rng, 3)}", f"http://tinyurl.com/{rid(rng, 7)}"])


def good_url(rng):
    return rng.choice([f"https://docs.example.com/d/{rid(rng)}", f"https://meet.example.org/{rid(rng, 8)}",
                       f"https://www.example.com/orders/{rng.randint(1000, 9999)}"])


def topic_sentence(rng, topic):
    b, c, s = rng.choice(BANKS), rng.choice(CARRIERS), rng.choice(SERVICES)
    n, amt, prize = rng.randint(100000, 999999), f"{rng.randint(2, 90)}.{rng.randint(10, 99)}", rng.choice([100, 250, 500])
    pool = {
        "account": [f"Your {b} account has a new notice.", f"We noticed activity on your {b} account.",
                    f"{b}: there is a message about your account."],
        "package": [f"{c}: your package is waiting.", f"Your parcel from {c} has an update.",
                    f"{c} could not deliver your package today."],
        "code": [f"Your verification code is {n}.", f"Use code {n} to continue.", f"Security code {n}."],
        "payment": [f"A payment of ${amt} is pending.", f"We received a payment of ${amt}.",
                    f"Your invoice of ${amt} is ready."],
        "subscription": [f"Your {s} subscription renews soon.", f"Your {s} trial ends in {rng.randint(2, 7)} days."],
        "prize": [f"You have been selected for a ${prize} reward.", f"Good news: a ${prize} gift card is reserved for you."],
        "appointment": [f"Your appointment is {rng.choice(TIMES)}.", f"Meeting moved to {rng.choice(TIMES)}.",
                        f"Reminder: class starts {rng.choice(TIMES)}."],
        "order": [f"Thanks for your order #{rng.randint(10000, 99999)}.", f"Your order #{rng.randint(10000, 99999)} has shipped."],
        "chat": [f"Are we still on for {rng.choice(MEALS)} at {rng.choice(TIMES)}?", "Can you call me later?",
                 f"Running late, see you {rng.choice(TIMES)}."],
    }
    return rng.choice(pool[topic])


def compose(rng, label):
    parts = []
    if rng.random() < P_OPENER[label]:
        parts.append(rng.choice(OPENERS[label]))
    parts.append(topic_sentence(rng, rng.choices(TOPICS, weights=WEIGHTS[label])[0]))
    action = rng.choice(ACTIONS[rng.choices(ACTION_KINDS, weights=ACTION_W[label])[0]])
    if action:
        parts.append(action)
    if rng.random() < P_URGENT[label]:
        parts.append(rng.choice(URGENCY))
    if rng.random() < P_LINK[label]:
        parts.append(bad_url(rng) if rng.random() < P_BAD_URL[label] else good_url(rng))
    return " ".join(parts)


def noisy(rng, text):
    """Light realistic noise: sign-off, random lower-casing, a dropped letter."""
    if rng.random() < 0.10:
        text += rng.choice([" Thanks", " Regards", " :)"])
    if rng.random() < 0.20:
        text = text.lower()
    if rng.random() < 0.10 and len(text) > 20:
        i = rng.randrange(5, len(text) - 5)
        text = text[:i] + text[i + 1:]
    return text


def main():
    rng = random.Random(SEED)
    rows = [(noisy(rng, compose(rng, "phish")), "phish") for _ in range(N_PHISH)]
    rows += [(noisy(rng, compose(rng, "legit")), "legit") for _ in range(N_LEGIT)]
    rng.shuffle(rows)
    os.makedirs("data", exist_ok=True)
    with open("data/messages.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"wrote data/messages.csv: {N_PHISH} phish + {N_LEGIT} legit = {len(rows)} messages")


if __name__ == "__main__":
    main()
