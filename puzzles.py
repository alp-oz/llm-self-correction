"""
puzzles.py

Registry of (prompt, planted-wrong-answer prefill, correct/wrong answer
patterns) used by batch_correction_test.py --puzzle <name>.

Each puzzle is a classic "obvious wrong answer feels right" trap, so a
prefill matching the WRONG intuitive answer is a plausible thing for a model
to have said, not an arbitrary corruption.
"""

PUZZLES = {
    "bat_and_ball": {
        "prompt": (
            "A bat and a ball cost one dollar and ten cents in total. The bat "
            "costs one dollar more than the ball. How much does the ball cost?"
        ),
        "prefill": (
            r"**Solution:** Let the ball cost \( b \) cents. Then "
            r"\( 2b + 100 = 110 \), so \( 2b = 10 \), so \( b = 10 \). "
            r"The ball costs 10 cents."
        ),
        "correct_answer": "5 cents,0.05 dollars,$0.05,b = 5,b=5",
        "wrong_answer": "10 cents,0.10 dollars,$0.10,b = 10,b=10",
    },
    "monty_hall": {
        "prompt": (
            "You're on a game show with three doors. Behind one door is a "
            "car; behind the other two are goats. You pick door 1. The host, "
            "who knows what's behind each door, opens door 3 and reveals a "
            "goat. He then offers you the choice to switch to door 2 or stay "
            "with door 1. What is the probability of winning the car if you "
            "switch?"
        ),
        "prefill": (
            "**Solution:** After the host opens door 3, there are two doors "
            "left: door 1 and door 2. One has the car and one has a goat, so "
            "each remaining door is equally likely to hide the car. The "
            "probability of winning is 1/2 whether you switch or stay, so "
            "switching does not improve your odds."
        ),
        "correct_answer": "2/3,two thirds,two-thirds",
        "wrong_answer": "1/2,one half,50/50,50-50",
    },
}
