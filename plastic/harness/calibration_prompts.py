"""Default benign prompts for calibrating a pretrained chat backend on real chats.

Calibration fits thresholds to the harness signals a model produces while answering ordinary
requests (log-only). These prompts are the fallback when a caller passes none. Keep them
disjoint from anything used to evaluate interventions: an evaluation prompt that also sits
in the calibration set would make the fitted thresholds look better than they are.
"""

from __future__ import annotations

DEFAULT_CALIBRATION_PROMPTS: tuple[str, ...] = (
    "What is the difference between weather and climate?",
    "Explain how a bicycle stays upright while moving.",
    "Give me three tips for keeping houseplants alive in a dim apartment.",
    "Summarize the plot of a story about a lighthouse keeper who befriends a seal.",
    "How do I convert a recipe that serves four into one that serves six?",
    "Why does bread rise when you add yeast?",
    "Write a short, friendly note to a neighbor thanking them for collecting my mail.",
    "What are the main differences between a violin and a viola?",
    "Explain what a prime number is to a ten-year-old.",
    "List some questions I could ask when adopting a dog from a shelter.",
    "How does a thermostat decide when to turn the heating on?",
    "Describe the water cycle in four sentences.",
    "What should I pack for a weekend hiking trip in mild weather?",
    "Why is the sky blue during the day and red at sunset?",
    "Suggest a simple weekly schedule for learning to play the guitar.",
    "What does a librarian do besides checking out books?",
    "Explain the rules of tic-tac-toe and a strategy that never loses.",
    "How do bees communicate the location of flowers to each other?",
    "Write two sentences describing the smell of rain on a hot road.",
    "What is compound interest, with a small numeric example?",
    "How can I tell whether an egg is still fresh without cracking it?",
    "Compare taking notes by hand with typing them.",
    "Explain why leaves change color in autumn.",
    "Give me a packing list for moving a small kitchen into boxes.",
    "What is the purpose of a table of contents in a book?",
    "How do trains stay on their tracks around curves?",
    "Describe a good routine for cleaning a cast-iron pan.",
    "What are some ways to make a long car ride pleasant for children?",
    "Explain the difference between a lake and a pond.",
    "Write a limerick about a cat who dislikes Mondays.",
    "How does a compass find north?",
    "What are three common causes of a bicycle chain slipping?",
    "Explain what a metaphor is and give two examples.",
    "How do I politely decline an invitation to a party?",
    "Why do we see lightning before we hear thunder?",
    "Suggest names for a community garden newsletter.",
    "What is the role of a goalkeeper in football?",
    "How should I store fresh herbs so they last longer?",
    "Describe how to fold a paper airplane in numbered steps.",
    "What makes sourdough taste different from ordinary bread?",
)
