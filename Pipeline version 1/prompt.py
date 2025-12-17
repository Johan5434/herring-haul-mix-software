# simulations/prompt.py

def ask_yes_no(prompt: str) -> bool:
    """
    Simple yes/no input from keyboard.
    """
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in ("y", "yes", "j", "ja"):
            return True
        if ans in ("n", "no", "nej"):
            return False
        print("  Please answer 'y' or 'n'.")


def ask_menu_choice(title: str, options):
    """
    Generic text menu.

    Parameters
    ----------
    title : str
        Menu title.

    options : sequence of (key, description)
        e.g. [("1", "Plot"), ("2", "Skip"), ("q", "Quit")]

    Returns
    -------
    key : str
        The selected option key (exactly as in options).
    """
    # allow case-insensitive input, but return the original key
    valid = {key.lower(): key for key, _ in options}

    while True:
        print(f"\n=== {title} ===")
        for key, desc in options:
            print(f"  [{key}] {desc}")
        choice = input("Choose an option: ").strip().lower()

        if choice in valid:
            return valid[choice]

        print("  Invalid choice, please try again.")


def ask_experiment_action() -> str:
    """
    Menu for what you want to do after a batch of simulated hauls
    has been computed and projected into the PCA space.

    Currently:
      [1] Plot simulated hauls in the PCA space
      [2] Classification experiment (centroid distances) on this batch
      [3] Continue to a NEW batch of mixes
      [q] Quit the experiment loop

    Returns:
      "1", "2", "3" or "q"
    """
    return ask_menu_choice(
        "Actions for this batch",
        [
            ("1", "Plot simulated hauls in the PCA space"),
            ("2", "Classification experiment (centroid distances) on this batch"),
            ("3", "Continue to a NEW batch of mixes"),
            ("q", "Quit the experiment loop"),
        ],
    )
