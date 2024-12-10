from yahtzee.yahtzee import State


class Agent:
    def __init__(self):
        pass

    def get_action(self, state: State) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        return "Unnamed Agent"
