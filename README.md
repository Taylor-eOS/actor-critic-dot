The `actor-critic-dot` project is a simple experimental setup for testing neural network behavior.
A moving dot repeatedly decides towards which of the four sides of the field to move from the center, learning over time to avoid red walls that are poisonous and prefer green walls that are rewarding.
It does this without any hardcoded knowledge of the environment or the rules of the game.
The dot makes decisions through an actor-critic reinforcement learning model.
Users can alter wall statuses by clicking on them in the game window, which will toggle them between safe (green) and poisonous (red).
This project was meant as a straightforward test for exploring neural network adaptation and reinforcement learning principles.
Programming this was incredibly finicky, as neural nets are fickle and impossible to get to do what you want. So this project did not progress very far.

How to use:
- Download the `py` file
- Create a venv environment
- Install `requirements`
- Run `actor_critic.py`
