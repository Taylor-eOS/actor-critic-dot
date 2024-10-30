The `actor-critic-dot` project is an experimental setup for testing neural network behavior in a simple environment.
A dot repeatedly navigates from the center towarda one of four sides, learning over time to avoid red walls that are poisonous and prefer green walls that are rewarding.
It does this without any hardcoded knowledge of the rules.
The dot makes decisions based on recent rewards, refining its choices to avoid negative outcomes through an actor-critic reinforcement learning model.
Users can alter wall statuses by clicking on them, in the game window: toggling walls between safe (green) and poisonous (red).
This project was meant as a straightforward test for exploring neural network adaptation and reinforcement learning principles.

How to use:
- Download the `py` file
- Create a venv environment
- Install `requirements`
- Run `actor_critic.py`
