from cgiar.utils import time_activity

# choose what you want to run from
# `solutions.manuel` or solutions.matthew
from solutions.manuel.v2.run import run

if __name__ == "__main__":
    with time_activity("Training & Submission"):
        run()
