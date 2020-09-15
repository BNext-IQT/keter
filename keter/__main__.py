import os
from fire import Fire
from keter import GPU, CPU, up, work

class Controller:
    def up(self, queue='cpu'):
        """
        Sets up the whole system and runs the agent. The kitchen sink command.
        """
        if queue in ['cpu', 'gpu']:
            up()
            work(queue)
        else:
            print(f"Error: Queue {queue} is unsupported")
            exit(-1)

def main():
    Fire(Controller)
