import os
from fire import Fire
from keter import GPU, CPU, up, work

def _work(queue, everything=False):
    if queue in ['cpu', 'gpu', 'all']:
        if everything:
            up()
        work(queue)
    else:
        print(f"Error: Queue {queue} is unsupported")
        exit(-1)

class Controller:
    def up(self, queue='all'):
        """
        Sets up the whole system and runs the worker. The kitchen sink command.
        """
        _work(queue, everything=True)

    def work(self, queue='all'):
        """
        Just spawn a worker and listen for new jobs.
        """
        _work(queue)

def main():
    Fire(Controller)
