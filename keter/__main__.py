import os
from fire import Fire
from keter import GPU, CPU, work, foreman

def _work(queue, everything=False):
    if queue in ['cpu', 'gpu', 'all']:
        if everything:
            foreman()
        work(queue)
    else:
        print(f"Error: Queue {queue} is unsupported")
        exit(-1)

class Controller:
    def up(self, queue='all'):
        """
        Run the foreman job and listen for more jobs.
        """
        _work(queue, everything=True)

    def work(self, queue='all'):
        """
        Just spawn a worker and listen for new jobs.
        """
        _work(queue)

def main():
    Fire(Controller)
