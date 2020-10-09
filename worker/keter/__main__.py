import os
import sys
from fire import Fire
import keter

def _work(queue, job):
    if job:
        try:
            getattr(keter, job)()
        except AttributeError:
            print(f"Error: No job named {job}")
            sys.exit(-1)
    if queue in ['cpu', 'gpu', 'all']:
        keter.work(queue)
    elif queue == 'none':
        pass
    else:
        print(f"Error: Queue {queue} is unsupported")
        sys.exit(-1)

class Controller:
    def up(self, queue='all'):
        """
        Run the foreman job and listen for more jobs.
        """
        _work(queue, 'foreman')

    def work(self, queue='all', job=''):
        """
        Just spawn a worker and listen for new jobs.
        """
        _work(queue, job)

def main():
    Fire(Controller)
