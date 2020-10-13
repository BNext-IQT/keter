import os
import sys
from fire import Fire
from redis import Redis
import keter

def _get_redis_url() -> str:
    return os.environ.get('KETER_QUEUE') or ''

def _work(queue, job=None, params=None):
    if job:
        try:
            if params:
                getattr(keter, job)(**params)
            else:
                getattr(keter, job)()
        except AttributeError:
            print(f"Error: No job named {job}")
            sys.exit(-1)
    if queue in ['cpu', 'gpu', 'all']:
        conn = _get_redis_url()
        keter.work(queue, conn)
    elif queue == 'none':
        pass
    else:
        print(f"Error: Queue {queue} is unsupported")
        sys.exit(-1)

class Controller:
    """
    Available commands: up, work.
    
    See specific commands for built-in help.
    """

    def up(self, queue='all'):
        """
        Run the foreman job and listen for more jobs.

        Keyword arguments:
        queue -- What queue to listen for (eg. gpu, cpu). Use "all" to listen for anything. 
        """
        keter.foreman(_get_redis_url())
        _work(queue)

    def work(self, queue='all', job='', params=''):
        """
        Spawn a worker and listen for new jobs.
        
        Keyword arguments:
        queue -- What queue to listen for (eg. gpu, cpu). Use "all" to listen for anything. 
                 The queue "none" can be used with the job param to just execute a job.
        job -- Job to execute before joining the queue.
        params -- Job parameters if applicable.
        """
        _work(queue, job, params)

def main():
    Fire(Controller)
