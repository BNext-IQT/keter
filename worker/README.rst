Keter Worker
~~~~~~~~~~~~

This Python project executes jobs for various tasks in coronavirus and biosecurity science.

Jobs may includine:
* Training or refining antivirial models
* Train or refining forecasting models
* Inference using trained models
* Statically build UIs

Installation
~~~~~~~~~~~~

:: 

    pip install keter


Usage
~~~~~

Requires a local Redis for job management.

To just start a worker simply run:

::
    
    keter up


You can also use a remote Redis by setting the KETER_QUEUE environment variable using a connection string.

There are other commands which you can read about using:

::

    keter -h

License and Acknowledgment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apache 2. See LICENSE file for details.

A project of `B.Next <https://www.bnext.org/>`_.