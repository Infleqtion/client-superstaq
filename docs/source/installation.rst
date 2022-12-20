Installation Guide
==================
Below are instructions for installing ``cirq-superstaq`` and ``qiskit-superstaq``. Please note that Python version 3.7 or higher is required.


Installing cirq-superstaq
-------------------------
First, create and then activate a virtual environment in your relevant local directory:

.. code-block:: bash

    python3 -m venv venv_cirq_superstaq
    source venv_cirq_superstaq/bin/activate

Then, install ``cirq-superstaq``:

.. code-block:: bash

    pip install cirq-superstaq

Run the following to install developer requirements, which is required if you intend to run checks locally.

.. code-block:: bash

    pip install .[dev]

Run the following to install neutral atom device dependencies.

.. code-block:: bash

    pip install -r neutral-atom-requirements.txt


Installing qiskit-superstaq
---------------------------
First, create and then activate a virtual environment in your relevant local directory:

.. code-block:: bash

    python3 -m venv venv_qiskit_superstaq
    source venv_qiskit_superstaq/bin/activate

Then, install ``qiskit-superstaq``:

.. code-block:: bash
    
    pip install qiskit-superstaq

Run the following to install developer requirements, which is required if you intend to run checks locally.

.. code-block:: bash

    pip install .[dev]

Run the following to install neutral atom device dependencies.

.. code-block:: bash

    pip install -r neutral-atom-requirements.txt