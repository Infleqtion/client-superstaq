===============
Developer Guide
===============

This is the developer guide for Superstaq. If you have questions, please email superstaq@infleqtion.com or reach out on our `Slack workspace <https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw>`_.

.. contents:: **Table of Contents**
   :depth: 3
   :local:
   :backlinks: none

Developer Workflows
===================

Installation
------------
Please visit our `README <https://github.com/Infleqtion/client-superstaq/blob/main/README.md#installation-for-development>`_ for developer installation instructions.

Modifying Superstaq
-------------------

Push Edits and Make a Pull Request for Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. Update the main branch (e.g., ``git pull``).
#. Update relevant dependencies:
    * **To work on a single package**, navigate to the relevant subdirectory (e.g., ``cd qiskit-superstaq``) and run ``python3 -m pip install -e .``.
    * **Alternatively, install all Superstaq packages** by navigating to the repository root and run verbatim ``python3 -m pip install -e ./checks-superstaq -e ./general-superstaq -e ./qiskit-superstaq -e ./cirq-superstaq -e ./supermarq-benchmarks`` (packages should be installed in this specific order).
#. Create a new branch, forked from the up-to-date main branch. E.g., ``git checkout -b new-branch-name``
#. Add commits for your new code. E.g., ``git status`` to see what files have been changed, ``git add <file_name>``, ``git commit -m "<commit_message>"``.
    NOTE: Avoid using ``git add .`` or ``git add --all|-A`` when staging files to commit, as this will also include any untracked file which might have found its way into your local directory. Instead, add files individually (``git add file1 file2 ...``), or use ``git commit -a`` to stage and commit all changes to tracked files only.
#. Test your changes by running local checks with ``./checks/all_.py``. No internet connection is required for local checks. You can also run specific checks. e.g. ``./checks/format_.py``, or use ``./checks/all_.py -i`` to run all checks but just on locally modified files. More information about local checks can be found in the **Testing** section of the Developer Guide.
#. Test your changes by running integration tests with ``./checks/pytest_.py -- integration``. These tests require the internet to run, so all appropriate access tokens will need to be set before running locally. More information about integration tests can be found in the Testing` section of the Developer Guide.
#. Some changes will require new tests to be added to the local checks and integration tests. For example, if a conditional if/then statement is added, a corresponding check should be written to cover the new statement within the appropriate ``*_test.py`` file. Existing tests can be used as an example for how to create new tests.
#. Once all the checks pass locally, push your code: ``git push -u origin new-branch-name``. It should pass all checks on Github too. 
#. Use a draft PR until you think your code is ready for review. When ready for review, you can mark accordingly on GitHub and the checks will automatically run. If needed, you can also run checks on a draft PR by navigating to the Actions tab and triggering manually.
#. Engage in code review on your Github Pull Request. If you need to make changes to your code, you can just push new commits to the branch. When the code is all good, one of your reviewers will approve your Pull Request.\*
    #. Please limit changes in the Pull Request to those related to the issue the PR is resolving. Unrelated changes should be made in a different Pull Request.
    #. Open Draft Pull Requests if code is not yet ready for review.
#. Squash and merge your Pull Request into the main branch.\**
#. After the merge completes, delete the branch.

\*See `here <https://gist.github.com/mrsasha/8d511770ad9b282f3a5d0f5c8acdd10e>`__ for some good tips on code reviews.

Design
======

Design Principles
-----------------
- "If it runs locally, it should run in production."
- If a feature has been implemented in ``cirq-superstaq``, it should be implemented in ``qiskit-superstaq`` ASAP to maintain parity or vice-versa.
- Provider backend names are formatted in the following manner: ``provider_name_type``, e.g., ``ibmq_bogota_qpu``.

Testing
=======

This section describes the tests used to make sure that Superstaq functions as expected and is user friendly. As a note, 'checks' and 'tests' can be thought of as interchangeable here (although 'check' is generally a broader term).

Local Checks
-----------------
Superstaq local checks do not need an internet connection to be run locally i.e. mocking is frequently used to isolate the code and prevent the requirement of external dependencies. If a local check (i.e. non integration check) attempts to connect to the internet, it will fail. Local checks confirm that the code follows Superstaq guidelines for properties such as formatting, typing, and functionality. All Superstaq code must be covered by a local check.

.. code-block:: bash

   ./checks/all_.py  # run local checks

The same command is used to run client local checks (i.e., ``cirq_superstaq``, ``qiskit_superstaq``)

Descriptions 
-----------------
A high-level description of key Superstaq tests are as follows:

.. code-block:: bash

    checks/format_.py  # Enforces basic formatting rules (e.g. line length, import ordering) for python files and notebooks.
    checks/format_.py --apply  # Automatically update files to conform to formatting rules.
    checks/flake8_.py  # Style guide enforcement for python files.
    checks/pylint_.py  # Further style guide enforcement, including docstrings style.
    checks/mypy_.py  # Static type check.
    checks/pytest_.py  # Runs local python tests (from `*_test.py` files, not including `*_integration_test.py`).
    checks/pytest_.py --integration  # Runs integration tests (`*_integration_test.py`).
    checks/pytest_.py --notebook  # Executes example notebooks to make they're working.
    checks/coverage_.py  # Same as checks/pytest_.py, but also requires that every line of code is executed at some point in the process.
    checks/requirements.py  # Makes sure *superstaq dependencies are up to date in all *requirements.txt files.
    checks/requirements.py --apply  # Automatically updates requirements files to use the latest available version of any *superstaq dependency.checks/configs.py and checks that `setup.cfg` files are consistent across repos.
    checks/build_docs.py  # Ensures docs can be built. This will fail for e.g. incorrectly formatted code blocks in docstrings.
    checks/all_.py  # Runs all the non-integration checks described above.

By default, all test scripts will consider any tracked file in the current repository (meaning that new files will not be checked until they've been added to the repo via ``git add``). Passing ``-i`` or ``--incremental`` to any check will limit its scope to just locally modified files. Test scripts can also be passed individual files or subdirectories or prevented from checking specific files using ``-x <path>`` or ``--exclude <path>``.


Style Guide
===========

Nomenclature and Abbreviations
------------------------------

Operations can be referred to as ``op``.

``target`` should always be used to refer to device names (type ``str``) in the Superstaq "<vendor>_<device>_<type>" format, e.g. "ibmq_qasm_simulator".  The same will be used accross all clients. ``backend_name`` should be used for vendor-specific names of particular hardware backends, and ``backend`` for actual instances thereof.

Circuits
--------
Prefer ``circuit += op`` to append circuit operations, rather than ``circuit.append(op)``.

Inline Comments
-----------------

* **Capitalize** the first letter of each sentence
* If the comment is one line, do not leave ending punctuation (no period).

.. code-block:: python

   # This is a great example of a one-line comment with no period and capitalized first letter
   # do not leave comments all lowercase like this
   # If you have multiple sentences, end with a period. Don't forget the periods.

Docstrings
----------
All public methods and functions (except for those in test files) should have a clear Google-style docstrings. A simple template is included below; `see here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ for more examples. Note that we enforce the use of PEP 484 type hints, so types aren't required in parameter or return descriptions.

.. code-block:: python

    def function(arg1: str, arg2: int) -> str:
        """A single-line summary of the function, starting with a capital and ending in a period.

        If necessary, a more detailed description of what this function does, again starting with a
        capital and ending in a period. Lines should wrap at 100 characters. Inline `code snippets`
        should be placed between backticks.

        (INCORRECT) Don't attempt to evenly distribute long
        descriptions using early linebreaks (like this).

        Args:
            arg1: A description of `arg1`, starting with a capital and ending in a period.
            arg2: A description of `arg2`. Long descriptions should wrap at 100 characters, with
                subsequent lines indented once.
            (INCORRECT) arg2: Don't use early linebreaks in order to
                evenly distribute argument descriptions between lines.
            (INCORRECT) arg2: Don't indent subsequent all the way to the end of the argument name,
                              like this.

        Returns:
            A description of what this function returns, in sentences. This block can be skipped
            for functions with no return statement or which only return None.

        (INCORRECT) Returns: Do not put the return description inline, like this. They will be
            misinterpreted by docstring checkers.

        Yields:
            Like "Returns:", iff the function has any `yield` statements, what they yield should be
            described (in sentences) under a "Yields:" header.

        Raises:
            ValueError: If the function can raise any exceptions, include a description of the
                circumstances under which they will be thrown. Should be formatted like "Args:".
            AnotherKindOfError: Each unique exception should have its own description.
        """

The "Returns:" block is not required for properties (i.e. methods decorated with ``@property``); instead the docstring itself should be a description of its return value. For example,

.. code-block:: python

    class Example:
        @property
        def name(self) -> str:
            """The name given to this example."""


Imports
-------
Superstaq interfaces with many libraries that have overlapping functionalities such as Cirq, Qiskit, PyQuil, etc. For example, each of these libraries defines a quantum circuit abstraction. To avoid confusion or local name clashes, we prefer specifying explicit names. For example:

+--------------------------------------------------+--------------------------------------------------+
|                                                  |                                                  |
|.. code-block::                                   |.. code-block::                                   |
|                                                  |                                                  |
|   # BAD                                          |   # GOOD                                         |
|   from cirq import Circuit                       |   import cirq                                    |
|   from qiskit import QuantumCircuit              |   import qiskit                                  |
|   ...                                            |   ...                                            |
|   cirq_circuit = Circuit(...)                    |   cirq_circuit = cirq.Circuit(...)               |
|   qiskit_circuit = QuantumCircuit(...)           |   qiskit_circuit = qiskit.QuantumCircuit(...)    |
+--------------------------------------------------+--------------------------------------------------+

Or

+--------------------------------------------------+--------------------------------------------------+
|                                                  |                                                  |
|.. code-block::                                   |.. code-block::                                   |
|                                                  |                                                  |
|   # BAD: confusing that "pulse" is a module      |   # GOOD                                         |
|   from qiskit import pulse                       |   import qiskit                                  |
|   with pulse.build(backend) as program:          |   with qiskit.pulse.build(backend) as program:   |
|       gaussian_pulse = library.gaussian(...)     |       gaussian_pulse = library.gaussian(...)     |
|       pulse.play(gaussian_pulse)                 |       pulse.play(gaussian_pulse)                 |
+--------------------------------------------------+--------------------------------------------------+


Exceptions can be made if the explicit name is extremely long (e.g. ``from qiskit.providers import JobStatus as qjs``) or if the name is used extremely frequently and has low risk of name clash (e.g. ``from typing import Iterable, Union``).


Productivity Tips
=================
* For command-line python testing, use the ``ipython`` shell instead of normal python. ``ipython`` is like Jupyter but within your terminal.
* For either ``ipython`` or Jupyter, consider starting with

  >>> %load_ext autoreload
  >>> %autoreload 2

  This will enable autoreload.
* Also recommended: 

.. code-block:: bash

   alias py='ipython'
   alias pyr="ipython --InteractiveShellApp.extensions 'autoreload' --InteractiveShellApp.exec_lines '%autoreload 2'"
alias pygrep="grep -r --color=auto --include='*.py'"

(That last one, ``pyr``, runs an ipython shell with autoreloading already set up.)


DevOps
=================
Token Safety
------------
It is important to never share your Superstaq access key. They should never be commited to GitHub, and if this does happen, remember to reset your key on https://superstaq.infleqtion.com (click the refresh icon next to your key).

To prevent accidental sharing to GitHub, you can try the following methods:

- Save your token to ``~/.local/share/super.tech/superstaq-api-key`` (or in any other directory listed `here <https://github.com/SupertechLabs/client-superstaq/blob/b22c911f292ba75e081449d75b937094d53ff13d/general-superstaq/general_superstaq/superstaq_client.py#L463-L470>`__), for example with

.. code-block:: bash

    token_dir=~/.local/share/super.tech
    mkdir -p $token_dir
    echo <token> > $token_dir/superstaq-api-key

where ``<token>`` is replaced by your token.

- Save your token in your shell ``rc`` file (such as ``.zshrc`` or ``.bashrc``) by adding a line to the bottom that says ``export SUPERSTAQ_API_KEY="<token>"``

- In the terminal where you're going to run `qss/css/gss`, run ``$ export SUPERSTAQ_API_KEY="<token>"``

When you have implemented one of these methods, you can set up access to Superstaq without entering the token argument:

.. code-block:: python

    # cirq-superstaq
    service = css.Service()
    
    # qiskit-superstaq
    provider = qss.SuperstaqProvider()
