.. docs-superstaq documentation master file, created by
   sphinx-quickstart on Thu Sep  8 16:32:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Superstaq Documentation
=======================
Welcome to Superstaq's documentation! Here you can find more information about our packages ``cirq-superstaq`` and ``qiskit-superstaq``, which allow access to Superstaq via a Web API through `Cirq <https://github.com/quantumlib/Cirq>`_ and `Qiskit <https://qiskit.org/>`_, respectively.


Check out some of our demos to see how Superstaq can help you:

.. raw:: html

   <div class="index-demos">
      <div class="index-demos-box">
         <div class="index-demos-headers"><a href="apps/supercheq/supercheq.html">Supercheq</a></div>
         <div class="index-demos-desc">Supercheq is our novel quantum fingerprinting protocol and can be used with both `qiskit-superstaq` and `cirq-superstaq`</div>
      </div>
      <div class="index-demos-box">
         <div class="index-demos-headers"><a href="apps/max_sharpe_ratio_optimization.html">Sharpe Ratio Maximization</a></div>
         <div class="index-demos-desc">An example of portfolio optimization formulated as a QUBO and solved using simulated annealing.</div>
      </div>
   </div>
   <br>


Learn more about Superstaq `here <https://www.infleqtion.com/superstaq>`_. To contact a member of our team, email us at superstaq@infleqtion.com or join our `Slack workspace <https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw>`_.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   get_started/credentials
   get_started/installation
   get_started/basics/basics
   get_started/access_info/access_info

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Optimizations

   optimizations/aqt/aqt
   optimizations/hilbert/hilbert
   optimizations/ibm/ibm
   optimizations/qscout/qscout

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Applications

   apps/supercheq/supercheq
   apps/max_sharpe_ratio_optimization

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contact Us

   contact

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Clients

   cirq_superstaq
   qiskit_superstaq
   general_superstaq
   supermarq