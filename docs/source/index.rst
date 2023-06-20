.. docs-superstaq documentation master file, created by
   sphinx-quickstart on Thu Sep  8 16:32:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SuperstaQ Documentation
=======================
Welcome to SuperstaQ's documentation! Here you can find more information about our packages ``cirq-superstaq`` and ``qiskit-superstaq``, which allow access to SuperstaQ via a Web API through `Cirq <https://github.com/quantumlib/Cirq>`_ and `Qiskit <https://qiskit.org/>`_, respectively.

Check out some of our demos to see how SuperstaQ can help you:

.. raw:: html

   <div class="index-demos">
      <div class="index-demos-box">
         <div class="index-demos-headers"><a href="apps/community_detection_football.html">Community Detection</a></div>
         <div class="index-demos-desc">A general form of a network problem that can be used to detect fraud rings in transaction networks.</div>
      </div>
      <div class="index-demos-box">
         <div class="index-demos-headers"><a href="apps/max_sharpe_ratio_optimization.html">Sharpe Ratio Maximization</a></div>
         <div class="index-demos-desc">An example of portfolio optimization formulated as a QUBO and solved using simulated annealing.</div>
      </div>
      <div class="index-demos-box">
         <div class="index-demos-headers"><a href="apps/transaction_settlement.html">Transaction Settlement</a></div>
         <div class="index-demos-desc">Walk through a formulaton of the transaction settlement problem that maximizes the number of settled equity trades.</div>
      </div>
      <div class="index-demos-box">
         <div class="index-demos-headers"><a href="apps/insurance_pricing.html">Optimizing Insurance Prices</a></div>
         <div class="index-demos-desc">An example of how to calculate insurance premiums that balance profitability with market growth.</div>
      </div>
   </div>
   <br>


Learn more about SuperstaQ `here <https://www.infleqtion.com/superstaq>`_. To contact a member of our team, email us at **superstaq@infleqtion.com** or join our `Slack workspace <https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw>`_.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   get_started/credentials
   get_started/installation
   get_started/basics/basics

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Optimizations

   optimizations/aqt/aqt
   optimizations/cq/cq
   optimizations/ibm/ibm
   optimizations/qscout/qscout

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Applications

   apps/community_detection_football
   apps/insurance_pricing
   apps/max_sharpe_ratio_optimization
   apps/transaction_settlement

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Other Work

   supercheq/supercheq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Clients

   cirq_superstaq
   qiskit_superstaq
   general_superstaq
   supermarq
