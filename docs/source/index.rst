.. docs-superstaq documentation master file, created by
   sphinx-quickstart on Thu Sep  8 16:32:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Superstaq Documentation
=======================
Welcome! Here you can find more about Infleqtion's state-of-the-art quantum software platform that uses proprietary cross-layer optimization techniques to deliver unmatched performance.

.. raw:: html

   <div class="container-index">
      <div class="grid-exec">
         <img src="_static/icons/EnhancedExecution.png" width=50px style="padding: 0 0 10px 0">
         <div class="grid-header">Enhanced Execution</div>
         Can improve performance by ≥ 10x. Read our <a href="https://arxiv.org/abs/2309.05157">white paper</a> to learn more.
      </div>
      <div class="grid-errmit">
         <img src="_static/icons/Errormitigation.png" width=50px style="padding: 0 0 10px 0">
         <div class="grid-header">Next-Gen Error Mitigation</div>
         Incorporates techniques like <a href="./optimizations/ibm/ibmq_dd.html">Dynamical Decoupling</a>.
      </div>
      <div class="grid-decomp">
         <img src="_static/icons/Optimaldecomp.png" width=50px style="padding: 0 0 10px 0">
         <div class="grid-header">Optimized Decomposition</div>
         Exploits the hardware's full set of native operations
         <img src="_static/graphics/stack_cropped.png" style="padding: 5px 0 0 0;">
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
   optimizations/sqorpius/sqorpius
   optimizations/ibm/ibm
   optimizations/qscout/qscout

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Applications

   apps/supercheq/supercheq
   apps/max_sharpe_ratio_optimization
   apps/dfe/dfe
   apps/aces/aces

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Resources

   resources/contact
   resources/developer_guide
   resources/links

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Clients

   cirq_superstaq
   qiskit_superstaq
   general_superstaq