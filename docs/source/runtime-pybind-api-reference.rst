Runtime Pybind API Reference
----------------------------------
Pybind APIs can load a serialized .pte file (see `Export to ExecuTorch Tutorial <tutorials/export-to-executorch-tutorial.html>`__ for how to get .pte file from a PyTorch `nn.Module`) and execute it using `torch.Tensor` as inputs, in a Python environment. The result also comes back as `torch.Tensor` so it can be used as a quick way to validate the correctness of the program.

For detailed information on how APIs evolve and the deprecation process, please refer to the `ExecuTorch API Life Cycle and Deprecation Policy <api-life-cycle.html>`__.

.. automodule:: executorch.runtime
.. autoclass:: Runtime
    :members: get, load_program

.. autoclass:: OperatorRegistry
    :members: operator_names

.. autoclass:: Program
    :members: method_names, load_method

.. autoclass:: Method
    :members: execute, metadata
