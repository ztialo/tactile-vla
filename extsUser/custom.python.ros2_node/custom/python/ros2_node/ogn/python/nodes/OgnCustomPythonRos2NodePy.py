"""
OmniGraph core Python API:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph/latest/Overview.html

OmniGraph attribute data types:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/ogn/attribute_types.html

Collection of OmniGraph code examples in Python:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/ogn/ogn_code_samples_python.html

Collection of OmniGraph tutorials:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.tutorials/latest/Overview.html
"""


class OgnCustomPythonRos2NodePyInternalState:
    """Convenience class for maintaining per-node state information"""

    def __init__(self):
        """Instantiate the per-node state information"""
        status = False


class OgnCustomPythonRos2NodePy:
    """The Ogn node class"""

    @staticmethod
    def internal_state():
        """Returns an object that contains per-node state information"""
        return OgnCustomPythonRos2NodePyInternalState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the output based on inputs and internal state"""
        state = db.internal_state

        try:
            # -----------------
            # read input values
            input1 = db.inputs.eef_topic
            input2 = db.inputs.gripper_topic
            # do custom computation
            state.status = True
            # ...
            # write output values
            db.outputs.outputAttribute1 = 0.0
            # -----------------
        except Exception as e:
            db.log_error(f"Computation error: {e}")
            return False
        return True
