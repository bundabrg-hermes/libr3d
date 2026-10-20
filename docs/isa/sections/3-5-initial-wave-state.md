# 3.5. Initial Wave State

_(pages 37-37)_

Before a wave begins execution, some of the state registers including SGPRs and VGPRs are initialized with values derived either from state data, dynamic or derived data (e.g. interpolants or unique per-wave data). The values are derived from register state and dynamic wave-launch state. Note that some of this state is common across all waves in a draw call, and other state is unique per wave. This section describes what state is initialized per shader stage. Note that as usual in this spec, the shader stages refer to hardware shader stages and these often are not identical to software shader stages. State initialization is controlled by state registers which are defined in other documentation.
