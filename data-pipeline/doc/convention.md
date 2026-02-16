## Conventions in datasets / Simpler

### EE proprio

Fractal data has xyzw quaternion in proprio (upon inspection), and I have been using wxyz in Simpler since it follows the transforms3d library. Bridge uses sxyz euler. EE pose saved in bridge data is relative to a top-down pose (instead of base pose). Both datasets use +x for forward, +y for left, and +z for upward.

### Gripper proprio and action

In Octo, bridge data has 1 for gripper state open and -1 for closed after normalization (continuous), and 1 for gripper action open and 0 for closing (without normalization, binarized). Fractal data has -1 for gripper state open and 1 for open closed after normalization (continuous), and also 1 for gripper action open and 0 for closing (without normalization, binarized).

I added gripper width (1 for open and 0 for closed) to the environment observation in Simpler. Then for the action in Simpler, widowx robot (bridge) has 1 for opening gripper and -1 for closing. Google robot has 1 for closing gripper and -1 for opening.
