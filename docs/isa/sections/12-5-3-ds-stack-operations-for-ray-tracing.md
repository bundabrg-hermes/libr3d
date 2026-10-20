# 12.5.3. DS Stack Operations for Ray Tracing

_(pages 138-138)_

DS_BVH_STACK_RTN_B32 is an LDS instruction to manage a per-thread shallow stack in LDS used in ray tracing BVH traversal. BVH structures consist of box nodes and triangle nodes. A box node has up to four child node pointers that may all be returned to the shader (to VGPRs) for a given ray (thread). A traversal shader follows one pointer per ray per iteration, and extra pointers can be pushed to a per-thread stack in LDS. Note: the returned pointers are sorted. This "short stack" has a limited size beyond that the stack wraps around and overwrites older items. When the stack is exhausted, the shader should switch to a stackless mode where it looks up the parent of the current node from a table in memory. The shader program tracks the last visited address to avoid re-traversing subtrees. DS_BVH_STACK_RTN_B32 vgpr(dst), vgpr(stack_addr), vgpr(lvaddr), vgpr[4](data)
