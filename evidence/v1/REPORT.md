# v1 geometric diagnostics

**PROXY_ONLY — INSUFFICIENT_EVIDENCE**. No independent execution outcomes.

| Metric | Mean | Scene bootstrap 95% CI |
|---|---:|---|
| panda/mount_a/ik_found | 0.8657 | [0.7639, 0.9491] |
| panda/mount_a/feasible_proxy | 0.3796 | [0.2315, 0.5417] |
| panda/mount_a/observable_proxy | 0.7917 | [0.7593, 0.8287] |
| panda/mount_a/conjunction_proxy | 0.2963 | [0.1806, 0.4259] |
| panda/mount_a/conflict_with_alternative | 0.4074 | [0.2222, 0.5926] |
| panda/mount_b/ik_found | 0.8657 | [0.7639, 0.9491] |
| panda/mount_b/feasible_proxy | 0.3796 | [0.2315, 0.5417] |
| panda/mount_b/observable_proxy | 0.8194 | [0.7778, 0.8657] |
| panda/mount_b/conjunction_proxy | 0.3102 | [0.1852, 0.4444] |
| panda/mount_b/conflict_with_alternative | 0.3333 | [0.1481, 0.5185] |
| sr5/mount_a/ik_found | 0.8704 | [0.7407, 0.9815] |
| sr5/mount_a/feasible_proxy | 0.1065 | [0.0139, 0.2177] |
| sr5/mount_a/observable_proxy | 0.7917 | [0.7593, 0.8287] |
| sr5/mount_a/conjunction_proxy | 0.0833 | [0.0139, 0.1667] |
| sr5/mount_a/conflict_with_alternative | 0.1111 | [0.0000, 0.2593] |
| sr5/mount_b/ik_found | 0.8704 | [0.7407, 0.9815] |
| sr5/mount_b/feasible_proxy | 0.1065 | [0.0139, 0.2177] |
| sr5/mount_b/observable_proxy | 0.8194 | [0.7778, 0.8657] |
| sr5/mount_b/conjunction_proxy | 0.0833 | [0.0139, 0.1667] |
| sr5/mount_b/conflict_with_alternative | 0.1111 | [0.0000, 0.2593] |
| arm_feasibility_delta | -0.2731 | [-0.4491, -0.1065] |
| factorial_conjunction_interaction | -0.0139 | [-0.0509, 0.0186] |
| baseline/panda/mount_a/analytic_and | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_a/geometry_planning | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_a/ik_collision | 0.4815 | [0.2963, 0.6667] |
| baseline/panda/mount_a/ik_only | 0.4815 | [0.2963, 0.6667] |
| baseline/panda/mount_a/same_information_mpc | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_a/visibility_only | 0.4074 | [0.2222, 0.5926] |
| baseline/panda/mount_a/weighted_feasibility | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_b/analytic_and | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_b/geometry_planning | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_b/ik_collision | 0.5185 | [0.3333, 0.7037] |
| baseline/panda/mount_b/ik_only | 0.5185 | [0.3333, 0.7037] |
| baseline/panda/mount_b/same_information_mpc | 0.5556 | [0.3704, 0.7407] |
| baseline/panda/mount_b/visibility_only | 0.4074 | [0.2222, 0.5926] |
| baseline/panda/mount_b/weighted_feasibility | 0.5556 | [0.3704, 0.7407] |
| baseline/sr5/mount_a/analytic_and | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_a/geometry_planning | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_a/ik_collision | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_a/ik_only | 0.1111 | [0.0000, 0.2222] |
| baseline/sr5/mount_a/same_information_mpc | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_a/visibility_only | 0.1111 | [0.0000, 0.2222] |
| baseline/sr5/mount_a/weighted_feasibility | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_b/analytic_and | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_b/geometry_planning | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_b/ik_collision | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_b/ik_only | 0.1111 | [0.0000, 0.2222] |
| baseline/sr5/mount_b/same_information_mpc | 0.1481 | [0.0370, 0.2963] |
| baseline/sr5/mount_b/visibility_only | 0.1111 | [0.0000, 0.2222] |
| baseline/sr5/mount_b/weighted_feasibility | 0.1481 | [0.0370, 0.2963] |

Analytic AND, geometric planning and finite-menu MPC use the evaluation conjunction itself: these are coverage / selection consistency diagnostics, never independent algorithm success.

Failures: `{"panda/interpolated_tracking_error": 46, "panda/arm_environment_proxy": 29, "sr5/arm_environment_proxy": 152, "panda/carried_object_proxy": 36, "panda/hand_environment_proxy": 75, "panda/IK_SEARCH_FAILED_UNKNOWN_REACHABILITY": 29, "sr5/carried_object_proxy": 24, "sr5/hand_environment_proxy": 58, "sr5/interpolated_tracking_error": 28, "sr5/IK_SEARCH_FAILED_UNKNOWN_REACHABILITY": 28}`
