# Cross-behavior Stage A offline diagnostic

Run `20261114T151144196413Z-de1d828d87`; input fingerprint `de1d828d87942b9087be7514729911e16fe4bc7643ddb43ac8c6dabdd54e4453`.

Computed from 19 indexed trajectories and 9160 saved control steps. No physics, training or demonstration collection was started.

| Task | Robot | Episodes | Recorded full successes | Native predicate ever |
|---|---|---:|---:|---:|
| drawer | panda | 3 | 0 | 0 |
| drawer | ur5e | 3 | 0 | 0 |
| push | panda | 3 | 0 | N/A |
| push | ur5e | 4 | 0 | N/A |
| stack | panda | 3 | 0 | 3 |
| stack | ur5e | 3 | 0 | 2 |

## Episode diagnostics

All numbers below are calculated from saved JSONL. The control-step 1 object state is the first logged post-action state, not a fabricated t=0 value. Perception was computed before the action, so perception/truth comparisons are explicitly approximate in time. A latched estimate is not fresh per-frame detection.

Native predicate events are separated from frozen full-contract outcomes. Tracking errors are geometric diagnostics, not evidence that perception or mechanics caused failure.

### push-ur5e-000

Steps: 40; last executed phase: None; diagnostic: `NO_TARGET_ESTIMATE`.
Object localization error: None m; goal XY localization error: None m; recorded displacement: 0.0 m; lateral displacement: 0.0 m; direction error: None degrees.
Mean reported pre-action tracking error: None m; final target-to-post-action TCP error: None m.

### push-ur5e-001

Steps: 600; last executed phase: 3; diagnostic: `OFF_GOAL_AFTER_CONTACT`.
Object localization error: 0.005805142150377742 m; goal XY localization error: 0.07400165729410023 m; recorded displacement: 0.13577154256829982 m; lateral displacement: 0.08512807311157078 m; direction error: 38.82876259107213 degrees.
Mean reported pre-action tracking error: 0.12170804471098881 m; final target-to-post-action TCP error: 3.802227515188327e-05 m.

### push-ur5e-002

Steps: 600; last executed phase: 3; diagnostic: `OFF_GOAL_AFTER_CONTACT`.
Object localization error: 0.004880464125833606 m; goal XY localization error: 0.06574172014340604 m; recorded displacement: 0.1394891845789003 m; lateral displacement: 0.08020062956589569 m; direction error: 35.09679487307614 degrees.
Mean reported pre-action tracking error: 0.11977618905239003 m; final target-to-post-action TCP error: 0.0007318754496973378 m.

### push-ur5e-003

Steps: 600; last executed phase: 3; diagnostic: `OFF_GOAL_AFTER_CONTACT`.
Object localization error: 0.005822373802021496 m; goal XY localization error: 0.07187264789909857 m; recorded displacement: 0.13891967172832925 m; lateral displacement: 0.08295443528553187 m; direction error: 36.66530910997864 degrees.
Mean reported pre-action tracking error: 0.11738379734370971 m; final target-to-post-action TCP error: 0.00038696269782772657 m.

### push-panda-000

Steps: 600; last executed phase: 0; diagnostic: `APPROACH_NOT_COMPLETED_VISUAL_OR_CONTROL_AMBIGUOUS`.
Object localization error: 0.5034671093776417 m; goal XY localization error: 0.07077205188628122 m; recorded displacement: 0.0 m; lateral displacement: 0.0 m; direction error: None degrees.
Mean reported pre-action tracking error: 0.5010605613784611 m; final target-to-post-action TCP error: 0.4898098816374314 m.

### push-panda-001

Steps: 600; last executed phase: 0; diagnostic: `APPROACH_NOT_COMPLETED_VISUAL_OR_CONTROL_AMBIGUOUS`.
Object localization error: 0.5193025405917701 m; goal XY localization error: 0.062487715158732665 m; recorded displacement: 0.0 m; lateral displacement: 0.0 m; direction error: None degrees.
Mean reported pre-action tracking error: 0.4837715497513283 m; final target-to-post-action TCP error: 0.47426760877961976 m.

### push-panda-002

Steps: 600; last executed phase: 0; diagnostic: `APPROACH_NOT_COMPLETED_VISUAL_OR_CONTROL_AMBIGUOUS`.
Object localization error: 0.5457770661553283 m; goal XY localization error: 0.06045526255443861 m; recorded displacement: 0.0 m; lateral displacement: 0.0 m; direction error: None degrees.
Mean reported pre-action tracking error: 0.46234251285726863 m; final target-to-post-action TCP error: 0.45650378701842403 m.

### stack-ur5e-000

Steps: 600; last executed phase: 0; diagnostic: `NATIVE_STACK_NOT_REACHED`.
First native success: step None; first recorded release in release phase: None; final TCP: [1.506662436835704, -0.45935423670170433, 1.4891067759748515]; retreat distance: 0.9613047686522174 m; threshold: 0.1 m; observed time remaining after release: None s; timeout reason: `NATIVE_STACK_NOT_REACHED`.
Mean reported pre-action tracking error: 0.2202745363516943 m; final target-to-post-action TCP error: 0.10749184934099805 m.

### stack-ur5e-001

Steps: 600; last executed phase: 7; diagnostic: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
First native success: step 560; first recorded release in release phase: 560; final TCP: [1.9554250565590205, -0.5494584844404699, 0.9914781452491677]; retreat distance: 0.003497534561296401 m; threshold: 0.1 m; observed time remaining after release: 2.0 s; timeout reason: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
Mean reported pre-action tracking error: 0.07315052702854115 m; final target-to-post-action TCP error: 0.101127596046813 m.

### stack-ur5e-002

Steps: 600; last executed phase: 7; diagnostic: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
First native success: step 563; first recorded release in release phase: 563; final TCP: [1.9598880251139879, -0.5545926880836114, 0.9872537630469249]; retreat distance: 0.0029058999971945322 m; threshold: 0.1 m; observed time remaining after release: 1.85 s; timeout reason: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
Mean reported pre-action tracking error: 0.07361889863152711 m; final target-to-post-action TCP error: 0.10534083422284939 m.

### stack-panda-000

Steps: 600; last executed phase: 7; diagnostic: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
First native success: step 554; first recorded release in release phase: 554; final TCP: [1.9576634559643504, -0.5533085480212745, 1.0019337227405851]; retreat distance: 0.0126496939763807 m; threshold: 0.1 m; observed time remaining after release: 2.3 s; timeout reason: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
Mean reported pre-action tracking error: 0.07096839232093764 m; final target-to-post-action TCP error: 0.09065114481699146 m.

### stack-panda-001

Steps: 600; last executed phase: 7; diagnostic: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
First native success: step 563; first recorded release in release phase: 563; final TCP: [1.9566517557978496, -0.5482877518878968, 0.9880174981628399]; retreat distance: 0.0026823813468926246 m; threshold: 0.1 m; observed time remaining after release: 1.85 s; timeout reason: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
Mean reported pre-action tracking error: 0.07214224120211607 m; final target-to-post-action TCP error: 0.10458524526330039 m.

### stack-panda-002

Steps: 600; last executed phase: 7; diagnostic: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
First native success: step 561; first recorded release in release phase: 561; final TCP: [1.960719781930728, -0.5539004786572561, 0.9912104524513083]; retreat distance: 0.002711491295995706 m; threshold: 0.1 m; observed time remaining after release: 1.95 s; timeout reason: `NATIVE_STACK_BUT_RETREAT_INCOMPLETE`.
Mean reported pre-action tracking error: 0.07163117630339157 m; final target-to-post-action TCP error: 0.10138804105424762 m.

### drawer-ur5e-000

Steps: 600; last executed phase: 3; diagnostic: `CONTACT_WITH_INSUFFICIENT_OPENING`.
Handle estimate available for 595/600 steps (latched); first contact: 556; total/longest contact: 2.25/2.25 s; maximum normalized opening: 0.019110917141055977; visual flag: `ESTIMATE_AVAILABLE_VISIBILITY_NOT_CERTIFIED`. Exact dynamic handle ground truth is unavailable in this schema.
Mean reported pre-action tracking error: 0.27003235498811884 m; final target-to-post-action TCP error: 0.3950829292836936 m.

### drawer-ur5e-001

Steps: 40; last executed phase: None; diagnostic: `VISUAL_NO_ESTIMATE`.
Handle estimate available for 0/40 steps (latched); first contact: None; total/longest contact: 0.0/0.0 s; maximum normalized opening: 2.4038774492230815e-07; visual flag: `NO_ESTIMATE`. Exact dynamic handle ground truth is unavailable in this schema.
Mean reported pre-action tracking error: None m; final target-to-post-action TCP error: None m.

### drawer-ur5e-002

Steps: 40; last executed phase: None; diagnostic: `VISUAL_NO_ESTIMATE`.
Handle estimate available for 0/40 steps (latched); first contact: None; total/longest contact: 0.0/0.0 s; maximum normalized opening: 0.0003481604619999574; visual flag: `NO_ESTIMATE`. Exact dynamic handle ground truth is unavailable in this schema.
Mean reported pre-action tracking error: None m; final target-to-post-action TCP error: None m.

### drawer-panda-000

Steps: 600; last executed phase: 1; diagnostic: `APPROACH_VISUAL_OR_CONTROL_AMBIGUOUS`.
Handle estimate available for 600/600 steps (latched); first contact: None; total/longest contact: 0.0/0.0 s; maximum normalized opening: 2.728114537116849e-06; visual flag: `ESTIMATE_AVAILABLE_VISIBILITY_NOT_CERTIFIED`. Exact dynamic handle ground truth is unavailable in this schema.
Mean reported pre-action tracking error: 0.10446587632691945 m; final target-to-post-action TCP error: 0.012727251561971567 m.

### drawer-panda-001

Steps: 600; last executed phase: 1; diagnostic: `APPROACH_VISUAL_OR_CONTROL_AMBIGUOUS`.
Handle estimate available for 600/600 steps (latched); first contact: None; total/longest contact: 0.0/0.0 s; maximum normalized opening: 2.403859823876551e-07; visual flag: `ESTIMATE_AVAILABLE_VISIBILITY_NOT_CERTIFIED`. Exact dynamic handle ground truth is unavailable in this schema.
Mean reported pre-action tracking error: 0.11109624778222399 m; final target-to-post-action TCP error: 0.012772588989374847 m.

### drawer-panda-002

Steps: 40; last executed phase: None; diagnostic: `VISUAL_NO_ESTIMATE`.
Handle estimate available for 0/40 steps (latched); first contact: None; total/longest contact: 0.0/0.0 s; maximum normalized opening: 0.0003481804340882538; visual flag: `NO_ESTIMATE`. Exact dynamic handle ground truth is unavailable in this schema.
Mean reported pre-action tracking error: None m; final target-to-post-action TCP error: None m.

## Evidence and reuse limits

- `input_manifest.json` records absolute paths, bytes, SHA256, presence, read status and expected-checksum matches. Raw MP4/NPZ/PKL/HDF5/weights are not uploaded.
- `code_context.json`, `code_snapshot/` and `source_changes.patch` preserve the relevant executed-source context and uncommitted changes without staging unrelated files.
- `phase_timeline.csv`, `episode_metrics.csv`, `perception_audit.json` and bounded initial-image overlays expose the calculations for review.
- Missing detections/ground truth are null, not zero. Visual/control ambiguity remains unresolved where saved evidence cannot distinguish it.
- The next scientific decision remains expert observability and trajectory timing. These offline diagnostics neither authorize another physical repair nor establish algorithm novelty.
