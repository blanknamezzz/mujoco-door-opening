# Preparing to Succeed
## World-Model Foresight for Downstream Task Feasibility

**English presentation script · Team 42 · Total: approximately 3 minutes**

配合最新版 PPT：Zheyu 讲第 1–3 页，Haoming 讲第 4–6 页；第 7 页留作问答备份。以下时间包含翻页及短暂停顿，实际时长请以排练为准。标题和时间提示不朗读。

## Speaker 1 — Zheyu Zheng · 0:00–1:30

### Slide 1 · 0:00–0:28

Our project is Preparing to Succeed. We study how world-model foresight can help robots choose preparatory actions that preserve downstream task feasibility. Door opening is our test case: a robot may grasp the handle successfully, yet become unable to open the door because its starting position leaves too little room for later motion.

### Slide 2 · 0:28–0:57

We divide the task into positioning, alignment, grasping, and opening. Two policies choose the base position and the grasp. After moving, the robot observes again before selecting its grasp. Both policies consider later task completion, because local success can leave the next stage infeasible. The base remains fixed during opening.

### Slide 3 · 0:57–1:30

Our reward combines stage milestones, angular progress, and safe completion. It uses separate penalties for collision and incomplete opening, with independently adjustable weights. At episode end, the incomplete-opening penalty scales with the remaining fraction to fully open, from zero to one. Both policies receive downstream opening feedback. Physical joint limits remain execution constraints. We will tune the penalty weights on validation scenes. Haoming will explain implementation and evaluation.

**[Handover to Haoming; advance to slide 4.]**

## Speaker 2 — Haoming Li · 1:30–3:00

### Slide 4 · 1:30–2:05

We train two policies in simulation. The positioning policy selects a base pose. After arrival, the grasping policy selects a grasp from a fresh observation. Both use a shared predictor of downstream feasibility. We warm-start them separately, then alternate reinforcement-learning updates while fixing the other policy. Positioning learns from the whole task outcome; grasping learns from grasp and opening outcomes. Fixed controllers execute their chosen parameters.

### Slide 5 · 2:05–2:36

Building on V-JEPA's predictive representations, we ask when lookahead changes the right preparatory choice. We compare fixed and learned positioning, direct and world-model policies, and local versus downstream objectives, with full-path geometry as a strong reference. We measure safe opening fraction, full-opening success, physical violations, and planning time. Longer predictions may reveal delayed constraints but accumulate error, so we vary the horizon and refresh predictions after observations.

### Slide 6 · 2:36–3:00

Weeks seven and eight establish simulation and controllers. Weeks nine and texn build data and prediction models. Weeks eleven and twelve cover reinforcement learning and evaluation. Week thirteen completes the report and demo. Our goal is to identify when foresight improves preparation, and where prediction errors limit that benefit.

[Slide 7 | Speaker B]
Implementation has two stages. For positioning, we reset a scene and vary five candidate base poses. For grasping, we reset a reached state and vary four grasps. We train the grasp policy on states the position policy actually reaches. Navigation and inverse kinematics are fixed, and the base locks after grasping. These candidate budgets are pilot settings, to be validated. The learned components are the two policies and the shared predictor.