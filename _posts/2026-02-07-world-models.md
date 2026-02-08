---
title: "What are World Models"
date: 2026-02-07
author: Jiale Zhi
---

## What are "world models"?

There are many different definitions of world models. Basically, a **world model** is an internal model an agent learns to **represent** what it is seeing or perceiving, **predict** how the world or environment will change. This change is often conditioned on actions, but it's not required. The agent then **uses those predictions** to choose actions—via planning, imagination, or policy learning. The framing is explicitly inspired by how humans rely on simplified internal models rather than raw sensory streams. ([Ha & Schmidhuber, 2018][1])

Mathematically, I'll borrow the definition from **LeCun's Enc/Pred + state $s_t$ + latent $z_t$ + action $a_t$** scaffold ([LeCun, 2024][20]), and then show how each "school" is just a different choice of **state definition**, **stochasticity placement**, and **training objective**.

I’ll keep notation consistent:

* Observation $o_t$ (image/video frame, proprioception, text token, etc.)
* Action $a_t$ (optional)
* Encoder $h_t = \mathrm{Enc}_\phi(o_t)$
* World-state $s_t$ (model's internal state / belief)
* Stochastic latent $z_t \sim p_\theta(z_t \mid \cdot)$ (captures uncertainty / multimodality)
* Predictor $s_{t+1} = \mathrm{Pred}_\theta(h_t, s_t, a_t, z_t)$

A “world model” is then a learned transition model that supports **counterfactual rollouts** under proposed actions.


In practice, “world model” can mean a few related things:

* **A generative/predictive simulator**: learn $p(o_{t+1} \mid o_t, a_t)$ or (more commonly) $p(z_{t+1} \mid z_t, a_t)$ in a latent space.
* **A latent state-space model**: infer a compact hidden state $z_t$ that summarizes the past, then predict transitions and rewards in that space.
* **A planning-centric model**: predict *only* what’s needed for planning (reward/value/policy), not necessarily pixels (MuZero-style).
* **A foundation “world” generator**: large video/3D-like interactive models that produce consistent, navigable environments from prompts (Genie/Genie 3).

The unifying idea: **learn a model of the environment**, then **use it to improve decision-making**—often with much better sample-efficiency than purely model-free RL.

---

## A useful taxonomy: what can a "world model" predict?

Here's a practical way to classify approaches:

* **Observation-predictive (pixels or tokens)**
  - Good for: simulation, video generation, detailed forecasting
  - Hard parts: long-horizon coherence, compounding error

* **Latent-state predictive (RSSM / MDN-RNN / Transformers in latent)**
  - Good for: scalable planning and imagination training
  - Hard parts: keeping latents "grounded" and useful for control

* **Value-aware / planning-centric (MuZero-style)**
  - Predict: reward/value/policy transitions, not pixels
  - Good for: strong planning performance without full simulation
  - Risk: can become an opaque "planning model" that's hard to interpret as a world simulator

* **Representation-predictive (JEPA-style)**
  - Predict: embeddings/features of masked future/regions
  - Good for: semantics, downstream tasks, robotics priors
  - Risk: needs a principled way to connect representations to action selection/planning

* **Foundation interactive world generators (Genie/Genie 3)**
  - Good for: broad interactive environment generation
  - Risk: evaluation is tricky (what does "correct physics" mean across prompts?), and action-conditioning is challenging.

---

## Major "families" of world-model approaches

It’s useful to read the literature as a few recurring families, rather than as a single line of progress.

A widely cited **reference point** is *World Models* (Ha & Schmidhuber, 2018): it popularized a clean, modular recipe—**compress → predict → control**—and the idea of training a controller *inside the model’s own “dream”* before deploying in the real environment. ([Ha & Schmidhuber, 2018][1])

(A deeper walk-through of the VAE + MDN-RNN + controller architecture is in the next section.)

### 1) Latent dynamics models for planning: PlaNet

**PlaNet** (Hafner et al.) is an early landmark for planning directly in a learned latent space. It uses a latent dynamics model with **both deterministic and stochastic** transition components and performs online planning in latent space. ([Hafner et al., 2019][3])

What it moved forward vs. the 2018 template:

* Stronger state-space modeling geared toward multi-step prediction/planning
* Planning algorithms tailored to latent rollouts (rather than only using the latent as policy input)

### 2) “Imagination training” with actor-critic: Dreamer → DreamerV2/V3 → Dreamer 4

**Dreamer** made “learn in latent imagination” a core, scalable strategy: it learns behaviors **purely by latent imagination** and propagates value gradients back through imagined trajectories. ([Hafner et al., 2020][4])
Google’s blog summary also describes Dreamer as a model-based pipeline of (1) learning the world model, (2) learning behaviors from predictions, (3) acting to collect more experience. ([Hafner et al., 2020][4])

Then:

* **DreamerV2** introduced discrete world-model representations and is positioned as achieving human-level Atari performance by learning behaviors within a separately trained world model. ([Hafner et al., 2021][6])
* **DreamerV3** aims at *robustness across diverse domains with fixed hyperparameters*, still centered on learning a world model for imagination training. ([Hafner et al., 2023][7])
* **Dreamer 4 (2025)** pushes scale and complexity: it frames “imagination training” in a more scalable setting (Minecraft), emphasizing accurate object interactions and even reporting a diamond-acquisition challenge from offline data. ([Hafner et al., 2025][8])

**Why this matters:** Dreamer-like agents operationalize world models as a *workhorse training substrate*—you spend much of your learning compute inside the model, not the environment.

### 4) Planning-centric, non-pixel models: MuZero

**MuZero** is sometimes described as a world model that doesn’t try to reconstruct the world. Instead, it learns a model that predicts the quantities most relevant to planning—**reward, value, and policy**—and uses tree search. ([Schrittwieser et al., 2020][9])

This is an important conceptual branch:

* A “world model” doesn’t have to be a realistic simulator in pixel space.
* It can be an *abstract dynamics model* that is only required to be accurate for decision-making.

### 5) Video-prediction world models for sample efficiency: SimPLe and beyond

**SimPLe** (Kaiser et al.) is an example of using learned video prediction models as simulators to train policies with limited environment interaction (often discussed in the Atari 100k setting). ([Kaiser et al., 2020][10])
This line emphasizes the classic tradeoff: pixel-predictive models can be expensive and brittle, but they can reduce real interaction needs if made good enough.

### 6) "Foundation world models": Genie → Genie 3

More recently, the term “world model” is also used for large models that generate **interactive environments** from broad data sources.

* **Genie (2024)** is presented as a “generative interactive environment” trained unsupervised from unlabeled Internet videos, with a tokenizer, an autoregressive dynamics model, and a latent action model; it’s described as a “foundation world model” at 11B parameters. ([Bruce et al., 2024][11])
* **Genie 3 (2025)** is announced as a general-purpose world model that can generate diverse interactive environments navigable in real time (24 FPS) for a few minutes at 720p. ([Google DeepMind, 2025][12])

This is a shift from “world model for one RL benchmark” to “world model as a general interactive generator.”

### 7) Representation-predictive (non-generative) world models: JEPA / V-JEPA 2

Another modern branch argues you don’t always need to predict pixels; you can predict *representations*.

* **I-JEPA** proposes a joint-embedding predictive architecture for learning semantic image representations by predicting representations of masked regions (non-generative). ([Assran et al., 2023][13])
* **V-JEPA 2 (2025)** extends this philosophy to video and positions itself explicitly as a self-supervised video model enabling understanding/prediction/planning in the physical world—pretraining action-free on **over 1 million hours** of video/images and then incorporating a small amount of interaction data. ([Bardes et al., 2025][14])

This family is often motivated by: *prediction at the pixel level forces the model to care about irrelevant detail*, while representation prediction can focus on task-relevant structure.

---

## The classic reference: *World Models* ([Ha & Schmidhuber, 2018][1])

When people say “World Models” in quotes, they very often mean the 2018 paper/blog by **David Ha & Jürgen Schmidhuber**.
It introduced a clean, modular recipe—**compress → predict → control**—and showcased training a policy *inside the model’s own “dream”* before transferring it back to the real environment.

### The architecture: V (vision) + M (memory/dynamics) + C (controller)

The agent is split into three parts:

* **V: Vision model (VAE)**

  * A convolutional **Variational Autoencoder** compresses each 64×64 RGB frame into a latent vector $z$.
  * In their experiments: $z \in \mathbb{R}^{32}$ for CarRacing, and $z \in \mathbb{R}^{64}$ for Doom.

* **M: Memory / dynamics model (MDN-RNN)**

  * An LSTM-based recurrent model predicts a **distribution** over the next latent $z_{t+1}$, not a single point estimate, using a **Mixture Density Network** head.
  * It models $p_\theta(z_{t+1} \mid a_t, z_t, h_t)$, and introduces a **temperature $\tau$** at sampling time to control stochasticity/uncertainty in imagined rollouts.

* **C: Controller (tiny policy)**

  * Deliberately **small**—in the paper it’s a **single-layer linear model** mapping $[z_t, h_t]$ to actions.
  * The key “trick”: make the controller small so optimizing behavior is easier, while letting the world model carry most complexity.

### Training pipeline (their recipe)

For CarRacing, the paper summarizes a straightforward pipeline:

1. **Collect rollouts** from a random policy (e.g., 10,000 rollouts for CarRacing).
2. Train **VAE** to encode frames into $z$.
3. Train **MDN-RNN** to predict next $z$ (and in Doom also predict “done/death”).
4. Train **controller C** using **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy).

They highlight the extreme parameter imbalance (world model huge, controller tiny), e.g. CarRacing controller has under 1k parameters while the VAE has ~4.3M.

### The “dream” idea: training and acting inside hallucinated rollouts

Once you have $V$ and $M$, you can run the environment “in your head”:

* Encode real observation → latent $z_t$
* Predict $z_{t+1}$ with $M$ (sampled with temperature $\tau$)
* (Optionally) decode predicted $z$ back to pixels for visualization
* Train/execute policy **inside this virtual environment**

They explicitly demonstrate the agent driving in a hallucinated CarRacing environment generated by the MDN-RNN and rendered through the VAE decoder.

### A particularly insightful result: temperature helps transfer in Doom

In the VizDoom “Take Cover” experiment, they train the policy entirely in the learned latent simulator and then deploy it in the real environment. They report that adjusting the MDN-RNN sampling temperature $\tau$ changes how well the learned behavior transfers, with some mid-range temperatures producing much higher real-environment scores than very “confident” low-temperature rollouts.

**Interpretation (high-level):** if the model is slightly “noisier” during imagination, the controller may become more robust to model errors and real-world variation.

### What *World Models (2018)* introduced

* A clear, modular template: **representation learning (V)** + **dynamics (M)** + **small controller (C)**.
* A compelling demo of **learning behaviors in imagination** and transferring them to reality.
* A practical demonstration that **unsupervised generative modeling** can produce features useful for control, even when the world model itself never sees reward.

### Limitations (then and still relevant)

These are not “gotchas”—they’re the classic failure modes that later work tries to fix:

* **Compounding error**: rollouts drift off-distribution as you imagine further into the future.
* **Partial observability**: what matters for control is often not fully visible in pixels; memory helps but is hard.
* **Model capacity & long-horizon coherence**: the paper itself notes capacity limits and hints that higher-capacity architectures could help.
* **Training data mismatch**: learning (M) from random policy rollouts can leave gaps in states encountered by a competent agent.

([Ha & Schmidhuber, 2018][1])

---

## 1) Generative latent-dynamics world models (World Models 2018, Dreamer/RSSM)

### State-space form

These are typically **stochastic state-space models**:

$$
\begin{aligned}
h_t &= \mathrm{Enc}_\phi(o_t) \\
z_t &\sim p_\theta(z_t \mid s_t, a_t) \\
s_{t+1} &= f_\theta(s_t, a_t, z_t) \\
o_t &\sim p_\theta(o_t \mid s_t)\quad \text{(decoder / observation model)}
\end{aligned}
$$

In **RSSM/Dreamer**, $s_t$ usually splits into deterministic memory $d_t$ and stochastic latent $z_t$:

$$
d_{t+1} = g_\theta(d_t, z_t, a_t), \qquad z_t \sim p_\theta(z_t \mid d_t, a_{t-1})
$$

### Training objective (variational, model learning)

They maximize an ELBO-style bound:

$$
\max_{\theta,\phi}\;\sum_t
\mathbb{E}_{q_\phi(z_t \mid o_{\le t}, a_{<t})}
\Big[
\log p_\theta(o_t \mid s_t)
+ \log p_\theta(r_t \mid s_t)
\Big]
- \beta\,\mathrm{KL}\big(q_\phi(z_t \mid \cdot)\,\|\,p_\theta(z_t \mid \cdot)\big)
$$

Key property: **explicit likelihood / reconstruction** (pixels or tokens) anchors the latent to observations.

### Planning/control usage
* “imagination rollouts” by sampling $z_t$, rolling $s_{t+k}$, and optimizing a policy/value in latent space.

**Signature**: *world model = probabilistic simulator (in latent), trained generatively.*

---

## 2) Planning-centric “value-equivalent” models (MuZero-style)

Here the model is trained to be accurate only in **decision-relevant predictions**, not to reconstruct $o_t$.

### State and dynamics

$$
\begin{aligned}
s_0 &= \mathrm{Enc}_\phi(o_{\le t}) \\
s_{k+1} &= f_\theta(s_k, a_k) \\
\hat r_k &= \rho_\theta(s_k, a_k), \quad
\hat v_k = v_\theta(s_k), \quad
\hat \pi_k = \pi_\theta(s_k)
\end{aligned}
$$

No decoder $p(o_t \mid s_t)$ is required.

### Training objective (prediction matching across search-unrolled steps)

A typical loss looks like:

$$
\min_{\theta,\phi}
\sum_{k=0}^K
\Big(
\ell_r(\hat r_k, r_{t+k})
+ \ell_v(\hat v_k, v_{t+k})
+ \ell_\pi(\hat\pi_k, \pi^{\text{target}}_{t+k})
\Big)
$$

### Interpretation under LeCun’s template

* $\mathrm{Enc}$ exists.
* $\mathrm{Pred}$ exists.
* $z_t$ is often **implicit** (stochasticity handled by policy/search + training noise), not an explicit latent.

**Signature**: *world model = abstract dynamics model optimized for planning targets.*

---

## 3) JEPA / Non-generative predictive world models (representation prediction)

These models typically avoid reconstructing pixels and instead predict **representations**.

### Representation and prediction
Let $y_t$ denote a “target” representation (often from a momentum / target encoder):

$$
\begin{aligned}
h_t &= \mathrm{Enc}_\phi(o_t) \\
\widetilde{y}_{t+\Delta} &= \mathrm{Pred}_\theta(h_{\le t}, a_{\le t}, z_{\le t})
\end{aligned}
$$

The key is: **learn to predict embeddings of future/masked parts**.

### Loss (embedding regression / contrastive / variance regularization)

A generic JEPA-ish objective:

$$
\min_{\theta,\phi}\;\sum_t
\left\|
\mathrm{Pred}_\theta(h_t, s_t, a_t, z_t) - \mathrm{StopGrad}\big( \mathrm{Enc}^{\text{tgt}}(o_{t+1}) \big)
\right\|_2^2
+ \lambda\,\mathcal{R}_{\text{anti-collapse}}
$$

Where $\mathcal{R}_{\text{anti-collapse}}$ can be:

* variance/covariance regularizers (VICReg-like),
* predictor/target asymmetry (BYOL-like),
* masking structure,
* entropy constraints, etc.

### Mapping to LeCun’s “collapse” point

Because there is no pixel likelihood, the encoder could collapse to a constant embedding unless the anti-collapse regularizer and architecture prevent it. (This is exactly the “prevent trivial solution” issue LeCun mentions.)

**Signature**: *world model = predictor in latent/embedding space; avoids full generative decoding.*

---

## 4) Internet-scale generative video models (diffusion / autoregressive token video)

These are “world-model-like” when they can be **conditioned on actions** and rolled forward consistently. Mathematically, they are sequence generative models over video tokens or pixels.

### Autoregressive form (token-based)
Let $o_t$ be discrete tokens for video/audio:

$$
p_\theta(o_{1:T}\mid c) = \prod_{t=1}^T p_\theta(o_t \mid o_{<t}, c)
$$

Sampling randomness is the $z_t$ in LeCun’s framing (random seed / sampling choice).

### Diffusion form

Diffusion learns to reverse a noising process:

$$
\min_\theta\; \mathbb{E}_{t,\epsilon}
\left[
\left\|
\epsilon - \epsilon_\theta(o_t^{\text{noised}}, t, c)
\right\|_2^2
\right]
$$

At generation time, the stochasticity is in the diffusion noise trajectory (again: $z$).

### Relation to LeCun template

* $\mathrm{Enc}$ is often a tokenizer/latent encoder.
* State $s_t$ is typically the transformer context / latent cache.
* Actions $a_t$ are often missing unless explicitly added (Genie-like interactive models add action conditioning).

**Signature**: *world model = high-fidelity generative predictor of observations; “worldness” depends on action-conditioning + coherence.*

---

## 5) Robotics-native / Vision-Language-Action world models (latent actions, inverse dynamics)

This family is basically about **learning a controllable dynamics model** when actions are missing, high-dimensional, or partially observed.

### Inverse dynamics (learn actions from video)
Given $(o_t, o_{t+1})$, infer an action (or latent action token) $\hat a_t$:

$$
\hat a_t = \mathrm{Inv}_\psi(o_t, o_{t+1})
$$

Then learn forward dynamics:

$$
s_{t+1} = f_\theta(s_t, \hat a_t, z_t)
$$

### Latent action modeling (LAM)
Instead of real robot actions, learn discrete/continuous latent actions $u_t$:

$$
u_t \sim p_\theta(u_t \mid o_t, o_{t+1})
\quad\text{and}\quad
s_{t+1} = f_\theta(s_t, u_t, z_t)
$$

### Objective (joint)

$$
\min\; \mathcal{L}_{\text{forward}}(o_{t+1}, \hat o_{t+1})
+ \alpha\,\mathcal{L}_{\text{inv}}(a_t, \hat a_t)
+ \beta\,\mathcal{R}
$$

**Signature**: *world model = action-grounded predictor; often learns action tokens from video to become “interactive.”*

---

## Canonical math template for world models

### Variables

* Observation: $o_t \in \mathcal{O}$ (pixels, video tokens, proprioception, text, etc.)
* Action: $a_t \in \mathcal{A}$ (robot control, joystick, discrete “latent action,” optional)
* Internal state (belief / memory): $s_t \in \mathcal{S}$
* Stochastic latent (uncertainty / multimodality): $z_t \in \mathcal{Z}$

### Core functions

A world model is defined by an **encoder** and a **predictor**:

$$
h_t = \mathrm{Enc}_\phi(o_t),\qquad
s_{t+1} = \mathrm{Pred}_\theta(s_t, h_t, a_t, z_t)
$$

To support multi-step rollouts under candidate actions, we unroll:

$$
s_{t+k+1} = \mathrm{Pred}_\theta(s_{t+k}, h_{t+k}, a_{t+k}, z_{t+k})
$$

### Optional decoding heads

Depending on the school, the model may include one or more heads:

* **Observation head** (generative): $\hat o_{t+1} \sim p_\theta(o_{t+1}\mid s_{t+1})$
* **Reward head**: $\hat r_{t+1} \sim p_\theta(r_{t+1}\mid s_{t+1})$
* **Value/policy heads**: $\hat v_{t+1}=v_\theta(s_{t+1}),\ \hat\pi_{t+1}=\pi_\theta(s_{t+1})$

### Training data

Most world models train from tuples/trajectories:

$$
\mathcal{D}=\{(o_t, a_t, o_{t+1}, r_t, \dots)\}
$$

Or *action-free* video: $\{(o_t, o_{t+1})\}$ (then learn latent actions).

---

## A clean mathematical taxonomy (3 “objective families”)

### Family A — Observation-likelihood / generative world models

They explicitly learn a predictive distribution over future observations:

$$
\max_{\theta,\phi}\sum_t \log p_\theta(o_{t+1}\mid o_{\le t}, a_{\le t})
$$
Often with latent variables $z_t$ and variational training (ELBO).

**Signature:** “model the world” by modeling $p(o)$ (pixels/tokens), plus dynamics.

---

### Family B — Representation-predictive (JEPA-style) world models

They predict *future representations* rather than pixels:

$$
y_{t+1}=\mathrm{Enc}^{\text{tgt}}(o_{t+1}),\quad
\hat y_{t+1}=\mathrm{Pred}_\theta(s_t,h_t,a_t,z_t)
$$

$$
\min_{\theta,\phi}\sum_t d\big(\hat y_{t+1},\mathrm{StopGrad}(y_{t+1})\big)+\lambda\,\mathcal{R}_{\text{anti-collapse}}
$$

**Signature:** no $p(o\mid s)$ decoder; collapse prevention becomes central.

---

### Family C — Task-equivalent / planning-centric world models (MuZero-style)

They do **not** try to predict $o$ at all; they predict decision-relevant quantities:

$$
s_{k+1}=f_\theta(s_k,a_k),\quad \hat r_k,\hat v_k,\hat \pi_k
$$

$$
\min\sum_{k=0}^K\ell_r(\hat r_k,r_{t+k})+\ell_v(\hat v_k,v_{t+k})+\ell_\pi(\hat\pi_k,\pi^*)
$$

**Signature:** “world model” = internal rollout model good enough for planning/search.

---

## Where does uncertainty $z_t$ live?

A useful discriminator across schools:

* **Explicit stochastic transition**: $z_t \sim p(z_t\mid s_t,a_t)$ (Dreamer/RSSM; classic latent state-space)
* **Explicit stochastic observation**: sampling noise controls $\hat o$ (diffusion/AR video)
* **Implicit stochasticity**: no explicit $z_t$; randomness via policy/search/training noise (often MuZero-style)
* **Latent action uncertainty**: $z_t$ or $u_t$ acts like an *action token* learned from video (LAM/UVA/AdaWorld)

---

## Core technical challenges (and why they're hard)

1. **Compounding error & distribution shift**
   If your policy visits states your model hasn't learned well, imagined rollouts diverge quickly.

2. **Stochasticity & uncertainty**
   Many environments are inherently stochastic; the 2018 paper bakes this in via probabilistic next-latent prediction and temperature control. ([Ha & Schmidhuber, 2018][1])

3. **Long-horizon planning with partial observability**
   Memory helps, but representing "what matters" over long horizons is nontrivial.

4. **Learning action-conditioning at scale**
   Large video-only pretraining is abundant, but action labels/interaction data are scarce—modern work tries to combine both (Dreamer 4, V-JEPA 2). ([Hafner et al., 2025][8])

5. **Evaluation**

   * "Looks right" video prediction ≠ "useful for control"
   * Conversely, MuZero can control well without predicting pixels. ([Schrittwieser et al., 2020][9])
     This makes benchmarking subtle: you often need both **prediction metrics** and **downstream control performance**.

---

## When world models tend to help most

World models are especially compelling when:

* **Environment interaction is expensive** (robotics, real-world systems)
* You want **sample efficiency** and can spend compute on learning/predicting
* You can benefit from **planning / counterfactual imagination**
* You have **lots of passive video** but limited action-labeled data (foundation-world-model direction)

They can struggle when:

* The environment is **highly chaotic/adversarial**
* Success requires **very long-horizon precise reasoning** and the model drifts
* The "right abstraction" is unclear and pixel prediction is a distraction

---

# Mapping your reading list into the template

I’ll tag each item with: **(Objective family)** + what plays the role of $(s_t, z_t, a_t)$.

## 1) Canonical foundation

* **World Models ([Ha & Schmidhuber, 2018][1])** — **(A)**. $s_t$ = RNN hidden, $z_t$ = MDN mixture component/noise; learns latent dynamics + reconstructs via VAE; controller trained on $[z_t, h_t]$.
* **Dreamer (V1–V3)** — **(A)**. $s_t$ = RSSM belief (deterministic + stochastic), $z_t$ = stochastic latent; trains by ELBO-style reconstruction + reward/value; learns policy “in imagination.”
* **Dreamer 4** — **(A)**. Same template; scaled world-model + imagination training in more complex settings (your “new peak” successor).

## 2) JEPA school (non-generative predictive)

* **V-JEPA (2024)** — **(B)**. Predict masked/future *embeddings*; $s_t$ is transformer context/belief; $z_t$ often implicit; no pixel likelihood.
* **V-JEPA 2 (2025)** — **(B)**. Same, but explicitly positioned for physical reasoning/planning with representation prediction.
* **What Drives Success in Physical Planning with JEPA-WMs? (Dec 2025)** — **(B → analysis)**. Not a new model family; it’s the “mechanism + eval” anchor explaining when **(B)** works.
* **LeJEPA (2025)** — **(B)**. Adds stronger anti-collapse / structure constraints and better planning behavior in JEPA-style worlds.

## 3) Video-as-world-simulators (generative)

* **Sora / Sora 2** — **(A)** in the *observation-model* sense: learns $p(o_{t+1:t+K}\mid o_{\le t}, c)$ with sampling noise as $z$. “World-model-ness” depends on controllable/action-conditioned rollouts; Sora 2 is explicitly positioned as higher fidelity + audio.  *(If you want, we can cite the exact OpenAI page again in this section.)*
* **Genie 3 (Aug 2025)** — **(A, interactive)**. Generative video world model with user interactivity; $a_t$ is effectively present (control inputs), $z_t$ = generation stochasticity.
* **Cosmos (NVIDIA, Jan 2025)** — **(A)**. “World foundation model” framing for physical AI; typically generative video backbone + control adaptation.
* **Commercial video models (Runway/Luma/Kling)** — generally **(A)**; useful as “capability pressure,” but cite sparingly unless you need benchmarking claims.

## 4) Embodied world models (robotics-native / VLA)

* **1X World Model (2024–2025)** — typically **(A or B)** depending on whether they decode pixels or predict structured latents; the key is $a_t$ is robot action and $s_t$ is embodied belief about a home scene.
* **Cosmos Policy (Jan 2026)** — **(A→policy)**. Start from a generative world model, then post-train into visuomotor control (policy/value/planning heads become primary).
* **AdaWorld (ICML 2025)** — **(A with learned actions)**. Learns **latent actions from video** then trains an autoregressive world model conditioned on those latent actions; adapts to new action spaces efficiently. ([Zhu et al., 2025][16])
* **TesserAct: Learning 4D Embodied World Models (2025)** — often **(A or B)** with 4D (3D+time) state; $s_t$ becomes a 4D scene representation for navigation/planning.

## 5) Critical analysis & philosophy

* **“Is Sora a World Simulator? A Comprehensive Survey”** — meta-taxonomy; use it to separate “video generator” vs “action-conditioned counterfactual simulator.” (Mostly about defining what qualifies as a world model.)
* **“Sora and V-JEPA Have Not Learned The Complete Real World Model” (Zhang, 2024)** — philosophical critique; helpful for your “limits: causality/agency” section.
* **“When do World Models Successfully Learn Dynamical Systems?”** — theory anchor: identifiability/observability conditions under which learning dynamics is possible.

## Other notable mentions (as mathematical “sub-branches”)

* **CWM (Code World Model)** — **(A/C hybrid)**: “world” is a programmatic environment; $o_t$ = text/state snapshots; $a_t$ = tool/code actions; strong for agentic planning in executable state spaces.
* **PSI (Probabilistic Structure Integration)** — **(A/B hybrid)**: introduces intermediate “structures” between pixels and actions; $z_t$ often corresponds to structured latent variables.
* **PAN (Physical, Agentic, Nested)** — **(A with interactivity emphasis)**: argues for nested/hierarchical state $s_t$ and interactive rollouts.
* **D4RT (4D representations)** — **(B-ish representation)**: emphasizes fast 4D scene/state representations; $s_t$ is explicitly geometric (useful as “geometry-first world models”).

---

## Latent actions / inverse dynamics cluster (your second block)

These are best described as: **learn $a_t$ (or a proxy) from video**, then learn forward dynamics for planning/control.

* **VPT (2022)** — inverse-dynamics flavored: $a_t \approx \mathrm{Inv}(o_t, o_{t+1})$; then train a policy on inferred actions (“act by watching”).
* **Watch and Learn: Learning to Use Computers from Online Videos (2025)** — same template with UI trajectories; $a_t$ is “UI action token,” $s_t$ is UI state.
* **Predictive Inverse Dynamics Models… (2024)** — $a_t$ is supervised (or weakly supervised); model predicts action sequences from observation change.

### UVA (Unified Video-Action latents)

* **Unified Video Action Model (Feb 2025)** — jointly learns video prediction + action inference in a shared latent space: $s_t$ supports both forward rollout and action decoding.

### LAM (latent action modeling)

* **LAPA (ICLR 2025)** — learn latent action tokens $u_t$ from video; use them as controllable “action space” for rollouts/policies.
* **villa-X (2025)** — explicit ViLLA framework: latent action model (IDM + forward dynamics) + actor; grounds $u_t$ with forward dynamics including proprioception. ([Yu et al., 2025][17])

### Geometric/flow-based connection

* **Track2Act (2024)** — replaces "predict pixels" with "predict point tracks" (a structured plan); converts tracks into robot transforms + residual policy. ([Dass et al., 2024][18])
* **Mask2Act (BMVC 2025)** — predicts future **object masks** as latent plans; avoids full RGB generation and uses mask dynamics to guide policy learning. ([Schmid et al., 2025][19])

---

## If you want this to be *even more plug-and-play*

Tell me your intended section headings (e.g., **“Definition,” “Taxonomy,” “Evaluation,” “Embodiment,” “Future directions”**), and I’ll rewrite the above into a polished 1–1.5 page intro with consistent terminology and a compact “classification table” that matches your paper’s narrative arc.


Yep — “policy” is a **second axis** orthogonal to the model objective (generative vs JEPA vs planning-centric). You can treat it as: **how is behavior represented, trained, and coupled to the model**?

Below is a **paste-ready taxonomy**: policy *presence* (none / implicit / explicit) and policy–model *interaction modes* (planning, imagination training, distillation, etc.), all in math aligned with your Enc/Pred terms.

---

## Policy axis: explicit vs implicit vs none

Let a policy be $\pi_\psi(a_t\mid \cdot)$ producing actions $a_t$.

### A) No policy (pure world model)

These works learn $\mathrm{Enc}$ and $\mathrm{Pred}$ (and maybe $\mathrm{Dec}$) but **do not** define a controller:

$$
h_t=\mathrm{Enc}(o_t), \quad s_{t+1}=\mathrm{Pred}(s_t,h_t,a_t,z_t)
$$
but there is no learned $\pi$; actions may be absent (video-only), or provided only as conditioning signals.

**Typical**: internet-scale video generators, some JEPA pretraining papers, pure representation/forecasting models.

---

### B) Implicit policy (planner/search “is” the policy)

There is no standalone neural policy $\pi_\psi$ that outputs actions directly. Instead, action selection is performed by an optimizer/search that queries the model.

You can write the “policy” as an operator:
$$
a_t = \Pi(\mathcal{M}_\theta, s_t) \quad \text{(planner uses model } \mathcal{M}_\theta\text{)}
$$

Example: **MPC / CEM** (model predictive control). Choose an action sequence that maximizes predicted return:

$$
a_{t:t+H-1}^* =
\arg\max_{a_{t:t+H-1}}
\mathbb{E}\Big[\sum_{k=0}^{H-1} \gamma^k\, \hat r_{t+k}\Big]
$$

where $\hat r_{t+k}=R_\theta(s_{t+k},a_{t+k})$ and $s_{t+k+1}=\mathrm{Pred}_\theta(s_{t+k},h_{t+k},a_{t+k},z_{t+k})$.

Then execute $a_t=a_t^*$ (first action), replan next step.

**Typical**: PlaNet-style planning, PETS/MBPO-type MPC loops, TD-MPC family.

---

### C) Explicit policy (a learned controller exists)

There is an explicit $\pi_\psi$ trained to output actions:

$$
a_t \sim \pi_\psi(a_t\mid s_t \text{ or } (h_{\le t}, s_t))
$$

How that policy is trained / coupled to the world model is the key “interaction mode” (next section).

**Typical**: Dreamer, World Models 2018 (controller), most robotics policies, MuZero (policy head + search).

---

# Policy–world model interaction modes (how they couple)

This is the dimension that really differentiates “schools in practice.” You can define $\mathcal{M}_\theta$ as the world model ($\mathrm{Enc}$ + $\mathrm{Pred}$ + optional heads).

## 1) Model-predictive control (implicit policy via planning)

**No explicit policy required** (or an optional warm-start policy).

* Action chosen by:
  $$
  a_t = \arg\max_{a} \mathbb{E}\left[\sum_{k=0}^{H-1}\gamma^k \hat r_{t+k}\right]
  $$
  using rollouts in $\mathcal{M}_\theta$.
* If a policy exists, it’s often used as a proposal distribution:
  $$
  a_{t:t+H-1} \sim \pi_\psi(\cdot\mid s_t) \quad \text{then refine via CEM/MPC}
  $$

**Where it shows up**: PlaNet, PETS, TD-MPC2.

---

## 2) Imagination training (explicit policy learns inside the model)

This is the Dreamer / “World Models 2018 dreams” coupling.

* World model learns from real trajectories (\mathcal{D}).
* Policy learns from *imagined* rollouts:
  $$
  s_{t+1}=\mathrm{Pred}_\theta(s_t,h_t,a_t,z_t),\quad a_t\sim\pi_\psi(\cdot\mid s_t)
  $$

Policy objective (generic actor-critic in latent rollouts):
$$
\max_\psi\; \mathbb{E}_{\pi_\psi,\mathcal{M}_\theta}\Big[\sum_{k=0}^{H-1}\gamma^k \hat r_{t+k}\Big]
$$
with gradients possibly flowing through the model (Dreamer-style) or not (evolution strategies / black-box like CMA-ES in World Models 2018).

**Where it shows up**: Dreamer V1–V4; Ha & Schmidhuber 2018.

---

## 3) Search-augmented policy (explicit + implicit hybrid)

Search produces a better action distribution; the policy is trained to imitate or to output priors for search.

MuZero is the canonical structure:

* Model unroll produces latent states:
  $$
  s_{k+1}=f_\theta(s_k,a_k)
  $$
* Search (MCTS) yields an improved policy (\pi^{\text{MCTS}}).
* Train policy head to match:
  $$
  \min_\psi\; \ell_\pi(\pi_\psi(\cdot\mid s_k),\ \pi^{\text{MCTS}}(\cdot\mid s_k))
  $$
  plus value and reward losses.

**Where it shows up**: MuZero, AlphaZero-style variants, modern “search + learned model” agents.

---

## 4) Distillation / post-training (world model → policy)

A world model is pretrained (often on video), then adapted into a policy.

Two common forms:

### 4a) World-model as a simulator for offline RL / BC

Generate synthetic trajectories and train policy:
$$
\mathcal{D}_{\text{sim}} = \text{Rollout}(\mathcal{M}_\theta,\ \pi_\psi)
$$

$$
\min_\psi\; \mathbb{E}_{(s,a)\sim \mathcal{D}_{\text{sim}}}\big[-\log \pi_\psi(a\mid s)\big]
$$

### 4b) Direct fine-tuning into action heads

Add an action head and train with supervised/BC loss:
$$
a_t \approx \pi_\psi(\cdot\mid s_t)
\quad,\quad
\min_\psi\ \sum_t \|a_t-\hat a_t\|_2^2 \ \text{or}\ -\log \pi_\psi(a_t\mid s_t)
$$

**Where it shows up**: “video world model → robotics policy” efforts (e.g., Cosmos Policy style), VLA models.

---

## 5) Latent-action coupling (policy over learned action tokens)

When true actions aren’t available (internet video), the “policy” is over latent actions $u_t$:

* Infer latent action:
  $$
  u_t \sim q_\psi(u_t\mid o_t,o_{t+1})
  $$
* Forward dynamics uses $u_t$:
  $$
  s_{t+1}=f_\theta(s_t,u_t,z_t)
  $$
* A controller learns to choose $u_t$, then map to real robot actions via a decoder:
  $$
  u_t \sim \pi_\omega(u_t\mid s_t),\quad a_t = g_\eta(u_t, s_t)
  $$

**Where it shows up**: LAPA, UVA, AdaWorld, inverse-dynamics-based robotics pretraining.

---

# “Policy presence” by school (your literature groups)

## JEPA pretraining papers

* **Policy**: usually **none** in the pretraining work; policy enters later as a downstream planner or controller.
* **Interaction**: typically (2) imagination training *in representation space* or (1)/(4) planning/distillation on top of learned representations.

## Generative latent dynamics (Dreamer / World Models 2018)

* **Policy**: **explicit** (controller/policy/value).
* **Interaction**: (2) imagination training is the hallmark; sometimes hybrid with (1) planning.

## Planning-centric (MuZero)

* **Policy**: **explicit + implicit** (policy head exists, but action selection is via search).
* **Interaction**: (3) search-augmented policy, planning targets define training.

## Video diffusion/AR “world simulators” (Sora/Genie/Cosmos base)

* **Policy**: often **none** in the base generator; sometimes **implicit** via user controls; becomes **explicit** when adapted to robotics.
* **Interaction**: (4) distillation/post-training; (5) latent-action modeling for controllability; in interactive models (Genie) there’s an explicit control channel.

## Embodied robotics world models (Cosmos Policy, 1X, AdaWorld, etc.)

* **Policy**: usually **explicit** (robot must act), sometimes with latent action layers.
* **Interaction**: (4) post-training + (1) MPC/planning + (5) latent-action coupling are common.

---

# One sentence you can use in your paper

> The “policy axis” distinguishes **whether behavior is (i) absent, (ii) implicit as a planner/search over the model, or (iii) explicit as a learned controller**, and how it couples to the world model through **planning (MPC/MCTS), imagination training, search-augmented learning, or distillation/post-training**.

---

If you want, I can produce a **2D classification table** (rows = model objective family A/B/C, columns = policy presence + coupling mode) and slot *each paper in your bib* into one cell, so your intro reads like a clean map rather than a list.



[1]: https://arxiv.org/pdf/1803.10122 "World Models"
[2]: https://arxiv.org/abs/1803.10122 "World Models"
[3]: https://arxiv.org/abs/1811.04551 "Learning Latent Dynamics for Planning from Pixels"
[4]: https://arxiv.org/abs/1912.01603 "Dream to Control: Learning Behaviors by Latent Imagination"
[5]: https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/ "Introducing Dreamer: Scalable Reinforcement Learning Using World Models"
[6]: https://arxiv.org/abs/2010.02193 "Mastering Atari with Discrete World Models"
[7]: https://arxiv.org/pdf/2301.04104v1 "Mastering Diverse Domains through World Models"
[8]: https://arxiv.org/abs/2509.24527 "Training Agents Inside of Scalable World Models"
[9]: https://arxiv.org/abs/1911.08265 "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
[10]: https://arxiv.org/pdf/1903.00374.pdf "Model-Based Reinforcement Learning for Atari (SimPLe)"
[11]: https://arxiv.org/abs/2402.15391 "Genie: Generative Interactive Environments"
[12]: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ "Genie 3: A new frontier for world models"
[13]: https://arxiv.org/abs/2301.08243 "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
[14]: https://arxiv.org/abs/2506.09985 "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
[15]: https://www.nature.com/articles/s41586-020-03051-4.pdf "Mastering Atari, Go, chess and shogi by planning with a learned model"
[16]: https://arxiv.org/abs/2503.18938 "AdaWorld: Learning Adaptable World Models with Latent Actions"
[17]: https://arxiv.org/abs/2507.23682 "villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models"
[18]: https://arxiv.org/abs/2405.01527 "Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation"
[19]: https://bmvc2025.bmva.org/proceedings/124/ "Mask2Act: Predictive Multi-Object Tracking as Video Pre-Training"
[20]: https://x.com/ylecun/status/1759933365241921817 "Yann LeCun on world models (X post)"

