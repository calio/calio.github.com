---
title: "What are World Models"
date: 2026-02-07
author: Jiale Zhi
---

> Disclaimer: this post is far from complete. It reflects my personal view of a small slice of a much bigger and fast-moving literature.

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


## What can a "world model" predict?

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


## Major "families" of world-model approaches

It’s useful to read the literature as a few recurring families, rather than as a single line of progress.

### 1) World Models (2018): compress → predict → control

*World Models* (Ha & Schmidhuber, 2018) popularized a clean, modular recipe—**compress → predict → control**—and the idea of training a controller *inside the model's own "dream"* before deploying in the real environment. ([Ha & Schmidhuber, 2018][1])

### 2) Latent dynamics models for planning: PlaNet

**PlaNet** (Hafner et al.) is an early landmark for planning directly in a learned latent space. It uses a latent dynamics model with **both deterministic and stochastic** transition components and performs online planning in latent space. ([Hafner et al., 2019][3])


### 3) "Imagination training" with actor-critic: Dreamer → DreamerV2/V3 → Dreamer 4

**Dreamer** made “learn in latent imagination” a core, scalable strategy: it learns behaviors **purely by latent imagination** and propagates value gradients back through imagined trajectories. ([Hafner et al., 2020][4])
Google’s blog summary also describes Dreamer as a model-based pipeline of (1) learning the world model, (2) learning behaviors from predictions, (3) acting to collect more experience. ([Hafner et al., 2020][4])

After the original Dreamer paper, there are a few follow up work:

* **DreamerV2** introduced discrete world-model representations and is positioned as achieving human-level Atari performance by learning behaviors within a separately trained world model. ([Hafner et al., 2021][6])
* **DreamerV3** aims at *robustness across diverse domains with fixed hyperparameters*, still centered on learning a world model for imagination training. ([Hafner et al., 2023][7])
* **Dreamer 4 (2025)** pushes scale and complexity: it frames “imagination training” in a more scalable setting (Minecraft), emphasizing accurate object interactions and even reporting a diamond-acquisition challenge from offline data. ([Hafner et al., 2025][8])


### 4) Planning-centric, non-pixel models: MuZero

**MuZero** is sometimes described as a world model that doesn’t try to reconstruct the world. Instead, it learns a model that predicts the quantities most relevant to planning—**reward, value, and policy**—and uses tree search. ([Schrittwieser et al., 2020][9])

This is an important conceptual branch:

* A “world model” doesn’t have to be a realistic simulator in pixel space.
* It can be an *abstract dynamics model* that is only required to be accurate for decision-making.

### 5) Video-prediction world models for sample efficiency: SimPLe and beyond

**SimPLe** (Kaiser et al.) is an example of using learned video prediction models as simulators to train policies with limited environment interaction (often discussed in the Atari 100k setting). ([Kaiser et al., 2020][10])
This line emphasizes the classic tradeoff: pixel-predictive models can be expensive and brittle, but they can reduce real interaction needs if made good enough.

### 6) Large-scale interactive environment generation: Genie series

More recently, the term "world model" is also used for large models that generate **interactive environments** from broad data sources.

* **Genie (2024)** is presented as a "generative interactive environment" trained unsupervised from unlabeled Internet videos, with a tokenizer, an autoregressive dynamics model, and a latent action model at 11B parameters. ([Bruce et al., 2024][11])
* **Genie 2 (2024)** extends this to generate a richer variety of 3D environments with better consistency and control.
* **Genie 3 (2025)** is announced as a general-purpose world model that can generate diverse interactive environments navigable in real time (24 FPS) for a few minutes at 720p. ([Google DeepMind, 2025][12])

This is a shift from “world model for one RL benchmark” to “world model as a general interactive generator.”

### 7) Representation-predictive (non-generative) world models: JEPA / V-JEPA 2

Another modern branch argues you don’t always need to predict pixels; you can predict *representations*.

* **I-JEPA** proposes a joint-embedding predictive architecture for learning semantic image representations by predicting representations of masked regions (non-generative). ([Assran et al., 2023][13])
* **V-JEPA 2 (2025)** extends this philosophy to video and positions itself explicitly as a self-supervised video model enabling understanding/prediction/planning in the physical world—pretraining action-free on **over 1 million hours** of video/images and then incorporating a small amount of interaction data. ([Bardes et al., 2025][14])

This family is often motivated by: *prediction at the pixel level forces the model to care about irrelevant detail*, while representation prediction can focus on task-relevant structure.


## *World Models* ([Ha & Schmidhuber, 2018][1])

*World Models* (Ha & Schmidhuber, 2018) introduces a clean, modular recipe for learning a latent simulator and using it for control:

**compress → predict → control**.

![World Models (2018) overall architecture: VAE (V) compresses frames to latents, MDN-RNN (M) predicts next-latent distributions, controller (C) acts on latent + RNN hidden state](/assets/images/world-models/world-models-2018-architecture.png)

*Figure 1: Overall architecture (VAE $\to$ MDN-RNN $\to$ controller) (source: [Ha & Schmidhuber, 2018][1]).*

At each time step $t$, the observation $o_t$ is encoded by $V$ into a compact latent $z_t$. The dynamics model $M$ maintains a recurrent hidden state $h_t$ (its “memory”) and updates it using the current latent and action, e.g. $h_{t+1}=\mathrm{RNN}(h_t, z_t, a_t)$; from the updated state it produces a distribution over the next latent $p(z_{t+1}\mid h_{t+1})$ for “dreamed” rollouts. The controller $C$ takes the concatenation $[z_t, h_t]$ and outputs an action $a_t$ to the environment. The real environment produces the next observation $o_{t+1}$, and the loop repeats. ([Ha & Schmidhuber, 2018][1])

### State and components (VAE + MDN-RNN + controller)

The model is explicitly modular:

* **V (Vision, VAE)**: compress observations to a latent.
  $$z_t = \mathrm{Enc}(o_t),\; \hat o_t = \mathrm{Dec}(z_t)$$
  In their setup, $o_t$ is a 64×64 RGB frame and $z_t$ is a low-dimensional latent (e.g. $\mathbb{R}^{32}$ for CarRacing, $\mathbb{R}^{64}$ for Doom).

* **M (Memory/dynamics, MDN-RNN)**: predict a *distribution* over next latents.
  $$h_{t+1}=\mathrm{RNN}(h_t, z_t, a_t),\; p(z_{t+1}\mid h_{t+1})=\mathrm{MDN}(h_{t+1})$$
  Here, an **MDN (mixture density network)** means we model the next-latent density as a **mixture of Gaussians** (so the network outputs mixture weights, means, variances).
  Sampling uses a **temperature $\tau$** to control how “stochastic” the imagined rollouts are (loosely: higher $\tau$ → more randomness → harder-to-exploit dreams).

* **C (Controller / policy)**: a deliberately tiny **single-layer linear** policy that maps the concatenated input $[z_t, h_t]$ directly to the action at each time step:
  $$a_t = W_c\,[z_t\; h_t] + b_c$$
  Here, $W_c$ and $b_c$ are the weight matrix and bias vector that map the latent+memory features into the action vector.

### Mapping to LeCun’s Enc/Pred scaffold

Using notation ($o_t, a_t, \mathrm{Enc}, \mathrm{Pred}, s_t, z_t$), the 2018 architecture can be read as a concrete instantiation. With a slight abuse of notation, I’ll reuse $z_t$ to denote the VAE latent code, and I’ll use $\xi_t$ for the MDN sampling randomness.

* **Observation $o_t$**: an observed (rendered) image frame.
* **Action $a_t$**: environment controls (e.g., steering/throttle/brake in CarRacing).

* **Encoder $h_t = \mathrm{Enc}_\phi(o_t)$**: corresponds to the **VAE encoder** output (latent code):
  $$z_t = \mathrm{Enc}^{\text{VAE}}_\phi(o_t)$$

* **World-state $s_t$**: corresponds to the **RNN hidden state** (the paper’s “memory”):
  $$s_{t+1} = \mathrm{RNN}_\theta(s_t, z_t, a_t)$$

* **Predictor $\mathrm{Pred}_\theta$**: the MDN-RNN is effectively the transition model that (i) updates $s_t$ and (ii) outputs a distribution over next latents:
  $$s_{t+1}=\mathrm{RNN}_\theta(s_t, z_t, a_t),\qquad p_\theta(z_{t+1}\mid s_{t+1}) = \mathrm{MDN}_\theta(s_{t+1})$$

* **Optional decoder**: $\hat o_t = \mathrm{Dec}(z_t)$ exists for visualization/reconstruction, but control is trained on latents + memory.


### Training objective (model learning)

This paper’s training is best understood as a **staged procedure**: learn a latent representation ($V$), learn latent dynamics ($M$), then learn a controller ($C$) that exploits the learned dynamics.

Concretely (CarRacing), the pipeline is:

1. Collect 10,000 rollouts from a random policy.
2. Train $V$ (VAE) to encode frames into $z_t\in\mathbb{R}^{32}$.
3. Train $M$ (MDN-RNN) to model $p(z_{t+1}\mid a_t, z_t, h_t)$.
4. Define the controller $C$ as a tiny linear policy, $a_t = W_c\,[z_t\; h_t] + b_c$.
5. Use CMA-ES to optimize $(W_c, b_c)$ to maximize expected cumulative reward.

Notes on what’s being optimized:

* **$V$ (VAE)** is trained to reconstruct frames while regularizing the latent distribution (so $z_t$ stays compact and smooth).
* **$M$ (MDN-RNN)** is trained by maximum likelihood: given $(z_t, a_t)$ and memory, assign high probability to the observed $z_{t+1}$ (with a mixture head to represent multimodal next-latents).
* **$C$** is not trained by backpropagating through $M$ here; it’s optimized as a black-box policy (CMA-ES) evaluated inside the dreamed rollouts.

### Planning/control usage (the “dream” loop)

The key demo is that you can train a controller *inside the learned model’s own “dream”* and then transfer it to the real environment.

Control is learned *through the model*, not by differentiable planning:

* Collect rollouts (random policy in the paper).
* Fit $V$ and $M$ from data.
* Roll out an imagined latent environment by sampling $z_{t+1}$ from $M$ (with $\tau$) and feeding it back into the RNN.
  - You can decode $z_t$ back to pixels for visualization, but control only needs latents.
* Optimize **C** inside the dreamed environment (they use **CMA-ES**, i.e. black-box optimization), then deploy C in the real environment.

The temperature result is a nice early lesson: making imagination slightly “noisier” can improve transfer by forcing robustness to model error.



## 1) Generative latent-dynamics world models (World Models 2018, PlaNet/RSSM, Dreamer)

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

(Here $h_t$ is just the encoder features of $o_t$.)

> Note: in the World Models (2018) deep dive, $h_t$ was an RNN hidden state (“memory”). In this family-level template, $h_t$ is *encoder features*. Below I’ll write the World Models RNN memory as $m_t$.

In **RSSM/Dreamer**, $s_t$ usually splits into deterministic memory $d_t$ and stochastic latent $z_t$:

$$
d_{t+1} = g_\theta(d_t, z_t, a_t), \qquad z_t \sim p_\theta(z_t \mid d_t, a_{t-1})
$$

### Mapping to LeCun’s $\mathrm{Enc}/\mathrm{Pred}$ scaffold (RSSM, PlaNet, Dreamer)

In LeCun’s notation, we have an encoder plus a predictor:

* $h_t = \mathrm{Enc}_\phi(o_t)$
* $s_{t+1} = \mathrm{Pred}_\theta(s_t, h_t, a_t, z_t)$

RSSM-based methods (PlaNet, Dreamer) fit this directly by taking the LeCun “world-state” $s_t$ to be a **belief state** with a deterministic part and a stochastic part:

* $s_t = (d_t, z_t)$ where $d_t$ is memory and $z_t$ is uncertainty.

One predictor step is then (conceptually) “update memory, then sample uncertainty”:

* $d_{t+1}=g_\theta(d_t, z_t, a_t)$
* $z_{t+1}\sim p_\theta(z_{t+1}\mid d_{t+1})$

During training there is also a posterior (filter) that infers $z_t$ from $(h_t,d_t)$; I’ll skip ELBO details.

Finally, **how actions are chosen is a separate layer on top of the world model**:

* **PlaNet**: choose $a_t$ by planning/search (MPC/CEM) that queries $\mathrm{Pred}_\theta$.
* **Dreamer**: learn an actor $\pi_\psi(a_t\mid s_t)$ inside imagined rollouts, then act with one forward pass.

### Comparison (World Models 2018 vs PlaNet vs Dreamer)

These three are closely related in *spirit* (latent rollouts for control), but they differ along two axes:

* **How the latent is learned** (separate VAE vs joint latent dynamics inference), and
* **How actions are chosen** (tiny controller vs online planning vs actor-critic).

| Aspect | World Models (2018) | PlaNet (2019) | Dreamer (2020) |
|---|---|---|---|
| Representation / encoder | Train a VAE; encode each frame $o_t\to z_t$ (then dynamics models sequences of $z$). | CNN encodes $o_t\to h_t$; RSSM learns posterior inference for $z_t$ using history. | Same as PlaNet (RSSM), but behavior learning is integrated into the loop. |
| State that is rolled forward | Two-part: stochastic code $z_t$ + RNN memory $m_t$. | $s_t=(d_t,z_t)$ (deterministic + stochastic). | $s_t=(d_t,z_t)$ (deterministic + stochastic). |
| Transition model | MDN-RNN: $m_{t+1}=\mathrm{RNN}(m_t,z_t,a_t)$ and a distribution for $z_{t+1}$. | RSSM prior: $d_{t+1}=g(d_t,z_t,a_t)$; $z_{t+1}\sim p(z_{t+1}\mid d_{t+1})$. | Same RSSM transition; designed to support long imagined rollouts for learning. |
| What the model predicts | Next-latent distribution; decoder mainly for reconstruction/visualization. | Latent dynamics + reward; plus observation reconstruction/features to keep state grounded. | Same, plus value/policy-related heads to support actor-critic learning in imagination. |
| How actions are generated | Explicit tiny controller $a_t=C([z_t,m_t])$, tuned by CMA-ES (black-box). | Online MPC: search over action sequences with CEM; execute first action; replan. | Learned actor $\pi_\psi(a_t\mid s_t)$ trained on imagined rollouts; fast action at runtime. |
| Behavior learning signal | Evolution strategy over controller params. | Planning objective (predicted return) solved at inference time. | Actor-critic gradients through imagined trajectories (plus value bootstrapping). |
| Runtime compute | Low. | High (many model rollouts per env step). | Low (one actor forward pass; optional value). |

### Quick takeaways

* **Same conceptual backbone**: stochastic latent dynamics + latent rollouts for decision-making.
* **Key tradeoff**: online planning (compute-heavy, strong per-step optimization) vs learned actor (compute-light, amortized control).
* **What “latent space” means differs**: VAE codes (World Models 2018) vs a belief state learned jointly with dynamics and reward (RSSM).

### PlaNet (2019): learn an RSSM, then plan with MPC

PlaNet ([Hafner et al., 2019][3]) is the planning-first instantiation of the RSSM idea:

* **Model**: learn an RSSM belief state $s_t=(d_t,z_t)$ from pixels and predict rewards from $s_t$.
* **Action selection**: use **online MPC** (typically CEM) to choose the next action by rolling the RSSM forward under candidate action sequences.

The key conceptual point: in PlaNet, **the planner *is* the policy**.

### Dreamer (2020): replace online planning with a learned actor-critic

Dreamer ([Hafner et al., 2020][4]) keeps the RSSM world model but changes the control layer:

* **Instead of planning with CEM every step**, train an explicit actor $\pi_\psi(a_t\mid s_t)$ and value function using imagined rollouts inside the RSSM.
* At runtime, actions come from the actor (amortized control), which is much cheaper than per-step planning.

**Signature**: *world model = probabilistic simulator (in latent), trained generatively.*

---

## 2) Model-based planning with “value-equivalent” latent dynamics (MuZero-style)

MuZero is best viewed as **model-based planning** rather than a (generative) “world model”: it learns an internal dynamics that is accurate only for **decision-relevant predictions** (reward / value / policy), without trying to reconstruct observations.

### State and dynamics

$$
\begin{aligned}
s_0 &= h_\phi(o_{\le t}) \\
(\hat r_{k+1},\, s_{k+1}) &= g_\theta(s_k, a_{t+k}) \\
(\hat \pi_k,\, \hat v_k) &= f_\theta(s_k)
\end{aligned}
$$

Interpretation:

* $s_k$ is a **latent planning state** (deterministic in the original MuZero formulation).
* $g_\theta$ is the learned **dynamics** (state transition + reward prediction).
* $f_\theta$ is the learned **prediction head** (policy prior + value prediction).

No decoder $p(o_t \mid s_t)$ is required; the model is *not* trained to be a good pixel simulator.

> Terminology note: $k$ counts **search/unroll steps** inside the model, not environment time steps.

### Acting: search is the policy

At inference time, MuZero does **MCTS in latent space**:

* Start from $s_0=h_\phi(o_{\le t})$.
* Expand candidate action sequences by repeatedly applying $g_\theta$.
* Use $f_\theta(s_k)$ to provide a policy prior $\hat\pi_k$ and a leaf value $\hat v_k$.
* Pick the real environment action $a_t$ from the root visit counts (or an equivalent search policy).

So while $f_\theta$ outputs a policy prior, the *executed* policy is best thought of as:
*a search procedure guided by learned priors and values.*

### Training objective (prediction matching across search-unrolled steps)

Training matches the model’s unrolled predictions to targets derived from real experience and search:

$$
\min_{\theta,\phi}
\sum_{k=0}^K
\Big(
\ell_r(\hat r_{k+1}, r^{\text{target}}_{t+k+1})
+ \ell_v(\hat v_k, v^{\text{target}}_{t+k})
+ \ell_\pi(\hat\pi_k, \pi^{\text{target}}_{t+k})
\Big)
$$

Concretely:

* $\pi^{\text{target}}$ is typically the **MCTS-improved policy** (search visit distribution).
* $v^{\text{target}}$ is an **$n$-step bootstrapped return** (often with value bootstrap at the end of the unroll).
* $r^{\text{target}}$ is the observed reward along the sampled real trajectory.

### Mapping to LeCun’s $\mathrm{Enc}/\mathrm{Pred}$ template

MuZero fits LeCun’s scaffold cleanly if we treat the latent state as the “world state”:

* **Encoder**: $h_\phi$ plays the role of $\mathrm{Enc}$ and produces the internal state $s_0$ from observations/history.
* **Predictor**: $\mathrm{Pred}$ is split into
  * a **dynamics predictor** $g_\theta$ (next-state + reward), and
  * a **readout** $f_\theta$ (policy prior + value).
* **Latent noise $z_t$**: usually **not explicit** (the model is typically deterministic; uncertainty is handled implicitly by the policy/value heads + search and the data distribution). You *can* make MuZero stochastic, but it’s not required for the core idea.

**Signature**: *model-based planning = learned latent dynamics optimized for search targets (not observation generation).*

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
limitations

World Models

### Takeaways and limits

**What it introduced:** a concrete template (learn representations + learn dynamics + keep the controller small) and a compelling “train in imagination” story that later work (PlaNet/Dreamer) made scalable.

**Limitations:** compounding error under rollout, partial observability, capacity/long-horizon coherence, and data mismatch when the world model is trained on trajectories unlike those visited by a competent policy.

**Signature**: *world model = generative latent simulator (VAE + probabilistic dynamics), used as a training substrate for control.*

([Ha & Schmidhuber, 2018][1])

---


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

