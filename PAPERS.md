# Awesome LLM Dynamics: the three clocks reading list

Papers that treat transformer language models as nonlinear dynamical systems, grouped by the three clocks of
*Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation*.
Every arXiv entry was checked by ID against the arXiv API; non-arXiv items were checked by DOI.

Last updated November 2026.

Pull requests with new papers are welcome. Please add one line: title, link, authors, year, and one sentence on why it matters.

## Clock 1: depth as time (47)

- [Stochastic Scaling Limits and Synchronization by Noise in Deep Transformer Models](https://arxiv.org/abs/2604.26898), Agazzi et al. 2026. Finite transformers with MLPs converge to a stochastic particle system; synchronization by noise.
- [Spectral Selection in Symmetric Self-Attention Dynamics](https://arxiv.org/abs/2604.26085), Kuehn & Yoon 2026. Spectral mode selection in symmetric self-attention dynamics.
- [Kinetic theory for Transformers and the lost-in-the-middle phenomenon](https://arxiv.org/abs/2605.09213), Duerinckx et al. 2026. Causal attention as non-exchangeable particles; links clustering to lost-in-the-middle.
- [The Role of Feed-Forward Layers in Transformer Dynamics](https://arxiv.org/abs/2609.36230), Maranzatto et al. 2026. Feed-forward layers as control steering token dynamics.
- [On the Diverse Dynamical Behaviors Arising in Deep Linear Transformers](https://arxiv.org/abs/2607.18584), Li et al. 2026. Deep linear transformers reduce to Kuramoto-type models; diverse behaviors.
- [Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology](https://arxiv.org/abs/2605.14258), Fernando & Guitchounts 2026. Full Jacobian eigendecomposition of residual stream in production LLMs.
- [A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955), Ostrow et al. 2026. Theory of the Jacobian lens for hidden-state features.
- [Parcae: Scaling Laws For Stable Looped Language Models](https://arxiv.org/abs/2604.12946), Prairie et al. 2026. Stability and scaling laws for looped (recurrent-depth) LMs.
- [Transformer Dynamics: A neuroscientific approach to interpretability of large language models](https://arxiv.org/abs/2502.12131), Fernando & Guitchounts 2025. Residual stream analyzed with neuroscience dynamics tools.
- [A Unified Perspective on the Dynamics of Deep Transformers](https://arxiv.org/abs/2501.18322), Castin et al. 2025. Unified PDE (Vlasov) view of attention dynamics across attention variants; depth only.
- [Dynamic metastability in the self-attention model](https://arxiv.org/abs/2410.06833), Geshkovski et al. 2024. Long-lived metastable multi-cluster states in self-attention dynamics.
- [Clustering in Causal Attention Masking](https://arxiv.org/abs/2411.04990), Karagodin et al. 2024. Clustering under causal masking (the LLM case).
- [On the Role of Attention Masks and LayerNorm in Transformers](https://arxiv.org/abs/2405.18781), Wu et al. 2024. Attention masks and LayerNorm decide whether tokens collapse (Jadbabaie).
- [Geometric Dynamics of Signal Propagation Predict Trainability of Transformers](https://arxiv.org/abs/2403.02579), Cowsik et al. 2024. Geometric signal-propagation theory predicting transformer trainability.
- [The Remarkable Robustness of LLMs: Stages of Inference?](https://arxiv.org/abs/2406.19384), Lad et al. 2024. Layer deletions/swaps reveal phases of inference.
- [Lines of Thought in Large Language Models](https://arxiv.org/abs/2410.01545), Sarfati et al. 2024. Token trajectories through depth as low-dimensional stochastic paths.
- [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762), Sun et al. 2024. Huge fixed activations acting as biases; attention sinks link.
- [A mathematical perspective on Transformers](https://arxiv.org/abs/2312.10794), Geshkovski et al. 2023. Tokens as interacting particles on the sphere; clustering theory.
- [The emergence of clusters in self-attention dynamics](https://arxiv.org/abs/2305.05465), Geshkovski et al. 2023. Self-attention dynamics converge to clusters as depth grows.
- [Demystifying Oversmoothing in Attention-Based Graph Neural Networks](https://arxiv.org/abs/2305.16102), Wu et al. 2023. Oversmoothing in attention-based GNNs via products of inhomogeneous matrices (Jadbabaie).
- [Finite-time Lyapunov exponents of deep neural networks](https://arxiv.org/abs/2306.12548), Storm et al. 2023. FTLEs of deep nets map input-space structure; template for our Exp. 1.
- [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112), Belrose et al. 2023. Iterative refinement of predictions across layers.
- [The geometry of hidden representations of large transformer models](https://arxiv.org/abs/2302.00294), Valeriani et al. 2023. Intrinsic dimension of hidden states through depth.
- [Do Residual Neural Networks discretize Neural Ordinary Differential Equations?](https://arxiv.org/abs/2205.14612), Sander et al. 2022. When ResNets do and do not converge to an ODE limit.
- [Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse](https://arxiv.org/abs/2206.03126), Noci et al. 2022. Signal propagation and rank collapse in transformers.
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555), Wang et al. 2022. Residual rescaling to train 1000-layer transformers.
- [Momentum Residual Neural Networks](https://arxiv.org/abs/2102.07870), Sander et al. 2021. Invertible second-order (momentum) ResNets.
- [Sinkformers: Transformers with Doubly Stochastic Attention](https://arxiv.org/abs/2110.11773), Sander et al. 2021. Attention as Wasserstein gradient flow; doubly stochastic attention.
- [Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth](https://arxiv.org/abs/2103.03404), Dong et al. 2021. Pure attention collapses to rank one doubly exponentially with depth.
- [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887), Bachlechner et al. 2020. Residual scaling at init gives fast, stable deep training.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), Xiong et al. 2020. Pre-LN vs Post-LN gradients and stability.
- [Understanding the Difficulty of Training Transformers](https://arxiv.org/abs/2004.08249), Liu et al. 2020. Amplified output perturbations destabilize Post-LN training.
- [Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies](https://arxiv.org/abs/2010.00951), Rusch & Mishra 2020. RNNs built from coupled nonlinear oscillators.
- [Lipschitz Recurrent Neural Networks](https://arxiv.org/abs/2006.12070), Erichson et al. 2020. Stability via Lipschitz constraints on RNN dynamics.
- [Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View](https://arxiv.org/abs/1906.02762), Lu et al. 2019. Transformer as splitting scheme for convection-diffusion of interacting particles.
- [AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks](https://arxiv.org/abs/1902.09689), Chang et al. 2019. Stable RNNs from ODE stability theory.
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366), Chen et al. 2018. Continuous-depth nets; residual blocks as Euler steps of an ODE.
- [Deep Neural Networks Motivated by Partial Differential Equations](https://arxiv.org/abs/1804.04272), Ruthotto & Haber 2018. PDE-motivated network design (parabolic, hyperbolic).
- [A Mean-Field Optimal Control Formulation of Deep Learning](https://arxiv.org/abs/1807.01083), E et al. 2018. Deep learning as mean-field optimal control.
- [Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks](https://arxiv.org/abs/1806.05393), Xiao et al. 2018. Dynamical isometry lets 10,000-layer CNNs train.
- [Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations](https://arxiv.org/abs/1710.10121), Lu et al. 2017. ResNet-type nets as numerical ODE schemes (Lu et al. 2018).
- [Stable Architectures for Deep Neural Networks](https://arxiv.org/abs/1705.03341), Haber & Ruthotto 2017. Stability of forward propagation in ResNets via ODE theory.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al. 2017. Transformer architecture.
- [Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice](https://arxiv.org/abs/1711.04735), Pennington et al. 2017. Dynamical isometry: Jacobian singular values near 1.
- [Exponential expressivity in deep neural networks through transient chaos](https://arxiv.org/abs/1606.05340), Poole et al. 2016. Order-to-chaos transition in random deep nets.
- [Deep Information Propagation](https://arxiv.org/abs/1611.01232), Schoenholz et al. 2016. Depth scales of signal propagation; edge of chaos for trainability.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He et al. 2015. ResNets; the residual update that makes depth a time variable.

## Clock 2: training as a dynamical system (83)

- [Understanding Optimization in Deep Learning with Central Flows](https://arxiv.org/abs/2410.24206), Cohen et al. 2024. Time-averaged ODE (central flow) for gradient descent at the edge of stability.
- [The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow](https://arxiv.org/abs/2609.01034), Berthier 2026. Multiple-scales derivation of the central flow with three time scales.
- [Edge Flow: A Tractable and Predictive Continuous-Time Model for Gradient Descent at the Edge of Stability](https://arxiv.org/abs/2606.18080), Marion 2026. Three coupled ODEs for center, direction and amplitude of the EoS oscillation.
- [A Bifurcation Theory Framework for Gradient Descent on the Edge of Stability](https://arxiv.org/abs/2606.15551), Gan 2026. Flip bifurcation and center manifold near a manifold of minimizers.
- [On the Stability of Nonlinear Dynamics in GD and SGD: Beyond Quadratic Potentials](https://arxiv.org/abs/2602.14789), Mulayoff & Stich 2026. Exact criterion for stable GD oscillations via third and fourth derivatives.
- [The Origin of Edge of Stability](https://arxiv.org/abs/2604.20446), Litman 2026. Telescoping identity forcing step-direction curvature to 2/eta.
- [Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory](https://arxiv.org/abs/2307.04204), Song & Yun 2023. GD trajectories align on a bifurcation diagram in simple models.
- [Second-order regression models exhibit progressive sharpening to the edge of stability](https://arxiv.org/abs/2210.04860), Agarwala et al. 2022. Progressive sharpening proved in a second-order regression model.
- [Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos](https://arxiv.org/abs/2311.02076), Kalra et al. 2023. Fixed points, EoS and route to chaos in a two-layer linear model.
- [Generalization at the Edge of Stability](https://arxiv.org/abs/2604.19740), Tuci et al. 2026. Optimizers as random dynamical systems with fractal attractors; generalization at EoS.
- [Taming Curvature: Architecture Warm-Up for Stable Transformer Training](https://arxiv.org/abs/2606.16768), Ramasinghe et al. 2026. Curvature control via architecture warm-up prevents transformer loss spikes.
- [Unbiased Gradients, Moving Stability Boundaries: Exact Mini-Batch Geometry in Linear Self-Attention](https://arxiv.org/abs/2605.21292), Balasubramanian 2026. Mini-batch sampling moves stability boundaries in linear self-attention training.
- [The Stability of Singular Distribution: A Spectral Perspective on the Two-Phase Dynamics of Language Model Pre-training](https://arxiv.org/abs/2605.26489), Zhang et al. 2026. Spectral stability explains two-phase (fast then slow) LM pretraining.
- [Learning-Induced Dynamical Transition in Recurrent Neural Networks](https://arxiv.org/abs/2609.19288), Vaidya 2026. Learning drives a chaotic-to-stable transition in RNNs (DMFT).
- [Phase Transitions in Attention: A Bayesian Theory of Copy Head Emergence](https://arxiv.org/abs/2606.12058), Lavie et al. 2026. Phase transitions in attention behind copy-head emergence.
- [A ghost mechanism: An analytical model of abrupt learning in recurrent networks](https://arxiv.org/abs/2501.02378), Dinc et al. 2025. Ghost of a saddle-node bifurcation explains abrupt learning.
- [The boundary of neural network trainability is fractal](https://arxiv.org/abs/2402.06184), Sohl-Dickstein 2024. Boundary between converging and diverging hyperparameters is fractal.
- [Loss Landscape Degeneracy and Stagewise Development in Transformers](https://arxiv.org/abs/2402.02364), Hoogland et al. 2024. Discrete developmental stages of ICL via local learning coefficient.
- [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/abs/2410.05192), Wen et al. 2024. WSD schedules via river-valley landscape.
- [Phase diagram of early training dynamics in deep neural networks: effect of the learning rate, depth, and width](https://arxiv.org/abs/2302.12250), Kalra & Barkeshli 2023. Phase diagram of early training vs LR, depth, width.
- [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217), Nanda et al. 2023. Circuit-level progress measures show grokking is gradual inside.
- [Grokking as the Transition from Lazy to Rich Training Dynamics](https://arxiv.org/abs/2310.06110), Kumar et al. 2023. Grokking as lazy-to-rich transition.
- [Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking](https://arxiv.org/abs/2311.18817), Lyu et al. 2023. Two implicit-bias phases provably cause grokking.
- [Grokking as a First Order Phase Transition in Two Layer Networks](https://arxiv.org/abs/2310.03789), Rubin et al. 2023. Grokking as a first-order phase transition.
- [The Transient Nature of Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2311.08360), Singh et al. 2023. ICL can emerge then vanish during training.
- [The mechanistic basis of data dependence and abrupt learning in an in-context classification task](https://arxiv.org/abs/2312.03002), Reddy 2023. Abrupt ICL learning explained by a minimal model.
- [Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs](https://arxiv.org/abs/2309.07311), Chen et al. 2023. Syntax acquisition as a phase transition with loss plateau.
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322), Wortsman et al. 2023. Attention logit growth and output divergence reproduce at small scale.
- [Stabilizing Transformer Training by Preventing Attention Entropy Collapse](https://arxiv.org/abs/2303.06296), Zhai et al. 2023. Attention entropy collapse causes instability.
- [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871), Molybog et al. 2023. Adam instability mechanism behind loss spikes.
- [Spike No More: Stabilizing the Pre-training of Large Language Models](https://arxiv.org/abs/2312.16903), Takase et al. 2023. Conditions on embeddings/residual scale prevent spikes.
- [The Quantization Model of Neural Scaling](https://arxiv.org/abs/2303.13506), Michaud et al. 2023. Emergence from discrete skill quanta.
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004), Schaeffer et al. 2023. Emergence may be a metric artifact.
- [Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability](https://arxiv.org/abs/2209.15594), Damian et al. 2022. Cubic Taylor term creates a self-stabilizing feedback at EoS.
- [Understanding Gradient Descent on Edge of Stability in Deep Learning](https://arxiv.org/abs/2205.09745), Arora et al. 2022. Mechanism for GD convergence at EoS.
- [Beyond the Edge of Stability via Two-step Gradient Updates](https://arxiv.org/abs/2206.04172), Chen & Bruna 2022. Period-2 orbits of GD beyond EoS.
- [Understanding Edge-of-Stability Training Dynamics with a Minimalist Example](https://arxiv.org/abs/2210.03294), Zhu et al. 2022. Minimal model of EoS with period doubling.
- [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/abs/2207.14484), Cohen et al. 2022. Adam operates at an adaptive edge of stability.
- [Understanding the unstable convergence of gradient descent](https://arxiv.org/abs/2204.01050), Ahn et al. 2022. GD converges while unstable; relation to EoS.
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177), Power et al. 2022. Delayed generalization on algorithmic data.
- [Towards Understanding Grokking: An Effective Theory of Representation Learning](https://arxiv.org/abs/2205.10343), Liu et al. 2022. Grokking via representation dynamics and phase diagrams.
- [Omnigrok: Grokking Beyond Algorithmic Data](https://arxiv.org/abs/2210.01117), Liu et al. 2022. Grokking tied to weight norm; landscape view.
- [The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon](https://arxiv.org/abs/2206.04817), Thilak et al. 2022. Cyclic norm growth and spikes accompany grokking with Adam.
- [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895), Olsson et al. 2022. Abrupt phase change in training when induction heads form.
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311), Chowdhery et al. 2022. Reports loss spikes at scale and restart-skip mitigation.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann et al. 2022. Chinchilla compute-optimal scaling.
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682), Wei et al. 2022. Sharp capability jumps with scale.
- [An Operator Theoretic View on Pruning Deep Neural Networks](https://arxiv.org/abs/2110.14856), Redman et al. 2021. Koopman view of magnitude/gradient pruning.
- [On the difficulty of learning chaotic dynamics with RNNs](https://arxiv.org/abs/2110.07238), Mikhaeil et al. 2021. Training RNNs on chaotic data forces diverging gradients.
- [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065), Cohen et al. 2021. Sharpness rises to 2/eta and hovers; GD is not in the stable regime.
- [A Loss Curvature Perspective on Training Instability in Deep Learning](https://arxiv.org/abs/2110.04369), Gilmer et al. 2021. Curvature explains training instability and warmup.
- [Power-law escape rate of SGD](https://arxiv.org/abs/2105.09557), Mori et al. 2021. Power-law (not exponential) escape rate of SGD.
- [Optimizing Neural Networks via Koopman Operator Theory](https://arxiv.org/abs/2006.02361), Dogra & Redman 2020. Koopman operator theory to accelerate NN training.
- [The large learning rate phase of deep learning: the catapult mechanism](https://arxiv.org/abs/2003.02218), Lewkowycz et al. 2020. Large-LR catapult phase: loss spikes then lands in flatter region.
- [Stochasticity of Deterministic Gradient Descent: Large Learning Rate for Multiscale Objective Function](https://arxiv.org/abs/2002.06189), Kong & Tao 2020. Large-LR GD behaves chaotically and samples like a stochastic process.
- [A Diffusion Theory For Deep Learning Dynamics: Stochastic Gradient Descent Exponentially Favors Flat Minima](https://arxiv.org/abs/2002.03495), Xie et al. 2020. Kramers-type escape: SGD exponentially favors flat minima.
- [Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics](https://arxiv.org/abs/2012.04728), Kunin et al. 2020. Symmetries give conservation laws broken by discretization and noise.
- [Implicit Gradient Regularization](https://arxiv.org/abs/2009.11162), Barrett & Dherin 2020. Backward error analysis of finite-step GD.
- [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522), Yang & Hu 2020. muP: parametrization with feature-learning limit.
- [Prevalence of Neural Collapse during the terminal phase of deep learning training](https://arxiv.org/abs/2008.08186), Papyan et al. 2020. Terminal-phase convergence to a symmetric fixed configuration.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), Kaplan et al. 2020. Power-law scaling of LM loss.
- [Beyond exploding and vanishing gradients: analysing RNN training using attractors and smoothness](https://arxiv.org/abs/1906.08482), Ribeiro et al. 2019. RNN training analyzed via attractors.
- [A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks](https://arxiv.org/abs/1901.06053), Simsekli et al. 2019. Heavy-tailed SGD noise; Levy-driven escape.
- [An Investigation into Neural Net Optimization via Hessian Eigenvalue Density](https://arxiv.org/abs/1901.10159), Ghorbani et al. 2019. Hessian spectral density during training.
- [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671), Frankle et al. 2019. Stability of SGD trajectories to noise; early bifurcation of runs.
- [Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks](https://arxiv.org/abs/1901.06523), Xu et al. 2019. F-principle: low-to-high frequency learning order.
- [The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior of Escaping from Sharp Minima and Regularization Effects](https://arxiv.org/abs/1803.00195), Zhu et al. 2018. Anisotropic SGD noise helps escape sharp minima.
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572), Jacot et al. 2018. Linearized (lazy) training dynamics.
- [A Mean Field View of the Landscape of Two-Layers Neural Networks](https://arxiv.org/abs/1804.06561), Mei et al. 2018. Mean-field PDE for two-layer net training.
- [On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport](https://arxiv.org/abs/1805.09545), Chizat & Bach 2018. Training as Wasserstein gradient flow of particles.
- [Gradient Descent Happens in a Tiny Subspace](https://arxiv.org/abs/1812.04754), Gur-Ari et al. 2018. GD lives in a tiny top-Hessian subspace.
- [On the Spectral Bias of Neural Networks](https://arxiv.org/abs/1806.08734), Rahaman et al. 2018. Low frequencies learned first.
- [How to Escape Saddle Points Efficiently](https://arxiv.org/abs/1703.00887), Jin et al. 2017. Perturbed GD escapes saddles in near dimension-free time.
- [Stochastic Gradient Descent as Approximate Bayesian Inference](https://arxiv.org/abs/1704.04289), Mandt et al. 2017. SGD as an OU process; LR as temperature.
- [Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks](https://arxiv.org/abs/1710.11029), Chaudhari & Soatto 2017. SGD noise is non-isotropic; SGD converges to limit cycles, not points.
- [A Bayesian Perspective on Generalization and Stochastic Gradient Descent](https://arxiv.org/abs/1710.06451), Smith & Le 2017. Noise scale g = eps N/B sets generalization.
- [Three Factors Influencing Minima in SGD](https://arxiv.org/abs/1711.04623), Jastrzębski et al. 2017. LR/batch ratio controls which minima SGD finds.
- [Empirical Analysis of the Hessian of Over-Parametrized Neural Networks](https://arxiv.org/abs/1706.04454), Sagun et al. 2017. Bulk-plus-outliers Hessian spectrum.
- [Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition](https://arxiv.org/abs/1503.02101), Ge et al. 2015. Noisy SGD escapes strict saddles.
- [Stochastic modified equations and adaptive stochastic gradient algorithms](https://arxiv.org/abs/1511.06251), Li et al. 2015. SDE approximations of SGD.
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572), Dauphin et al. 2014. Saddles, not local minima, dominate high-dim landscapes.
- [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120), Saxe et al. 2013. Plateaus and stage-wise mode learning in deep linear nets.
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063), Pascanu et al. 2012. Exploding gradients via bifurcations and attractors.

## Clock 3: generation (and in-context learning) (44)

- [When Do Repetition Loops Lock? Truncation Sampling, Bounded Penalties, and a Blind Spot in DRY](https://arxiv.org/abs/2610.24715), Chen 2026. Exact loop survival from per-step continuation probabilities; when samplers lock loops.
- [Wait, Wait, Wait... Why Do Reasoning Models Loop?](https://arxiv.org/abs/2512.12895), Pipis et al. 2025. Why reasoning models loop at low temperature: learning errors and correlated errors.
- [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858), Keskar et al. 2019. Introduces the repetition penalty used as a decoding baseline.
- [Chaos and bifurcations in continuous language-model feedback](https://arxiv.org/abs/2610.31692), Jones 2026. Continuous LM feedback: temperature moves loops into chaos; Feigenbaum scaling; proven periodic orbits.
- [When Do Repetition Loops Lock? Truncation Sampling, Bounded Penalties, and a Blind Spot in DRY](https://arxiv.org/abs/2610.24715), Chen & Wang 2026. When repetition loops lock under sampling; exit probability per loop.
- [Uncovering Uncontrolled Repetition through Residual Stream Dynamics](https://arxiv.org/abs/2609.38802), Zhang et al. 2026. Repetition traced through residual-stream dynamics.
- [Chaos in reason: How chain-of-thought LLMs can look for an answer](https://arxiv.org/abs/2607.27805), Jaca et al. 2026. Chaos signatures (sensitivity, fractal dimension) in chain-of-thought trajectories.
- [Stochastic Dynamics of Generative AI: From Autoregressive Models to Interacting LLM Networks and Neural SDEs](https://arxiv.org/abs/2611.07065), Krishnamurthy et al. 2026. Stochastic dynamical-systems view of autoregressive LLMs and interacting LLM networks.
- [Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](https://arxiv.org/abs/2502.15208), Wang et al. 2025. Successive paraphrasing converges to 2-cycle attractors.
- [The Effect of Sampling Temperature on Problem Solving in Large Language Models](https://arxiv.org/abs/2402.05201), Renze & Guven 2024. Temperature effect on problem solving.
- [Is Temperature the Creativity Parameter of Large Language Models?](https://arxiv.org/abs/2405.00492), Peeperkorn et al. 2024. Temperature vs novelty and coherence.
- [Phase transition in large language models and the criticality of natural languages](https://arxiv.org/abs/2406.05335), Nakaishi et al. 2024. Temperature-driven phase transition in GPT-2 output statistics.
- [Phase Transitions in the Output Distribution of Large Language Models](https://arxiv.org/abs/2405.17088), Arnold et al. 2024. Detecting phase transitions in LLM output distributions.
- [The pitfalls of next-token prediction](https://arxiv.org/abs/2403.06963), Bachmann & Nagarajan 2024. Teacher forcing vs autoregressive inference failure.
- [LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](https://arxiv.org/abs/2402.00795), Liu et al. 2024. LLMs learn dynamical systems in context.
- [Intelligence at the Edge of Chaos](https://arxiv.org/abs/2410.02536), Zhang et al. 2024. Training on edge-of-chaos cellular automata yields smarter models.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), Xiao et al. 2023. Attention sinks and streaming generation.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), Gu & Dao 2023. Selective state-space models; input-dependent dynamics.
- [Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349), Orvieto et al. 2023. LRU: eigenvalue placement for stable long-range memory.
- [Transformers learn to implement preconditioned gradient descent for in-context learning](https://arxiv.org/abs/2306.00297), Ahn et al. 2023. Linear transformers implement preconditioned GD.
- [Trained Transformers Learn Linear Models In-Context](https://arxiv.org/abs/2306.09927), Zhang et al. 2023. Gradient-flow training dynamics of ICL in linear attention.
- [The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/abs/2305.17493), Shumailov et al. 2023. Model collapse under recursive training.
- [Self-Consuming Generative Models Go MAD](https://arxiv.org/abs/2307.01850), Alemohammad et al. 2023. Self-consuming generative loops degrade (MAD).
- [On the Stability of Iterative Retraining of Generative Models on their own Data](https://arxiv.org/abs/2310.00429), Bertrand et al. 2023. Stability conditions for retraining on own outputs.
- [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452), Kirk et al. 2023. RLHF reduces output diversity.
- [How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534), Zhang et al. 2023. Hallucinations compound along generations.
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654), Dziri et al. 2023. Error compounding in multi-step reasoning.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), Liu et al. 2023. Long-context position effects.
- [Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820), Gruver et al. 2023. LLMs forecast time series in context.
- [Transformers learn in-context by gradient descent](https://arxiv.org/abs/2212.07677), Oswald et al. 2022. Forward pass implements GD steps: depth as optimization time.
- [What learning algorithm is in-context learning? Investigations with linear models](https://arxiv.org/abs/2211.15661), Akyürek et al. 2022. ICL implements known learning algorithms.
- [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://arxiv.org/abs/2208.01066), Garg et al. 2022. ICL of function classes.
- [Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation](https://arxiv.org/abs/2206.02369), Xu et al. 2022. Self-reinforcing repetition: loops strengthen themselves.
- [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191), Hewitt et al. 2022. Truncation sampling theory.
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666), Meister et al. 2022. Information-theoretic decoding.
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), Gu et al. 2021. S4: explicit linear dynamical system layers for sequences.
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174), Schlag et al. 2021. Linear transformers as fast weight programmers.
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236), Katharopoulos et al. 2020. Linear attention as a recurrent state update.
- [A Theoretical Analysis of the Repetition Problem in Text Generation](https://arxiv.org/abs/2012.14660), Fu et al. 2020. Theory of repetition in text generation.
- [Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity](https://arxiv.org/abs/2007.14966), Basu et al. 2020. Feedback control of perplexity; boredom and confusion traps.
- [On Exposure Bias, Hallucination and Domain Shift in Neural Machine Translation](https://arxiv.org/abs/2005.03642), Wang & Sennrich 2020. Exposure bias and hallucination.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), Holtzman et al. 2019. Repetition/blandness under greedy decoding; nucleus sampling.
- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319), Welleck et al. 2019. Training fix for repetition loops.
- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099), Bengio et al. 2015. Exposure bias and train/inference mismatch.

## Foundations and tools (15)

- [Model scale versus domain knowledge in statistical forecasting of chaotic systems](https://arxiv.org/abs/2303.08011), Gilpin 2023. Large models forecasting chaos.
- [Chaos as an interpretable benchmark for forecasting and data-driven modelling](https://arxiv.org/abs/2110.05266), Gilpin 2021. Chaotic systems as forecasting benchmark.
- [Lyapunov spectra of chaotic recurrent neural networks](https://arxiv.org/abs/2006.02427), Engelken et al. 2020. Full Lyapunov spectra of chaotic RNNs.
- [On Lyapunov Exponents for RNNs: Understanding Information Propagation Using Dynamical Systems Tools](https://arxiv.org/abs/2006.14123), Vogt et al. 2020. Lyapunov exponents as a tool to analyze RNNs.
- [Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics](https://arxiv.org/abs/1906.10720), Maheswaranathan et al. 2019. Trained RNNs implement line attractors.
- [Nonlinear Dynamics and Chaos](https://doi.org/10.1201/9780429492563), Strogatz 2018. Primer reference.
- [Model-Free Prediction of Large Spatiotemporally Chaotic Systems](https://doi.org/10.1103/PhysRevLett.120.024102), Pathak et al. 2018. Learned models forecasting chaos.
- [Deep learning for universal linear embeddings of nonlinear dynamics](https://arxiv.org/abs/1712.09707), Lusch et al. 2017. Deep Koopman embeddings of nonlinear dynamics.
- [A Proposal on Machine Learning via Dynamical Systems](https://doi.org/10.1007/s40304-017-0103-z), E 2017. Founding statement of the dynamical view of deep nets.
- [Discovering governing equations from data: Sparse identification of nonlinear dynamical systems](https://arxiv.org/abs/1509.03580), Brunton et al. 2015. SINDy: sparse identification of governing equations.
- [Opening the Black Box](https://doi.org/10.1162/NECO_a_00409), Sussillo & Barak 2013. Fixed-point analysis of trained RNNs.
- [Nonlinear Targeted Energy Transfer in Mechanical and Structural Systems](https://doi.org/10.1007/978-1-4020-9130-8), Vakakis et al. 2009. TET/NES reference for open problem 9.
- [Reaction-rate theory: fifty years after Kramers](https://doi.org/10.1103/RevModPhys.62.251), Hanggi, Talkner & Borkovec 1990. Standard review of escape rates.
- [Chaos in Random Neural Networks](https://doi.org/10.1103/PhysRevLett.61.259), Sompolinsky, Crisanti & Sommers 1988. Order-chaos transition in random networks; ancestor of edge-of-chaos work.
- [Brownian motion in a field of force](https://doi.org/10.1016/S0031-8914(40)90098-2), Kramers 1940. Noise-driven escape over a barrier; the escape-rate law.
