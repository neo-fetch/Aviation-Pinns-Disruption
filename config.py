"""Central configuration for the Airbus supply chain PI-GNN proof of concept.

All tunables live here so experiments stay reproducible: network size is
implied by the builder in src/pignn/network.py; everything temporal,
architectural, and physics-related is set below.
"""

SEED = 42

# ---------------------------------------------------------------- simulation
N_WEEKS = 156                # 3 years of weekly snapshots
N_DISRUPTION_EPISODES = 60   # injected disruption scenarios over the horizon
BASE_STOCK_WEEKS = 3.0       # target inventory cover (weeks of demand)

# Severity thresholds on realized capacity reduction, from the paper (§4.1):
# none <10%, minor 10-30% is split as minor <10%? — paper: minor <10% reduction,
# moderate 10-30%, major >30%. We use: none = 0, minor = (0, 0.10],
# moderate = (0.10, 0.30], major = > 0.30.
SEVERITY_MINOR = 0.02
SEVERITY_MODERATE = 0.10
SEVERITY_MAJOR = 0.30

# ------------------------------------------------------------------- dataset
T_IN = 6                     # input window length (paper §3.3, Figure 3)
HORIZONS = (1, 2, 4, 8)      # forecast horizons in weeks
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2               # remaining 0.2 is test (temporal split, paper §4.1)

# --------------------------------------------------------------------- model
NODE_EMBED_DIM = 64
GNN_LAYERS = 2
LSTM_HIDDEN = 96
N_CLASSES = 4                # none / minor / moderate / major

# ------------------------------------------------------------------ training
EPOCHS = 120
WARMUP_EPOCHS = 30           # prediction-only pretraining (paper §3.4)
LR = 2e-3
WEIGHT_DECAY = 1e-5
BATCH_WINDOWS = 16           # temporal windows per mini-batch

# Physics constraint weights (final values; curriculum ramps toward these)
LAMBDA_FLOW = 0.05
LAMBDA_CAPACITY = 0.05
LAMBDA_LEAD = 0.02
CURRICULUM_TAU = 20.0        # lambda(e) = lambda_final * (1 - exp(-(e - warmup)/tau))

# ---------------------------------------------------------------- evaluation
DATA_EFFICIENCY_FRACTIONS = (1.0, 0.5, 0.25)

OUTPUT_DIR = "outputs"
