from dataclasses import dataclass

@dataclass
class Config:
  seed: int
  pool_size: int
  episodes_per_iteration: int
  processes: int
  step_size: int
  workers_per_process: int
  move_threshold: int
  simulations_per_move: int
  mcts_thread_count: int
  batch_size: int
  c_puct: float
  virtual_loss: int
  alpha: float

@dataclass
class TrainConfig:
  epochs: int
  lr: float
  weight_decay: float

def get_config(iter):
  config = Config(
    seed=1233232,
    pool_size=256,
    episodes_per_iteration=1152,
    processes=6,
    step_size=96,
    workers_per_process=32,
    move_threshold=0,
    simulations_per_move=512,
    mcts_thread_count=8,
    batch_size=128,
    c_puct=1.25,
    virtual_loss=1,
    alpha=0.1
  )

  if iter < 0:
    config.alpha = -1
    return config

  if iter <= 8:
    config.simulations_per_move = 512
  elif iter <= 16:
    config.simulations_per_move = 1024
  else:
    config.simulations_per_move = 2048

  return config

def get_train_config(_):
  config = TrainConfig(
    epochs=8,
    lr=1e-3,
    weight_decay=1e-2
  )

  return config