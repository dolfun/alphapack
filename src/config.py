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

def get_config(iter):
  config = Config(
    seed=23894734,
    pool_size=2048,
    episodes_per_iteration=1152,
    processes=6,
    step_size=96,
    workers_per_process=32,
    move_threshold=4,
    simulations_per_move=512,
    mcts_thread_count=8,
    batch_size=128,
    c_puct=1.25,
    virtual_loss=1,
    alpha=0.3
  )

  if iter < 0:
    config.episodes_per_iteration //= 2
    config.step_size //= 2
    config.simulations_per_move = 2048
    config.move_threshold = 0
    config.alpha = -1
    return config

  if iter <= 4:
    config.simulations_per_move = 400
  elif iter <= 12:
    config.simulations_per_move = 800
  elif iter <= 24:
    config.simulations_per_move = 1600
  else:
    config.simulations_per_move = 2400

  return config

@dataclass
class TrainConfig:
  epochs: int
  lr: float
  weight_decay: float

def get_train_config(_):
  config = TrainConfig(
    epochs=4,
    lr=4e-3,
    weight_decay=1e-2
  )

  return config