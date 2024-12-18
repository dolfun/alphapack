add_library(mcts INTERFACE mcts.h mcts_async.h generate_episode.h)
target_compile_features(mcts INTERFACE cxx_std_20)