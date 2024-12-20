add_library(mcts INTERFACE mcts_common.h mcts.h generate_episode.h)
target_compile_features(mcts INTERFACE cxx_std_20)