# To enable C++ Torch DQN, you will need to set BUILD_WITH_LIBTORCH. 
if (BUILD_WITH_LIBTORCH)
  add_library (dqn_pytorch OBJECT
    # dqn_pytorch.h
    # dqn_pytorch.cc
    simple_nets.h
    simple_nets.cc
  )
  target_include_directories (dqn_pytorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(simple_nets_test simple_nets_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:dqn_pytorch> $<TARGET_OBJECTS:tests>)
  add_test(simple_nets_test simple_nets_test)

  # add_executable(dqn_pytorch_test dqn_pytorch_test.cc ${OPEN_SPIEL_OBJECTS}
  #                $<TARGET_OBJECTS:dqn_pytorch> $<TARGET_OBJECTS:tests>)
  # add_test(dqn_pytorch_test dqn_pytorch_test)

  target_link_libraries(simple_nets_test ${TORCH_LIBRARIES})
  # target_link_libraries(dqn_pytorch_test ${TORCH_LIBRARIES})
endif()
