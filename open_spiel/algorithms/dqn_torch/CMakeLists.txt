# To enable C++ Torch DQN, you will need to set BUILD_WITH_LIBTORCH. 
if (BUILD_WITH_LIBTORCH)
  add_library (dqn_torch OBJECT
    dqn_torch.h
    dqn_torch.cc
    simple_nets.h
    simple_nets.cc
  )
  target_include_directories (dqn_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(simple_nets_test simple_nets_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:dqn_torch> $<TARGET_OBJECTS:tests>)
  add_test(simple_nets_test simple_nets_test)

  add_executable(dqn_torch_test dqn_torch_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:dqn_torch> $<TARGET_OBJECTS:tests>)
  add_test(dqn_torch_test dqn_torch_test)

  target_link_libraries(simple_nets_test ${TORCH_LIBRARIES})
  target_link_libraries(dqn_torch_test ${TORCH_LIBRARIES})
endif()
