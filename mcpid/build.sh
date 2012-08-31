g++ -o mcpid mcpid.cpp PIDMiniSim.cc -I. -I$RATROOT/src/geo -I$RATROOT/src/stlplus $RATROOT/src/physics/PhotonThinning.cc -I$RATROOT/include -I$RATROOT/src/core -I$RATROOT/src/util -I$RATROOT/src/physics -L$RATROOT/lib `clhep-config --libs` `geant4-config --libs --cflags` `root-config --libs --cflags` -lMinuit -lRATEvent_Linux-g++ -lrat_Linux-g++ -g

