#ifndef __RAT_PIDMiniSim__
#define __RAT_PIDMiniSim__

#include <string>
#include <G4ThreeVector.hh>
#include <RAT/MiniSim.hh>

class G4Event;

namespace RAT {

class PIDMiniSim : public MiniSim {
public:
  PIDMiniSim(const std::string pid, const G4ThreeVector& center, double energy);
  virtual ~PIDMiniSim();

  void Run(const int nevents);

  std::vector<float> GetHitProbability();

  virtual void GeneratePrimaries(G4Event* event);
  virtual void BeginOfEventAction(const G4Event* event);
  virtual void EndOfEventAction(const G4Event* event);

protected:
  std::string fParticleType;
  G4ThreeVector fCenter;
  double fEnergy;
  int fNevents;
  int fEventNumber;

  std::vector<unsigned> fHitCount;
};

} // namespace RAT

#endif // __RAT_PIDMiniSim__

