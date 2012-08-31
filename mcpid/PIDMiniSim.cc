#include <RAT/GLG4VEventAction.hh>
#include <RAT/GLG4HitPMTCollection.hh>
#include <G4ParticleTable.hh>
#include <G4PrimaryVertex.hh>
#include <G4PrimaryParticle.hh>
#include <Randomize.hh>
#include <G4Event.hh>

#include "PIDMiniSim.hh"

namespace RAT {

PIDMiniSim::PIDMiniSim(const std::string pid, const G4ThreeVector &center, double energy)
  : fParticleType(pid), fCenter(center), fEnergy(energy), fEventNumber(0), fNevents(0) { }

PIDMiniSim::~PIDMiniSim() { }

void PIDMiniSim::Run(const int nevents) {
  fNevents = nevents;
  const int nchannels = 19 * 16 * 32;
  fHitCount.clear();
  fHitCount.resize(nchannels, 0);
  BeamOn(nevents);
}

std::vector<float> PIDMiniSim::GetHitProbability() {
  std::vector<float> hitProbability(fHitCount.size());
  for (size_t i=0; i<hitProbability.size(); i++) {
    hitProbability[i] = 1.0 * fHitCount[i] / fNevents;
  }

  return hitProbability;
}

G4ThreeVector GetRandomDirection() {
    double px = (G4UniformRand() * 2.0 - 1.0);
    double py = (G4UniformRand() * 2.0 - 1.0);
    double pz = (G4UniformRand() * 2.0 - 1.0);
    return G4ThreeVector(px, py, pz);
}

G4ThreeVector GetRandomPolarization(const G4ThreeVector& mom) {
    double phi = (G4UniformRand() * 2.0 - 1.0) * pi;
    G4ThreeVector e1 = mom.orthogonal().unit();
    G4ThreeVector e2 = mom.unit().cross(e1);
    G4ThreeVector pol = e1*cos(phi)+e2*sin(phi);
    return pol;
}

void PIDMiniSim::GeneratePrimaries(G4Event* event) {
  double dt = 0.0;
  G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");

  if (fParticleType == "e-") {
    G4ThreeVector dx = fCenter;

    // random momentum
    G4ThreeVector mom = GetRandomDirection();
    mom *= sqrt(fEnergy * (fEnergy + 2.0 * particle->GetPDGMass()));

    G4PrimaryVertex* vertex = new G4PrimaryVertex(dx, dt);
    G4PrimaryParticle* beta = new G4PrimaryParticle(particle, mom.x(), mom.y(), mom.z());

    // random polarization
    G4ThreeVector pol = GetRandomPolarization(mom);
    beta->SetPolarization(pol.x(), pol.y(), pol.z());

    vertex->SetPrimary(beta);
    event->AddPrimaryVertex(vertex);
  }
  else if (fParticleType == "gamma") {
    G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");
    G4ThreeVector x1 = fCenter - G4ThreeVector(150.0, 0.0, 0.0);
    G4ThreeVector x2 = fCenter + G4ThreeVector(150.0, 0.0, 0.0);

    float energy = fEnergy / 2;

    // random momentum
    G4ThreeVector mom1 = GetRandomDirection();
    mom1 *= sqrt(energy * (energy + 2.0 * particle->GetPDGMass()));
    G4PrimaryVertex* vertex1 = new G4PrimaryVertex(x1, dt);
    G4PrimaryParticle* beta1 = new G4PrimaryParticle(particle, mom1.x(), mom1.y(), mom1.z());

    G4ThreeVector mom2 = GetRandomDirection();
    mom2 *= sqrt(energy * (energy + 2.0 * particle->GetPDGMass()));
    G4PrimaryVertex* vertex2 = new G4PrimaryVertex(x2, dt);
    G4PrimaryParticle* beta2 = new G4PrimaryParticle(particle, mom2.x(), mom2.y(), mom2.z());

    // random polarization
    G4ThreeVector pol1 = GetRandomPolarization(mom1);
    beta1->SetPolarization(pol1.x(), pol1.y(), pol1.z());

    G4ThreeVector pol2 = GetRandomPolarization(mom2);
    beta2->SetPolarization(pol2.x(), pol2.y(), pol2.z());

    vertex1->SetPrimary(beta1);
    event->AddPrimaryVertex(vertex1);
    vertex2->SetPrimary(beta2);
    event->AddPrimaryVertex(vertex2);
  }
}

void PIDMiniSim::BeginOfEventAction(const G4Event* /*event*/)
{
  GLG4VEventAction::GetTheHitPMTCollection()->Clear();
}

void PIDMiniSim::EndOfEventAction(const G4Event* /*event*/)
{
  GLG4HitPMTCollection* hitpmts = GLG4VEventAction::GetTheHitPMTCollection();

  int nhits = hitpmts->GetEntries();
  std::cout << "PIDMiniSim: Event " << fEventNumber << ", nhit " << nhits << std::endl;

  for (int i=0; i<hitpmts->GetEntries(); i++) {
    GLG4HitPMT* pmt = hitpmts->GetPMT(i);
    int id = pmt->GetID();
    fHitCount.at(id)++;
  }

  fEventNumber++;
}

} // namespace RAT

