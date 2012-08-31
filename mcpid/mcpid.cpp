#include <iostream>
#include <sys/time.h>

#include <RAT/DS/Root.hh>
#include <RAT/DS/EV.hh>
#include <RAT/DSReader.hh>
#include <RAT/DB.hh>

#include <RAT/G4Stream.hh>
#include <RAT/RunManager.hh>
#include <RAT/GLG4param.hh>
#include <G4RunManager.hh>

#include "PIDMiniSim.hh"

static int events_per_iteration = 100;

int main(int argc, char* argv[]) {
    // quiet down G4 output
    SetG4coutStream(G4Stream::DETAIL);
    SetG4cerrStream(G4Stream::WARN);

    // seed random number generator, same way as rat
    time_t start_time = time(NULL);
    pid_t pid = getpid();
    long seed = start_time ^ (pid << 16);
    CLHEP::HepRandom::setTheSeed(seed);
    std::cout << "Random seed: " << seed << std::endl;

    // load ratdb tables
    std::cout << "Load DB..." << std::endl;
    RAT::DB* ratdb = RAT::DB::Get();
    ratdb->LoadAll(std::string("/home/mastbaum/snoplus/minisim/sp/data"));

    // disable muon and hadronic processes
    GLG4param& db(GLG4param::GetDB());
    db["omit_muon_processes"] = 1.0;
    db["omit_hadronic_processes"] = 1.0;

    // set up the geant4 run environment
    // RAT's RunManager does most of the work for us
    RAT::RunManager* run_manager = new RAT::RunManager;
    RAT::gTheRunManager = run_manager;
    G4RunManager* g4_run_manager = G4RunManager::GetRunManager();
    g4_run_manager->Initialize();

    G4ThreeVector origin(0,0,0);

    RAT::DSReader r(argv[1]);
    RAT::DS::Root* ds = NULL;

    FILE* fout = NULL;

    int count = 0;
    while((ds = r.NextEvent()) && (count < 10)) {
        RAT::DS::EV* ev = ds->GetEV(0);

        // run betas
        RAT::PIDMiniSim* ms = new RAT::PIDMiniSim("e-", origin, 3.5*MeV);
        ms->Run(10000);
        std::vector<float> ehp = ms->GetHitProbability();
        delete ms;

        // run gammas
        ms = new RAT::PIDMiniSim("gamma", origin, 3.5*MeV);
        ms->Run(10000);
        std::vector<float> ghp = ms->GetHitProbability();
        delete ms;


        float lbeta = 0;
        float lgamma = 0;
        for (int i=0; i<ev->GetPMTUnCalCount(); i++) {
            int id = ev->GetPMTUnCal(i)->GetID();
            lbeta += ehp[id];
            lgamma += ghp[id];
        }

        float lratio = -2.0 * log(lbeta / lgamma);
        std::cout << "Lratio = " << lratio << std::endl;

        char fn[100];
        sprintf(fn, "likelihoods_%s_10k_%s.txt", argv[2], argv[3]);
        fout = fopen(fn, "a");
        char s[100];
        sprintf(s, "%f %f\n", lbeta, lgamma);
        fputs(s, fout);
        fclose(fout);

        count++;
    }

    delete ratdb;
    delete g4_run_manager;
}

