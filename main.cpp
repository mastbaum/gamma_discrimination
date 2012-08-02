#include <RAT/DSReader.hh>
#include <RAT/DS/Root.hh>
#include <RAT/DS/EV.hh>

#include <TFile.h>
#include <TH1F.h>
#include <TVector3.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>

#include <cuda_runtime.h>

#include "pmtpos.h"

extern "C" std::map<std::string, float> calculate_betas(float3 &fit_position, float3* hit_pos, unsigned nhits);

int main(int argc, char* argv[]) {
    std::map<std::string, TH1F*> h;
    h["1"] = new TH1F("hbeta_1", "hbeta_1", 100, 1, 2);
    h["2"] = new TH1F("hbeta_2", "hbeta_2", 100, 3.5, 4.5);
    h["3"] = new TH1F("hbeta_3", "hbeta_3", 100, 12.2, 13.2);
    h["4"] = new TH1F("hbeta_4", "hbeta_4", 100, 47.9, 48.9);
    h["14"] = new TH1F("hbeta14", "hbeta14", 200, 194.0, 196.0);

    TFile f("betas_e.root", "recreate");
    f.cd();

    const unsigned jobs = 10;
    for (unsigned jobnum=0; jobnum<jobs; jobnum++) {
        char filename[100];
        snprintf(filename, 100, "data/e_%i.root", jobnum);
        RAT::DSReader r(filename);
        unsigned count = 0;
        while(RAT::DS::Root* ds = r.NextEvent()) {
            for (int iev=0; iev<1; iev++) {
                RAT::DS::EV* ev = ds->GetEV(iev);
                printf("Processing job %u, event %u (GTID %x)\n", jobnum, count, ev->GetEventID());

                // get fit position
                TVector3 v = ev->GetFitResult("scintFitter").GetVertex(0).GetPosition();
                float3 fit_position = make_float3((float)v[0], (float)v[1], (float)v[2]);

                // load positions of hit pmts
                size_t nhits = ev->GetNhits();
                float3* hit_pos = (float3*) malloc(nhits * sizeof(float3));
                for (size_t ipmt=0; ipmt<nhits; ipmt++) {
                    int pmtid = ev->GetPMTUnCal(ipmt)->GetID();
                    hit_pos[ipmt] = make_float3(pmtpos::x[ipmt], pmtpos::y[ipmt], pmtpos::z[ipmt]);
                }

                std::map<std::string, float> betas = calculate_betas(fit_position, hit_pos, nhits);

                std::map<std::string, float>::iterator it;
                for (it=betas.begin(); it!=betas.end(); it++) {
                    printf("%s: %f\n", (*it).first.c_str(), (*it).second);
                }

                count++;           
            }
        }
    }

    return 0;
}

